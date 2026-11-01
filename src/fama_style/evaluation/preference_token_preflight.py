"""CPU-only tokenizer/mask and synthetic loss-contract checks; never load weights."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys

PROTOCOL='actual_error_token_preflight_v1'
MODEL_ID='google/gemma-3-4b-it'
REVISION='093f9f388b31de276ce2de164bdc2081324b9767'
ASSETS=('config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','added_tokens.json','chat_template.json')
PACKAGES={'transformers':'5.20.1','huggingface_hub':'1.33.0'}


def sha(text):return hashlib.sha256(text.encode()).hexdigest()
def file_sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write_json(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def norm(text):return ' '.join(text.split())


def load_inputs(data_path,config_path):
    config=json.loads(config_path.read_text())
    if (config['protocol']!=PROTOCOL or config['model_id']!=MODEL_ID or config['model_revision']!=REVISION
            or config['max_length']!=1024 or config['training_authorized'] is not False
            or config['packages']!=PACKAGES or config['data_sha256']!=file_sha(data_path)):
        raise ValueError('frozen_token_preflight_config_changed')
    rows=[json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
    if len(rows)!=17 or len({r['pair_id'] for r in rows})!=17 or len({r['anchor_group_id'] for r in rows})!=11:
        raise ValueError('frozen_seventeen_pairs_eleven_groups_required')
    group=Counter(r['anchor_group_id'] for r in rows)
    for r in rows:
        if (r['split']!='training' or any(not r[k].strip() or sha(r[k])!=r[k+'_sha256'] for k in ('prompt','chosen','rejected'))
                or norm(r['chosen'])==norm(r['rejected']) or r['anchor_balanced_weight']!=1/group[r['anchor_group_id']]
                or r['archived_chosen']['decoded_target_matches'] is not True):
            raise ValueError('pair_text_group_weight_or_archive_changed')
    return rows,config


def mask(prefix,full,decoded,target,limit,end_turn_id,bos_id):
    if (not prefix or full[:len(prefix)]!=prefix or len(full)<=len(prefix) or len(full)>limit
            or full[0]!=bos_id or full.count(bos_id)!=1 or norm(decoded)!=norm(target)):
        raise ValueError('prefix_boundary_bos_decode_or_length_failed')
    completion=full[len(prefix):]
    if completion.count(end_turn_id)!=1:raise ValueError('exactly_one_template_end_of_turn_required')
    return [-100]*len(prefix)+completion


def left_pad(sequences,pad_id):
    width=max(len(s['input_ids']) for s in sequences);output=[]
    for s in sequences:
        n=width-len(s['input_ids'])
        output.append({'input_ids':[pad_id]*n+s['input_ids'],'attention_mask':[0]*n+[1]*len(s['input_ids']),
            'labels':[-100]*n+s['labels']})
    return output


def causal_response_score(ids,labels,next_token_logps):
    if len(ids)!=len(labels) or len(next_token_logps)!=len(ids)-1:raise ValueError('causal_score_shape_failed')
    values=[]
    for i in range(1,len(ids)):
        if labels[i]==-100:continue
        if labels[i]!=ids[i] or not math.isfinite(next_token_logps[i-1]) or next_token_logps[i-1]>0:
            raise ValueError('causal_label_or_log_probability_failed')
        values.append(next_token_logps[i-1])
    if not values:raise ValueError('positive_supervised_token_count_required')
    total=sum(values)
    return {'total_log_probability':total,'mean_nll':-total/len(values),'supervised_tokens':len(values)}


def dpo_loss(policy_chosen,policy_rejected,reference_chosen,reference_rejected,beta=0.1):
    values=(policy_chosen,policy_rejected,reference_chosen,reference_rejected)
    if not math.isfinite(beta) or beta<=0 or any(not math.isfinite(v) or v>0 for v in values):
        raise ValueError('finite_log_probabilities_and_positive_beta_required')
    z=beta*((policy_chosen-policy_rejected)-(reference_chosen-reference_rejected))
    return max(0.,-z)+math.log1p(math.exp(-abs(z)))


def encode_chat(tokenizer,messages,generation):
    # Transformers 5 defaults to a BatchEncoding. Masking uses flat token IDs.
    ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=generation,return_dict=False)
    if not isinstance(ids,list) or not ids or any(type(i) is not int or i<0 for i in ids):
        raise ValueError('flat_nonnegative_integer_token_ids_required')
    return ids


def audit_pair(row,encode,decode,end_turn_id,bos_id,pad_id,limit=1024):
    user=[{'role':'user','content':[{'type':'text','text':row['prompt']}]}]
    prefix=encode(user,True);encoded={};audits=[]
    for candidate in ('chosen','rejected'):
        target=row[candidate];full=encode(user+[{'role':'assistant','content':[{'type':'text','text':target}]}],False)
        labels=mask(prefix,full,decode(full[len(prefix):]),target,limit,end_turn_id,bos_id)
        if candidate=='chosen':
            expected=row['archived_chosen']
            if (len(prefix)!=expected['prompt_tokens'] or len(full)!=expected['full_tokens']
                    or len(full)-len(prefix)!=expected['supervised_tokens'] or sha(json.dumps(full))!=expected['input_ids_sha256']):
                raise ValueError('archived_processor_chosen_ids_not_reproduced')
        encoded[candidate]={'input_ids':full,'labels':labels}
        audits.append({'pair_id':row['pair_id'],'candidate':candidate,'prompt_tokens':len(prefix),'full_tokens':len(full),
            'supervised_tokens':len(full)-len(prefix),'input_ids_sha256':sha(json.dumps(full)),
            'labels_sha256':sha(json.dumps(labels)),'decoded_response_matches':True,'template_end_of_turn_supervised':True})
    if encoded['chosen']['input_ids']==encoded['rejected']['input_ids']:raise ValueError('preference_token_sequences_identical')
    padded=left_pad([encoded['chosen'],encoded['rejected']],pad_id)
    for candidate,seq,original in zip(('chosen','rejected'),padded,encoded.values()):
        if ([v for v in seq['labels'] if v!=-100]!=[v for v in original['labels'] if v!=-100]
                or any(label!=-100 for label,attention in zip(seq['labels'],seq['attention_mask']) if attention==0)):
            raise ValueError('padding_changed_response_supervision')
        # Constant synthetic observed-token log probabilities check causal indexing;
        # they are not model outputs or a measured likelihood distribution.
        fake=[-math.log(7)]*(len(seq['input_ids'])-1)
        value=causal_response_score(seq['input_ids'],seq['labels'],fake)
        if not math.isclose(value['mean_nll'],math.log(7),abs_tol=1e-10):raise ValueError('synthetic_scoring_contract_failed')
    return {'pair_id':row['pair_id'],'anchor_group_id':row['anchor_group_id'],
        'anchor_balanced_weight':row['anchor_balanced_weight'],'chosen':encoded['chosen'],'rejected':encoded['rejected']},audits


def run_training(rows,config,output,tokenizer_dir=None):
    """Repository lazy-import convention; CPU tokenizer work only, no training."""
    if output.exists():raise ValueError('fresh_preflight_output_required')
    output.mkdir(parents=True)
    state={'complete':False,'phase':'start','pairs_completed':0,'training_called':False,'model_weights_loaded':False,'gpu_used':False}
    write_json(output/'state.json',state)
    try:
        if 'torch' in sys.modules:raise RuntimeError('Fresh CPU process required before tokenizer-only check')
        os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['USE_TORCH']='0';os.environ['USE_TF']='0';os.environ['USE_FLAX']='0'
        from transformers import AutoTokenizer
        state['phase']='tokenizer_assets';write_json(output/'state.json',state)
        packages={name:importlib.metadata.version(name) for name in PACKAGES}
        if packages!=PACKAGES:raise RuntimeError('Pinned tokenizer package versions required')
        assets=output/'tokenizer_assets';assets.mkdir()
        if tokenizer_dir is None:
            from huggingface_hub import HfApi,hf_hub_download
            try:
                from kaggle_secrets import UserSecretsClient
                token=UserSecretsClient().get_secret('HF_TOKEN')
            except Exception:raise RuntimeError('Enable HF_TOKEN in Kaggle Secrets') from None
            if not isinstance(token,str) or not token.strip():raise RuntimeError('Enable HF_TOKEN in Kaggle Secrets')
            if HfApi().model_info(MODEL_ID,revision=REVISION,token=token,timeout=60).sha!=REVISION:
                raise RuntimeError('Pinned tokenizer revision changed')
            for name in ASSETS:
                path=hf_hub_download(MODEL_ID,name,revision=REVISION,token=token)
                shutil.copyfile(path,assets/name)
            token=None
            write_json(assets/'asset_manifest.json',{'model_id':MODEL_ID,'model_revision':REVISION,
                'files':{name:file_sha(assets/name) for name in ASSETS},'contains_model_weights':False})
        else:
            manifest=json.loads((tokenizer_dir/'asset_manifest.json').read_text())
            if (manifest['model_id']!=MODEL_ID or manifest['model_revision']!=REVISION or set(manifest['files'])!=set(ASSETS)
                    or any(file_sha(tokenizer_dir/name)!=digest for name,digest in manifest['files'].items())):
                raise ValueError('Exported_tokenizer_bundle_changed')
            for name in (*ASSETS,'asset_manifest.json'):shutil.copyfile(tokenizer_dir/name,assets/name)
        tokenizer=AutoTokenizer.from_pretrained(str(assets),local_files_only=True,use_fast=True,trust_remote_code=False)
        template=json.loads((assets/'chat_template.json').read_text())['chat_template']
        tokenizer.chat_template=template
        eot=tokenizer.convert_tokens_to_ids('<end_of_turn>')
        if eot is None or eot==tokenizer.unk_token_id:raise ValueError('Gemma_end_of_turn_token_missing')
        pad=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        def encode(messages,generation):
            return encode_chat(tokenizer,messages,generation)
        def decode(ids):return tokenizer.decode(ids,skip_special_tokens=True)
        state['phase']='tokenizer_masks';audits=[];packed=[]
        for row in rows:
            state['current_pair_id']=row['pair_id'];write_json(output/'state.json',state)
            encoded,checks=audit_pair(row,encode,decode,eot,tokenizer.bos_token_id,pad,config['max_length'])
            packed.append(encoded);audits.extend(checks);state['pairs_completed']+=1
            write_json(output/'token_audit.json',audits)
            print('Checked %d/17 preference pairs'%state['pairs_completed'],flush=True)
        if not math.isclose(dpo_loss(-5,-8,-5,-8),math.log(2),abs_tol=1e-12):raise ValueError('dpo_initial_identity_failed')
        if 'torch' in sys.modules:raise RuntimeError('Unexpected model backend import')
        (output/'encoded_pairs.private.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in packed))
        summary={'protocol':PROTOCOL,'complete':True,'pairs':17,'response_sequences':34,'archived_chosen_sequences_reproduced':17,
            'max_full_tokens':max(r['full_tokens'] for r in audits),'min_supervised_tokens':min(r['supervised_tokens'] for r in audits),
            'response_padding_and_causal_scoring_contract_passed':True,'synthetic_dpo_identity_passed':True,
            'actual_model_log_probabilities_computed':False,'training_called':False,'optimizer_steps':0,'model_weights_loaded':False,
            'gpu_used':False,'training_authorized':False,'gpu_training_ready':False,'packages':packages,
            'tokenizer_files_sha256':{name:file_sha(assets/name) for name in ASSETS},'runtime_source_sha256':file_sha(Path(__file__))}
        write_json(output/'summary.json',summary);state.update(complete=True,phase='complete');write_json(output/'state.json',state)
    except Exception as error:
        state.update(phase='failed',error_type=type(error).__name__);write_json(output/'state.json',state)
        raise RuntimeError('CPU tokenizer preflight failed; preserve state and partial audits. Error type: '+type(error).__name__) from None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--check',action='store_true');p.add_argument('--run-tokenizer-check',action='store_true')
    p.add_argument('--tokenizer-dir',type=Path);p.add_argument('--output',type=Path,default=Path('/kaggle/working/actual_error_preflight_v1'))
    a=p.parse_args();rows,config=load_inputs(a.data,a.config)
    if a.check:print('Validated 17 candidate rows, 11 groups; no tokenizer, model, GPU or training.');return
    if not a.run_tokenizer_check:raise ValueError('Explicit tokenizer-only execution flag required')
    run_training(rows,config,a.output,a.tokenizer_dir)


if __name__=='__main__':main()
