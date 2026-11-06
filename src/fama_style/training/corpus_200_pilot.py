"""Isolated one-epoch, 200-pair QLoRA pilot and matched development inference.

CPU validation uses only the standard library. All ML imports remain inside
run_training; the root Gemma 1B CLI and historical experiments are unchanged.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import time

PROTOCOL='corpus_200_pilot_v1'
MODEL_ID='google/gemma-3-4b-it'
REVISION='093f9f388b31de276ce2de164bdc2081324b9767'
GROUPS=('Fama_coauthors','Shiller','Holmstrom','Baker_Wurgler')
INSTRUCTION='''Rewrite the passage as one paragraph of natural academic economics prose.
Retain its substantive claims, actors, quantities, comparisons, conditions,
exceptions, uncertainty, negation, attribution and causal direction. Improve
expression and organization where useful, without adding economic facts or
relationships. Do not fill gaps using outside knowledge. Return only the paragraph.'''
HYPER={'epochs':1,'learning_rate':2e-5,'rank':8,'alpha':16,'dropout':0.05,
       'batch_size':1,'gradient_accumulation':2,'optimizer_steps':100,
       'max_length':1536,'seed':1729,'weight_decay':0.0,'max_grad_norm':1.0}
DECODING={'do_sample':False,'num_beams':1,'max_new_tokens':512,'repetition_penalty':1.0}
PACKAGES={'transformers':'5.20.1','peft':'0.21.2','accelerate':'1.15.0',
          'bitsandbytes':'0.50.2','huggingface_hub':'1.33.0'}
TARGET_PATTERN=r'.*language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)$'
MAX_SECONDS=5400


def sha(text):return hashlib.sha256(text.encode()).hexdigest()
def norm(text):return ' '.join(text.split())
def write_json(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def prompt(source):return INSTRUCTION+'\n\nPassage:\n'+source


def load_inputs(data_path,config_path):
    config=json.loads(config_path.read_text())
    fixed={'protocol':PROTOCOL,'model_id':MODEL_ID,'model_revision':REVISION,
           'hyperparameters':HYPER,'decoding':DECODING,'packages':PACKAGES,
           'max_runtime_seconds':MAX_SECONDS,'training_authorized':True,
           'quantization':'NF4 double quantization; FP32 compute',
           'runtime_source_sha256':sha(Path(__file__).read_text()),
           'dataset_sha256':sha(data_path.read_text())}
    if any(config.get(k)!=v for k,v in fixed.items()):raise ValueError('frozen_pilot_settings_or_files_changed')
    rows=[json.loads(l) for l in data_path.read_text().splitlines() if l.strip()]
    splits={'training':[],'development':[]};seen=set();targets=set();parents=set()
    fields={'example_id','parent_id','paper_id','style_group','authors','split',
            'source','target','source_sha256','target_sha256','prompt','pilot_eligible'}
    for row in rows:
        if set(row)!=fields or row['split'] not in splits or row['pilot_eligible'] is not True:
            raise ValueError('unapproved_or_reserved_row')
        if row['example_id'] in seen or row['parent_id'] in parents:raise ValueError('duplicate_example_or_parent')
        seen.add(row['example_id']);parents.add(row['parent_id'])
        if row['style_group'] not in GROUPS or not row['authors']:raise ValueError('group_or_authors_missing')
        if any(not isinstance(row[k],str) or not row[k].strip() for k in ('source','target')):
            raise ValueError('missing_text')
        if row['source_sha256']!=sha(row['source']) or row['target_sha256']!=sha(row['target']) or row['prompt']!=prompt(row['source']):
            raise ValueError('text_or_prompt_changed')
        target=norm(row['target']).casefold()
        if target in targets:raise ValueError('duplicate_target')
        targets.add(target);splits[row['split']].append(row)
    if [len(splits[k]) for k in splits]!=[200,16]:raise ValueError('expected_200_training_16_development')
    if Counter(r['style_group'] for r in splits['training'])!=Counter({g:50 for g in GROUPS}):
        raise ValueError('balanced_200_training_required')
    if Counter(r['style_group'] for r in splits['development'])!=Counter({g:4 for g in GROUPS}):
        raise ValueError('balanced_16_development_required')
    for group in GROUPS:
        counts=Counter(r['paper_id'] for r in splits['development'] if r['style_group']==group)
        if sorted(counts.values())!=[2,2]:raise ValueError('two_development_papers_per_group_required')
    if {r['paper_id'] for r in splits['training']}&{r['paper_id'] for r in splits['development']}:
        raise ValueError('paper_leakage')
    for split in splits:
        if [r['example_id'] for r in splits[split]]!=config[split+'_ids']:raise ValueError('split_order_changed')
    probes=config['training_probe_ids'];train={r['example_id']:r for r in splits['training']}
    if len(probes)!=4 or len(set(probes))!=4 or not set(probes)<=set(train) or {train[x]['style_group'] for x in probes}!=set(GROUPS):
        raise ValueError('four_stratified_training_loss_probes_required')
    return splits,config


def mask_labels(prefix,ids,decoded_target,target):
    if ids[:len(prefix)]!=prefix or len(ids)<=len(prefix) or len(ids)>HYPER['max_length']:
        raise ValueError('token_boundary_or_length_no_truncation')
    if norm(decoded_target)!=norm(target):raise ValueError('supervised_target_mismatch')
    return [-100]*len(prefix)+ids[len(prefix):]


def learning_rate(step):
    if type(step) is not int or not 1<=step<=100:raise ValueError('step_outside_budget')
    return HYPER['learning_rate']*(1-(step-1)/100)


def run_training(splits,config,output,max_seconds):
    if output.exists():raise ValueError('fresh_output_directory_required')
    output.mkdir(parents=True);started=time.monotonic()
    state={'protocol':PROTOCOL,'complete':False,'phase':'start','optimizer_steps':0,
           'training_examples_seen':0,'training_called':False}
    report={'protocol':PROTOCOL,'complete':False,'config':config,'training_called':False,'snapshots':[]}
    def persist():
        write_json(output/'state.json',state);write_json(output/'evaluation.json',report)
    def budget():
        if time.monotonic()-started>=min(MAX_SECONDS,max_seconds):raise TimeoutError('pilot_wall_time_cap')
    def phase(name):
        state['phase']=name;persist();budget();print('Phase: '+name,flush=True)
    persist()
    try:
        import torch
        import torch.nn.functional as F
        from huggingface_hub import HfApi
        from transformers import AutoProcessor,Gemma3ForConditionalGeneration,BitsAndBytesConfig,GenerationConfig,LogitsProcessor,LogitsProcessorList
        from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training,PeftModel,get_peft_model_state_dict
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or 'T4' not in torch.cuda.get_device_name(0):
            raise RuntimeError('Expose GPU 0 only on Kaggle T4 x2')
        packages={k:importlib.metadata.version(k) for k in PACKAGES}
        if packages!=PACKAGES:raise RuntimeError('tested_package_versions_required')
        torch.cuda.set_device(0);torch.manual_seed(HYPER['seed']);torch.cuda.reset_peak_memory_stats(0)
        phase('model_access')
        try:
            from kaggle_secrets import UserSecretsClient
            token=UserSecretsClient().get_secret('HF_TOKEN')
        except Exception:raise RuntimeError('Enable HF_TOKEN in Kaggle Secrets') from None
        if not isinstance(token,str) or not token.strip():raise RuntimeError('Enable HF_TOKEN in Kaggle Secrets')
        if HfApi().model_info(MODEL_ID,revision=REVISION,token=token,timeout=60).sha!=REVISION:
            raise RuntimeError('base_revision_changed')
        processor=AutoProcessor.from_pretrained(MODEL_ID,revision=REVISION,token=token);tokenizer=processor.tokenizer
        def messages(row):return [{'role':'user','content':[{'type':'text','text':row['prompt']}]}]
        phase('tokenizer_preflight');encoded={};audits=[]
        for row in splits['training']+splits['development']:
            budget();state['current_example_id']=row['example_id'];persist()
            user=messages(row)
            prefix=processor.apply_chat_template(user,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt')['input_ids'][0].tolist()
            full=processor.apply_chat_template(user+[{'role':'assistant','content':[{'type':'text','text':row['target']}]}],tokenize=True,return_dict=True,return_tensors='pt')
            ids=full['input_ids'][0].tolist()
            state.update(prompt_tokens=len(prefix),full_tokens=len(ids));persist()
            labels=mask_labels(prefix,ids,tokenizer.decode(ids[len(prefix):],skip_special_tokens=True),row['target'])
            if len(prefix)+DECODING['max_new_tokens']>HYPER['max_length']:raise ValueError('generation_context_budget')
            if 'token_type_ids' in full and bool(full['token_type_ids'].any()):raise ValueError('text_only_token_types_required')
            encoded[row['example_id']]={'input_ids':ids,'prefix_ids':prefix,'labels':labels}
            audits.append({'example_id':row['example_id'],'split':row['split'],'full_tokens':len(ids),
                           'prompt_tokens':len(prefix),'supervised_tokens':len(ids)-len(prefix),
                           'target_roundtrip_matches':True,'truncated':False,'input_ids_sha256':sha(json.dumps(ids))})
            write_json(output/'label_audit.json',audits)
        report.update(label_preflight_passed=True,label_preflight_rows=len(audits),truncated_rows=0)
        report['runtime']={'model_id':MODEL_ID,'model_revision':REVISION,'packages':dict(packages,torch=torch.__version__),
                           'gpu':torch.cuda.get_device_name(0),'gpu_count':1,'runtime_source_sha256':sha(Path(__file__).read_text())}
        phase('load_model')
        model=Gemma3ForConditionalGeneration.from_pretrained(MODEL_ID,revision=REVISION,token=token,dtype=torch.float32,
            device_map={'':0},attn_implementation='sdpa',quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float32))
        token=None;model.eval();model.requires_grad_(False)
        generation=GenerationConfig(**DECODING,eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,use_cache=True)
        write_json(output/'generation_config.json',generation.to_dict())
        class FiniteScores(LogitsProcessor):
            def __call__(self,input_ids,scores):
                if bool(torch.isnan(scores).any()|torch.isposinf(scores).any()|~torch.isfinite(scores).any(dim=-1).all()):
                    raise RuntimeError('nonfinite_generation_scores')
                return scores
        def inputs(seq):
            ids=torch.tensor([seq['input_ids']],device='cuda:0',dtype=torch.long)
            return {'input_ids':ids,'attention_mask':torch.ones_like(ids),'token_type_ids':torch.zeros_like(ids)}
        def nll(row):
            budget();seq=encoded[row['example_id']];batch=inputs(seq);n=len(seq['prefix_ids'])
            result=model(**batch,use_cache=False,logits_to_keep=torch.arange(n-1,len(seq['input_ids'])-1,device='cuda:0'))
            loss=F.cross_entropy(result.logits[0].float(),batch['input_ids'][0,n:],reduction='mean')
            if not bool(torch.isfinite(loss)):raise RuntimeError('nonfinite_target_loss')
            return loss,len(seq['input_ids'])-n
        def generate(row):
            budget();prefix=encoded[row['example_id']]['prefix_ids'];ids=torch.tensor([prefix],device='cuda:0',dtype=torch.long)
            with torch.inference_mode():result=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),token_type_ids=torch.zeros_like(ids),
                generation_config=generation,logits_processor=LogitsProcessorList([FiniteScores()]))
            new=result[0,len(prefix):];text=tokenizer.decode(new,skip_special_tokens=True);eos=generation.eos_token_id
            eos=eos if isinstance(eos,list) else [eos]
            src=set(re.findall(r'\b\d+(?:[.,]\d+)*%?',row['source']));out=set(re.findall(r'\b\d+(?:[.,]\d+)*%?',text))
            return {'example_id':row['example_id'],'paper_id':row['paper_id'],'style_group':row['style_group'],
                'source':row['source'],'reference':row['target'],'prompt_sha256':sha(row['prompt']),
                'output':text,'output_sha256':sha(text),'output_tokens':int(new.numel()),'empty':not bool(text.strip()),
                'output_limited':bool(new.numel()>=DECODING['max_new_tokens'] and int(new[-1]) not in eos),
                'copied_source':norm(text)==norm(row['source']),
                'numeric_string_triage':{'missing':sorted(src-out),'added':sorted(out-src),
                                         'semantic_verdict':None,'interpretation':'Surface flags only; not semantic verification'}}
        def snapshot(name,step):
            model.eval();snap={'name':name,'optimizer_step':step,'complete':False,'development_losses':[],
                               'training_probe_losses':[],'generations':[]}
            report['snapshots'].append(snap);phase('snapshot_'+name)
            for split,key,selected in [('development','development_losses',splits['development']),
                                      ('training','training_probe_losses',[r for r in splits['training'] if r['example_id'] in config['training_probe_ids']])]:
                for row in selected:
                    with torch.no_grad():loss,n=nll(row)
                    snap[key].append({'example_id':row['example_id'],'loss':float(loss.item()),'supervised_tokens':n});persist()
                values=snap[key]
                snap[split+'_token_weighted_nll']=sum(r['loss']*r['supervised_tokens'] for r in values)/sum(r['supervised_tokens'] for r in values)
            for row in splits['development']:
                snap['generations'].append(generate(row));persist()
                print(name+': saved %d/16 held-out outputs'%len(snap['generations']),flush=True)
            snap['complete']=True;persist()
        # Check the optimized target-only likelihood against built-in masking.
        phase('likelihood_sanity');row=splits['development'][0]
        with torch.no_grad():
            manual,_=nll(row);seq=encoded[row['example_id']]
            builtin=float(model(**inputs(seq),labels=torch.tensor([seq['labels']],device='cuda:0'),use_cache=False).loss.item())
        if not math.isclose(float(manual.item()),builtin,rel_tol=1e-5,abs_tol=1e-5):raise RuntimeError('masked_likelihood_mismatch')
        report['likelihood_sanity']={'passed':True,'manual_mean_nll':float(manual.item()),'builtin_mean_nll':builtin}
        snapshot('base',0)
        phase('prepare_adapter');model.config.use_cache=False;model.config.text_config.use_cache=False
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
        matched=[name for name,_ in model.named_modules() if re.fullmatch(TARGET_PATTERN,name)]
        if len(matched)!=2*model.config.text_config.num_hidden_layers:raise RuntimeError('language_qv_layout_changed')
        torch.manual_seed(HYPER['seed'])
        model=get_peft_model(model,LoraConfig(r=HYPER['rank'],lora_alpha=HYPER['alpha'],lora_dropout=HYPER['dropout'],
                                             target_modules=TARGET_PATTERN,bias='none',task_type='CAUSAL_LM'))
        trainable={k:v for k,v in model.named_parameters() if v.requires_grad}
        if not trainable or any('lora_' not in k or 'language_model' not in k or v.dtype!=torch.float32 for k,v in trainable.items()):
            raise RuntimeError('only_fp32_language_adapters_may_train')
        initial={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
        report['trainable_parameters']=sum(v.numel() for v in trainable.values())
        optimizer=torch.optim.AdamW(list(trainable.values()),lr=HYPER['learning_rate'],betas=(.9,.999),eps=1e-8,weight_decay=0.)
        order=list(range(200));random.Random(HYPER['seed']).shuffle(order)
        report['training_order_ids']=[splits['training'][i]['example_id'] for i in order]
        updates=[];gradients=[];seen=[];phase('training');model.train();torch.manual_seed(HYPER['seed'])
        for offset in range(0,200,2):
            budget();optimizer.zero_grad(set_to_none=True);losses=[]
            state['training_called']=True;report['training_called']=True
            for index in order[offset:offset+2]:
                row=splits['training'][index];state['current_example_id']=row['example_id'];persist()
                value,n=nll(row);losses.append(float(value.detach().item()));loss=value/2;loss.backward();del value,loss
                seen.append(row['example_id']);state['training_examples_seen']=len(seen)
            grads=[v.grad for v in trainable.values() if v.grad is not None]
            if not grads or any(not bool(torch.isfinite(g).all()) for g in grads):raise RuntimeError('nonfinite_gradient')
            grad_norm=math.sqrt(sum(float(g.float().square().sum().item()) for g in grads))
            if not math.isfinite(grad_norm) or grad_norm<=0:raise RuntimeError('nonpositive_gradient')
            step=offset//2+1;lr=learning_rate(step)
            for group in optimizer.param_groups:group['lr']=lr
            torch.nn.utils.clip_grad_norm_(list(trainable.values()),HYPER['max_grad_norm'],error_if_nonfinite=True)
            optimizer.step();state['optimizer_steps']=step
            gradients.append({'step':step,'finite':True,'norm':grad_norm})
            updates.append({'step':step,'learning_rate':lr,'loss':sum(losses)/len(losses),'example_ids':seen[-2:]})
            write_json(output/'gradient_checks.json',gradients);write_json(output/'training_log.json',updates);persist()
            print('Training: %d/100 updates; %d/200 examples'%(step,len(seen)),flush=True)
        if len(set(seen))!=200 or state['optimizer_steps']!=100:raise RuntimeError('one_epoch_coverage_failed')
        phase('save_adapter');adapter=output/'epoch_1'/'adapter';model.save_pretrained(str(adapter),save_embedding_layers=False)
        saved={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
        if not any(not torch.equal(v,initial[k]) for k,v in saved.items()):raise RuntimeError('adapter_did_not_change')
        report['adapter_weights_changed']=True;report['adapter_files_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in adapter.iterdir() if p.is_file()}
        phase('reload_adapter');del optimizer
        model=PeftModel.from_pretrained(model.unload(),str(adapter),is_trainable=False)
        loaded=get_peft_model_state_dict(model)
        if set(saved)!=set(loaded) or any(not torch.equal(v,loaded[k].detach().cpu()) for k,v in saved.items()):raise RuntimeError('adapter_reload_mismatch')
        model.eval();model.requires_grad_(False);report['adapter_reload_weights_match']=True
        snapshot('epoch_1',100)
        if [len(s['generations']) for s in report['snapshots']]!=[16,16]:raise RuntimeError('matched_generation_coverage_failed')
        report.update(complete=True,training_examples_seen=200,optimizer_steps=100,
                      semantic_assessment='Pending automated source/reference review after download; numeric flags are not verdicts',detector_results=None)
        state.update(complete=True,phase='complete',elapsed_seconds=time.monotonic()-started,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(0))
        persist();print('Complete: 200 examples, 100 updates, 32 matched held-out generations; evaluation.json saved.',flush=True)
    except Exception as error:
        state.update(failed_phase=state['phase'],phase='failed',error_type=type(error).__name__,elapsed_seconds=time.monotonic()-started)
        persist()
        # Never serialize arbitrary provider exceptions or credential-bearing locals.
        raise RuntimeError('Pilot stopped; preserve state.json and partial outputs. Error type: '+type(error).__name__) from None


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--check',action='store_true');parser.add_argument('--train-authorized',action='store_true')
    parser.add_argument('--max-runtime-seconds',type=int,default=MAX_SECONDS)
    parser.add_argument('--output',type=Path,default=Path('/kaggle/working/corpus_200_pilot_v1'))
    args=parser.parse_args(argv);splits,config=load_inputs(args.data,args.config)
    if args.check:
        print('Validated 200 balanced training pairs / 16 paper-held-out development pairs; 100 updates. No model or GPU accessed.');return
    if not args.train_authorized or not 0<args.max_runtime_seconds<=MAX_SECONDS:
        raise ValueError('explicit_bounded_training_authorization_required')
    run_training(splits,config,args.output,args.max_runtime_seconds)


if __name__=='__main__':main()
