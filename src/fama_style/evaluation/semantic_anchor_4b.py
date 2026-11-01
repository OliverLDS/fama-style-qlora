"""Inference-only original/variant rewrites and teacher-forced semantic contrasts."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

PROTOCOL='semantic_anchor_inference_4b_v1'
MODEL_ID='google/gemma-3-4b-it'
REVISION='093f9f388b31de276ce2de164bdc2081324b9767'
INSTRUCTION='''Rewrite the passage as one paragraph of natural academic economics prose.
Retain its substantive claims, actors, quantities, comparisons, conditions,
exceptions, uncertainty, negation, attribution and causal direction. Improve
expression and organization where useful, without adding economic facts or
relationships. Do not fill gaps using outside knowledge. Return only the paragraph.'''
DECODING={'do_sample':False,'num_beams':1,'max_new_tokens':512,'repetition_penalty':1.0}
MAX_LENGTH=1024
STAGES=('base','expanded_epoch_2')


def sha(text):return hashlib.sha256(text.encode()).hexdigest()
def file_sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write_json(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def prompt(text):return INSTRUCTION+'\n\nPassage:\n'+text
def messages(job):return [{'role':'user','content':[{'type':'text','text':job['prompt']}]}]


def load_inputs(jobs_path,config_path):
    config=json.loads(config_path.read_text())
    if (config.get('protocol')!=PROTOCOL or config.get('model_id')!=MODEL_ID
            or config.get('model_revision')!=REVISION or config.get('decoding')!=DECODING
            or config.get('max_length')!=MAX_LENGTH or config.get('seed')!=1729
            or config.get('stages')!=list(STAGES) or config.get('training_authorized') is not False
            or config.get('quantization')!='NF4 double quantization; FP32 compute'
            or config.get('jobs_sha256')!=file_sha(jobs_path)):
        raise ValueError('frozen_inference_config_or_jobs_changed')
    jobs=[json.loads(line) for line in jobs_path.read_text().splitlines() if line.strip()]
    if len(jobs)!=24 or len({r['job_id'] for r in jobs})!=24:raise ValueError('exactly_24_unique_jobs_required')
    by={}
    fields={'job_id','example_id','paper_id','split','family','input_kind','source','prompt','source_sha256',
            'chosen','rejected','target_sha256','rejected_sha256'}
    for job in jobs:
        if (set(job)!=fields or job['split']!='training' or job['input_kind'] not in ('original','variant')
                or job['job_id']!=job['example_id']+'__'+job['input_kind']
                or job['prompt']!=prompt(job['source']) or job['source_sha256']!=sha(job['source'])
                or job['target_sha256']!=sha(job['chosen']) or job['rejected_sha256']!=sha(job['rejected'])
                or not job['source'].strip() or not job['chosen'].strip() or job['chosen']==job['rejected']):
            raise ValueError('job_direction_text_or_prompt_changed')
        by.setdefault(job['example_id'],{})[job['input_kind']]=job
    if len(by)!=12 or sorted(by)!=config['anchor_ids']:raise ValueError('frozen_12_anchor_ids_changed')
    for group in by.values():
        if set(group)!= {'original','variant'}:raise ValueError('both_input_conditions_required')
        a,b=group['original'],group['variant']
        if (a['source']==b['source'] or any(a[k]!=b[k] for k in ('chosen','rejected','paper_id','family','target_sha256','rejected_sha256'))):
            raise ValueError('matched_conditions_changed')
    if Counter(g['original']['family'] for g in by.values())!=Counter({'scope':3,'time':3,'numbers':3,'reasoning':3}):
        raise ValueError('balanced_families_required')
    if set(config['adapter_sha256'])!={'adapter_config.json','adapter_model.safetensors'}:
        raise ValueError('two_frozen_adapter_files_required')
    return jobs,config


def verify_adapter(folder,config):
    if any(file_sha(folder/name)!=digest for name,digest in config['adapter_sha256'].items()):
        raise ValueError('attach_exact_reviewed_epoch_2_adapter')
    adapter=json.loads((folder/'adapter_config.json').read_text())
    if (adapter['base_model_name_or_path']!=MODEL_ID or adapter['r']!=8
            or adapter['lora_alpha']!=16 or adapter['lora_dropout']!=0.05):
        raise ValueError('adapter_model_or_rank_changed')


def mask_labels(prefix,ids,decoded,target):
    if ids[:len(prefix)]!=prefix or len(ids)<=len(prefix) or len(ids)>MAX_LENGTH:
        raise ValueError('target_boundary_or_length_no_truncation')
    if ' '.join(decoded.split())!=' '.join(target.split()):raise ValueError('decoded_target_mismatch')
    return [-100]*len(prefix)+ids[len(prefix):]


def margin(chosen,rejected):
    if any(not math.isfinite(r['mean_nll']) or r['mean_nll']<0 or r['supervised_tokens']<=0 for r in (chosen,rejected)):
        raise ValueError('finite_positive_token_scores_required')
    mean=rejected['mean_nll']-chosen['mean_nll']
    total=rejected['mean_nll']*rejected['supervised_tokens']-chosen['mean_nll']*chosen['supervised_tokens']
    def winner(delta):return 'tie' if abs(delta)<=1e-8 else 'chosen' if delta>0 else 'rejected'
    return {'mean_nll_margin_rejected_minus_chosen':mean,'total_nll_margin_rejected_minus_chosen':total,
            'mean_nll_preference':winner(mean),'total_nll_preference':winner(total),
            'mean_total_disagree':winner(mean)!=winner(total)}


def load_hf_token():
    try:
        from kaggle_secrets import UserSecretsClient
        token=UserSecretsClient().get_secret('HF_TOKEN')
    except Exception:raise RuntimeError('Enable HF_TOKEN in Kaggle Secrets') from None
    if not isinstance(token,str) or not token.strip():raise RuntimeError('Enable HF_TOKEN in Kaggle Secrets')
    return token


def run_training(jobs,config,adapter,output):
    """Repository ML entrypoint convention; this function performs inference only."""
    if output.exists() and any(output.iterdir()):raise ValueError('fresh_output_directory_required')
    verify_adapter(adapter,config)
    output.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    state={'protocol':PROTOCOL,'complete':False,'phase':'start','training_called':False,'optimizer_steps':0,
           'completed_generations':0,'completed_completion_scores':0}
    report={'protocol':PROTOCOL,'complete':False,'training_called':False,'config':config,'snapshots':[]}
    def persist():write_json(output/'state.json',state);write_json(output/'evaluation.json',report)
    def phase(name):state['phase']=name;persist();print('Phase: '+name,flush=True)
    persist()
    try:
        import torch
        from huggingface_hub import HfApi
        from transformers import (AutoProcessor,Gemma3ForConditionalGeneration,BitsAndBytesConfig,
                                  GenerationConfig,LogitsProcessor,LogitsProcessorList)
        from peft import PeftModel
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or 'T4' not in torch.cuda.get_device_name(0):
            raise RuntimeError('Expose only GPU 0 on Kaggle T4 x2 before importing torch')
        torch.cuda.set_device(0);torch.manual_seed(config['seed']);torch.cuda.reset_peak_memory_stats(0)
        phase('model_access');token=load_hf_token()
        if HfApi().model_info(MODEL_ID,revision=REVISION,token=token,timeout=60).sha!=REVISION:raise RuntimeError('model_revision_changed')
        processor=AutoProcessor.from_pretrained(MODEL_ID,revision=REVISION,token=token);tokenizer=processor.tokenizer
        phase('tokenizer_preflight');encoded={};audits=[]
        for job in jobs:
            state['current_job_id']=job['job_id'];persist()
            user=messages(job)
            inputs=processor.apply_chat_template(user,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt')
            prefix=inputs['input_ids'][0].tolist()
            state['prompt_tokens']=len(prefix);persist()
            if len(prefix)+DECODING['max_new_tokens']>MAX_LENGTH:raise ValueError('generation_context_budget')
            encoded[job['job_id']]={'generation':{k:v[0].tolist() for k,v in inputs.items()}}
            for candidate in ('chosen','rejected'):
                full=processor.apply_chat_template(user+[{'role':'assistant','content':[{'type':'text','text':job[candidate]}]}],
                    tokenize=True,return_dict=True,return_tensors='pt')
                ids=full['input_ids'][0].tolist();state['candidate']=candidate;state['full_tokens']=len(ids);persist()
                labels=mask_labels(prefix,ids,tokenizer.decode(ids[len(prefix):],skip_special_tokens=True),job[candidate])
                encoded[job['job_id']][candidate]={k:v[0].tolist() for k,v in full.items() if k in ('input_ids','attention_mask','token_type_ids')}
                encoded[job['job_id']][candidate]['labels']=labels
                audits.append({'job_id':job['job_id'],'candidate':candidate,'prompt_tokens':len(prefix),'full_tokens':len(ids),
                    'supervised_tokens':len(ids)-len(prefix),'decoded_target_matches':True,'input_ids_sha256':sha(json.dumps(ids))})
                write_json(output/'label_audit.json',audits)
        report['runtime']={'model_id':MODEL_ID,'model_revision':REVISION,'gpu':torch.cuda.get_device_name(0),'gpu_count':1,
            'runtime_source_sha256':file_sha(Path(__file__)),'packages':{p:importlib.metadata.version(p) for p in
                ('torch','transformers','peft','accelerate','bitsandbytes','huggingface_hub')}}
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
                if bool(torch.isnan(scores).any()|torch.isposinf(scores).any()|~torch.isfinite(scores).any(dim=-1).all()):raise RuntimeError('nonfinite_generation_scores')
                return scores
        def tensors(values):return {k:torch.tensor([v],dtype=torch.long,device='cuda:0') for k,v in values.items()}
        for stage in STAGES:
            state['current_stage']=stage
            if stage!='base':
                phase('load_frozen_adapter')
                model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False)
                model.eval();model.requires_grad_(False)
            if any(p.requires_grad for p in model.parameters()):raise RuntimeError('all_parameters_must_be_frozen')
            snap={'name':stage,'complete':False,'generations':[],'scores':[],'contrasts':[]};report['snapshots'].append(snap)
            phase(stage)
            for job in jobs:
                state['current_job_id']=job['job_id'];state['current_action']='generation'
                state.pop('candidate',None);persist()
                inputs=tensors(encoded[job['job_id']]['generation'])
                with torch.inference_mode():result=model.generate(**inputs,generation_config=generation,logits_processor=LogitsProcessorList([FiniteScores()]))
                new=result[0,inputs['input_ids'].shape[-1]:];text=tokenizer.decode(new,skip_special_tokens=True)
                eos=generation.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
                snap['generations'].append({k:job[k] for k in ('job_id','example_id','paper_id','family','input_kind','source','prompt','target_sha256')}|
                    {'output':text,'output_tokens':int(new.numel()),'empty':not bool(text.strip()),
                     'output_limited':bool(new.numel()>=DECODING['max_new_tokens'] and int(new[-1]) not in eos),
                     'copied_source':' '.join(text.split())==' '.join(job['source'].split())})
                del inputs,result,new
                state['completed_generations']+=1;persist()
                scored={}
                for candidate in ('chosen','rejected'):
                    state['current_action']='completion_score';state['candidate']=candidate;persist()
                    values=encoded[job['job_id']][candidate]
                    with torch.inference_mode():loss=float(model(**tensors(values)).loss.item())
                    if not math.isfinite(loss) or loss<0:raise RuntimeError('invalid_completion_loss')
                    n=sum(v!=-100 for v in values['labels']);score={'job_id':job['job_id'],'candidate':candidate,
                        'completion_sha256':sha(job[candidate]),'mean_nll':loss,'supervised_tokens':n,'total_nll':loss*n}
                    snap['scores'].append(score);scored[candidate]=score;state['completed_completion_scores']+=1;persist()
                snap['contrasts'].append({k:job[k] for k in ('job_id','example_id','family','input_kind')}|margin(scored['chosen'],scored['rejected']))
                persist();print(stage+': saved %d/24 rewrites and contrasts'%len(snap['generations']),flush=True)
            snap['complete']=True;persist()
        if state['completed_generations']!=48 or state['completed_completion_scores']!=96:raise RuntimeError('incomplete_fixed_coverage')
        report['complete']=True
        state.update(complete=True,phase='complete',elapsed_seconds=time.monotonic()-start,
                     peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(0));persist()
    except Exception as error:
        state.update(phase='failed',error_type=type(error).__name__);persist()
        raise RuntimeError('Inference failed; preserve state and partial outputs. Error type: '+type(error).__name__) from None


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs',type=Path,required=True);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--adapter',type=Path);parser.add_argument('--output',type=Path,default=Path('/kaggle/working/semantic_anchor_4b_v1'))
    parser.add_argument('--check',action='store_true');parser.add_argument('--run-inference',action='store_true')
    args=parser.parse_args(argv);jobs,config=load_inputs(args.jobs,args.config)
    if args.adapter is not None:verify_adapter(args.adapter,config)
    if args.check:print('Validated 24 matched input jobs / 12 training anchors. No model loaded or trained.');return
    if not args.run_inference or args.adapter is None:raise ValueError('explicit_inference_flag_and_frozen_adapter_required')
    run_training(jobs,config,args.adapter,args.output)


if __name__=='__main__':main()
