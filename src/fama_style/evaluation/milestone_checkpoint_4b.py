"""Generate only missing jobs for a frozen eight-case milestone; no training."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

PROTOCOL='multi_author_checkpoint_4b_v1'
MODEL_ID='google/gemma-3-4b-it'
REVISION='093f9f388b31de276ce2de164bdc2081324b9767'
INSTRUCTION='''Rewrite the passage as one paragraph of natural academic economics prose.
Retain its substantive claims, actors, quantities, comparisons, conditions,
exceptions, uncertainty, negation, attribution and causal direction. Improve
expression and organization where useful, without adding economic facts or
relationships. Do not fill gaps using outside knowledge. Return only the paragraph.'''
CONDITIONS=('base_ordinary','base_source_structural_prior','adapter_ordinary')
DECODING={'do_sample':False,'num_beams':1,'max_new_tokens':512,'repetition_penalty':1.0}


def sha(text):return hashlib.sha256(text.encode()).hexdigest()
def file_sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write_json(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def messages(job):return [{'role':'user','content':[{'type':'text','text':job['prompt']}]}]


def load_inputs(path,config_path):
    config=json.loads(config_path.read_text())
    if (config['protocol']!=PROTOCOL or config['model_id']!=MODEL_ID or config['model_revision']!=REVISION
            or config['decoding']!=DECODING or config['seed']!=1729 or config['context_limit']!=4096
            or config['quantization']!='NF4 double quantization; FP32 compute'
            or config['training_authorized'] is not False or config['jobs_sha256']!=file_sha(path)):
        raise ValueError('checkpoint_config_or_jobs_changed')
    jobs=[json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if len(jobs)!=24 or len({j['job_id'] for j in jobs})!=24:raise ValueError('full_24_job_matrix_required')
    fields={'job_id','example_id','condition','source','source_sha256','prompt','prompt_sha256','run_required'}
    by={}
    for j in jobs:
        if (set(j)!=fields or j['condition'] not in CONDITIONS or type(j['run_required']) is not bool
                or j['job_id']!=j['example_id']+'__'+j['condition']
                or j['source_sha256']!=sha(j['source']) or j['prompt_sha256']!=sha(j['prompt'])
                or not j['prompt'].startswith(INSTRUCTION) or not j['prompt'].endswith('\n\nPassage:\n'+j['source'])):
            raise ValueError('source_only_job_binding_failed')
        if j['condition']!='base_source_structural_prior' and j['prompt']!=INSTRUCTION+'\n\nPassage:\n'+j['source']:
            raise ValueError('ordinary_prompt_changed')
        by.setdefault(j['example_id'],{})[j['condition']]=j['source_sha256']
    if (sorted(by)!=config['example_ids'] or len(by)!=8
            or any(set(g)!=set(CONDITIONS) or len(set(g.values()))!=1 for g in by.values())):
        raise ValueError('same_eight_cases_all_three_arms_required')
    pending=Counter(j['condition'] for j in jobs if j['run_required'])
    if pending!=Counter({'base_ordinary':3,'base_source_structural_prior':8,'adapter_ordinary':3}):
        raise ValueError('only_fourteen_missing_jobs_may_run')
    reused={j['job_id'] for j in jobs if not j['run_required']}
    if reused!=set(config['reused_prompt_tokens']) or len(reused)!=10:raise ValueError('ten_archived_jobs_required')
    return jobs,config


def verify_adapter(folder,config):
    if any(file_sha(folder/name)!=digest for name,digest in config['adapter_sha256'].items()):
        raise ValueError('exact_original_input_epoch_2_adapter_required')


def load_hf_token():
    try:
        from kaggle_secrets import UserSecretsClient
        token=UserSecretsClient().get_secret('HF_TOKEN')
    except Exception:raise RuntimeError('Enable HF_TOKEN in Kaggle Secrets') from None
    if not isinstance(token,str) or not token.strip():raise RuntimeError('Enable HF_TOKEN in Kaggle Secrets')
    return token


def run_training(jobs,config,adapter,output):
    """Lazy ML entrypoint under repository convention; frozen inference only."""
    if output.exists() and any(output.iterdir()):raise ValueError('fresh_output_directory_required')
    verify_adapter(adapter,config);output.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    state={'protocol':PROTOCOL,'complete':False,'phase':'start','training_called':False,'optimizer_steps':0,'completed_jobs':0,'planned_jobs':14}
    report={'protocol':PROTOCOL,'complete':False,'training_called':False,'config':config,'generations':[]}
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
        packages={p:importlib.metadata.version(p) for p in config['expected_packages']}
        report['runtime']={'packages':packages,'model_id':MODEL_ID,'model_revision':REVISION,
            'runtime_source_sha256':file_sha(Path(__file__)),'gpu':torch.cuda.get_device_name(0),'gpu_count':1,
            'evaluation_references_supplied':False}
        phase('environment_check')
        if packages!=config['expected_packages']:raise RuntimeError('archive_reuse_requires_recorded_package_versions')
        torch.cuda.set_device(0);torch.manual_seed(config['seed']);torch.cuda.reset_peak_memory_stats(0)
        phase('model_access');token=load_hf_token()
        if HfApi().model_info(MODEL_ID,revision=REVISION,token=token,timeout=60).sha!=REVISION:raise RuntimeError('revision_changed')
        processor=AutoProcessor.from_pretrained(MODEL_ID,revision=REVISION,token=token);tokenizer=processor.tokenizer
        phase('tokenizer_preflight');encoded={};audits=[]
        for j in jobs:
            state['current_job_id']=j['job_id'];persist()
            batch=processor.apply_chat_template(messages(j),tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt')
            ids=batch['input_ids'][0].tolist();state['prompt_tokens']=len(ids);persist()
            limit=4096 if j['condition']=='base_source_structural_prior' else 1024
            if len(ids)+DECODING['max_new_tokens']>limit:raise ValueError('context_budget_no_truncation')
            if not j['run_required'] and len(ids)!=config['reused_prompt_tokens'][j['job_id']]:raise ValueError('archived_prompt_token_count_changed')
            audits.append({'job_id':j['job_id'],'input_tokens':len(ids),'input_ids_sha256':sha(json.dumps(ids)),
                'run_required':j['run_required'],'context_guard':limit})
            encoded[j['job_id']]={k:v[0].tolist() for k,v in batch.items()}
            write_json(output/'token_audit.json',audits)
        phase('load_model')
        model=Gemma3ForConditionalGeneration.from_pretrained(MODEL_ID,revision=REVISION,token=token,dtype=torch.float32,
            device_map={'':0},attn_implementation='sdpa',quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float32))
        token=None;model.eval();model.requires_grad_(False)
        generation=GenerationConfig(**DECODING,eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,use_cache=True)
        if generation.to_dict()!=config['expected_generation_config']:raise ValueError('archived_generation_config_changed')
        write_json(output/'generation_config.json',generation.to_dict())
        class FiniteScores(LogitsProcessor):
            def __call__(self,input_ids,scores):
                if bool(torch.isnan(scores).any()|torch.isposinf(scores).any()|~torch.isfinite(scores).any(dim=-1).all()):raise RuntimeError('nonfinite_scores')
                return scores
        for stage in ('base','adapter'):
            if stage=='adapter':
                phase('load_frozen_adapter');model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False)
                model.eval();model.requires_grad_(False)
            if any(p.requires_grad for p in model.parameters()):raise RuntimeError('model_parameters_must_be_frozen')
            phase(stage)
            for j in jobs:
                if not j['run_required'] or (j['condition']=='adapter_ordinary')!=(stage=='adapter'):continue
                state['current_job_id']=j['job_id'];persist()
                batch={k:torch.tensor([v],dtype=torch.long,device='cuda:0') for k,v in encoded[j['job_id']].items()}
                with torch.inference_mode():result=model.generate(**batch,generation_config=generation,logits_processor=LogitsProcessorList([FiniteScores()]))
                new=result[0,batch['input_ids'].shape[-1]:];text=tokenizer.decode(new,skip_special_tokens=True)
                eos=generation.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
                report['generations'].append({k:j[k] for k in ('job_id','example_id','condition','source','source_sha256','prompt_sha256')}|
                    {'output':text,'output_tokens':int(new.numel()),'empty':not bool(text.strip()),
                     'output_limited':bool(new.numel()>=512 and int(new[-1]) not in eos),
                     'copied_source':' '.join(text.split())==' '.join(j['source'].split())})
                state['completed_jobs']+=1;persist();print('Saved %d/14 missing outputs'%state['completed_jobs'],flush=True)
                del batch,result,new
        if state['completed_jobs']!=14:raise ValueError('incomplete_missing_job_coverage')
        report['complete']=True;state.update(complete=True,phase='complete',elapsed_seconds=time.monotonic()-start,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(0));persist()
    except Exception as error:
        state.update(phase='failed',error_type=type(error).__name__);persist()
        raise RuntimeError('Checkpoint inference failed; preserve partial outputs. Error type: '+type(error).__name__) from None


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jobs',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--adapter',type=Path);p.add_argument('--output',type=Path,default=Path('/kaggle/working/milestone_checkpoint_4b_v1'))
    p.add_argument('--check',action='store_true');p.add_argument('--run-inference',action='store_true')
    a=p.parse_args(argv);jobs,config=load_inputs(a.jobs,a.config)
    if a.adapter is not None:verify_adapter(a.adapter,config)
    if a.check:print('Validated 8 frozen cases / 24 arm slots: 10 archived, 14 pending; no models or training.');return
    if not a.run_inference or a.adapter is None:raise ValueError('explicit_inference_flag_and_reviewed_adapter_required')
    run_training(jobs,config,a.adapter,a.output)


if __name__=='__main__':main()
