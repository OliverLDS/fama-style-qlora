"""Frozen-model multi-author baseline. CPU checks never load ML libraries."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import time

MODEL_ID='google/gemma-3-1b-it'
FIELDS={'job_id','example_id','condition','source','prompt','prompt_sha256'}
CONDITIONS={'ordinary','source_structural_prior'}


def sha(text):return hashlib.sha256(text.encode()).hexdigest()
def write_json(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def load_inputs(jobs_path,config_path):
    config=json.loads(config_path.read_text())
    if config['model_id']!=MODEL_ID or config['protocol']!='multi_author_milestone_v1':raise ValueError('Unexpected baseline model/protocol')
    if hashlib.sha256(jobs_path.read_bytes()).hexdigest()!=config['jobs_sha256']:raise ValueError('Jobs file changed')
    jobs=[json.loads(line) for line in jobs_path.read_text().splitlines() if line.strip()]
    if len(jobs)!=config['jobs'] or len({j['job_id'] for j in jobs})!=len(jobs):raise ValueError('Job coverage')
    groups={}
    for job in jobs:
        if set(job)!=FIELDS or job['condition'] not in CONDITIONS:raise ValueError('Unapproved generator fields/condition')
        if sha(job['prompt'])!=job['prompt_sha256'] or not job['prompt'].endswith('\n\nPassage:\n'+job['source']):raise ValueError('Prompt changed')
        group=groups.setdefault(job['example_id'],{})
        if job['condition'] in group:raise ValueError('Repeated condition')
        group[job['condition']]=job['source']
    if len(groups)!=config['development_examples'] or any(set(g)!=CONDITIONS or len(set(g.values()))!=1 for g in groups.values()):
        raise ValueError('Matched condition coverage/source')
    if config['decoding']!={'do_sample':False,'max_new_tokens':512,'context_limit':4096}:raise ValueError('Decoding changed')
    if config['quantization']!='NF4 double quantization; FP32 compute':raise ValueError('Quantization changed')
    return jobs,config


def load_hf_token():
    if Path('/kaggle').exists():
        try:
            from kaggle_secrets import UserSecretsClient
            value=UserSecretsClient().get_secret('HF_TOKEN')
        except Exception:raise RuntimeError('Enable HF_TOKEN in Kaggle Secrets') from None
    else:value=os.environ.get('HF_TOKEN')
    if not isinstance(value,str) or not value.strip():raise RuntimeError('Configure HF_TOKEN outside the repository')
    return value


def safe_error(error,token=None):
    text=str(error)
    if token:text=text.replace(token,'[REDACTED]')
    return re.sub(r'\bhf_[A-Za-z0-9]{20,}\b','[REDACTED]',text)[:1000]


def collect_outputs(jobs,generate,output_path):
    """Testable incremental save; one first-attempt output per immutable job."""
    with output_path.open('x') as stream:
        for i,job in enumerate(jobs,1):
            result=generate(job)
            if not isinstance(result.get('output'),str):raise ValueError('Missing output string')
            row={**{k:job[k] for k in ('job_id','example_id','condition','source','prompt_sha256')},**result}
            stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
            print('Completed %d/%d baseline jobs'%(i,len(jobs)),flush=True)


def run_training(jobs,config,output):
    """Inference only; the name follows this repository's GPU-import convention."""
    if output.exists() and any(output.iterdir()):raise ValueError('Use a fresh output directory')
    output.mkdir(parents=True,exist_ok=True)
    state={'complete':False,'training_called':False,'phase':'start','completed_jobs':0,'total_jobs':len(jobs)}
    token=None;start=time.monotonic()
    try:
        # All GPU/model imports stay lazy in this entry point.
        import torch
        from huggingface_hub import HfApi
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LogitsProcessor, LogitsProcessorList
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Expose exactly one T4 before importing torch; restart with the notebook first cell')
        torch.cuda.init();torch.cuda.set_device(0);torch.manual_seed(config['seed']);torch.cuda.reset_peak_memory_stats(0)
        token=load_hf_token();state['phase']='resolve_model_snapshot';write_json(output/'state.json',state)
        revision=HfApi().model_info(MODEL_ID,revision='main',token=token,timeout=60).sha
        if not re.fullmatch(r'[0-9a-f]{40}',revision or ''):raise RuntimeError('No immutable model revision')
        tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=revision,token=token)
        def encode(job):
            return tokenizer.apply_chat_template([{'role':'user','content':job['prompt']}],tokenize=True,
                add_generation_prompt=True,truncation=False,return_dict=True,return_tensors='pt')
        audits=[]
        for job in jobs:
            batch=encode(job);ids=batch['input_ids'][0].tolist();n=len(ids)
            if n+config['decoding']['max_new_tokens']>config['decoding']['context_limit']:raise RuntimeError('Prompt/output budget exceeds fixed context; no truncation permitted')
            audits.append({'job_id':job['job_id'],'input_tokens':n,'input_ids_sha256':sha(json.dumps(ids))})
        write_json(output/'token_audit.json',audits)
        metadata={'model_id':MODEL_ID,'model_revision':revision,'config':config,'training_called':False,
            'packages':{p:importlib.metadata.version(p) for p in ('torch','transformers','accelerate','bitsandbytes','huggingface_hub')},
            'gpu':torch.cuda.get_device_name(0),'visible_gpu_count':torch.cuda.device_count(),
            'evaluation_references_supplied':False,'generation_attempts_per_job':1}
        write_json(output/'runtime_metadata.json',metadata)
        state['phase']='load_frozen_model';write_json(output/'state.json',state)
        model=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=revision,token=token,dtype=torch.float32,
            device_map={'':0},attn_implementation='sdpa',quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float32))
        token=None;model.eval();model.requires_grad_(False)
        class FiniteScores(LogitsProcessor):
            def __call__(self,input_ids,scores):
                if bool(torch.isnan(scores).any() | torch.isposinf(scores).any() | ~torch.isfinite(scores).any(dim=-1).all()):raise RuntimeError('Nonfinite generation scores')
                return scores
        expected={a['job_id']:a for a in audits}
        def generate(job):
            batch=encode(job);ids=batch['input_ids'][0].tolist();n=len(ids)
            if sha(json.dumps(ids))!=expected[job['job_id']]['input_ids_sha256']:raise RuntimeError('Prompt tokenization changed')
            batch={k:v.to(model.get_input_embeddings().weight.device) for k,v in batch.items()};before=time.monotonic()
            with torch.inference_mode():
                result=model.generate(**batch,do_sample=False,max_new_tokens=512,use_cache=True,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                    logits_processor=LogitsProcessorList([FiniteScores()]),remove_invalid_values=False)
            new=result[0,n:];eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
            text=tokenizer.decode(new,skip_special_tokens=True).strip()
            state.update(phase='generating',completed_jobs=state['completed_jobs']+1);write_json(output/'state.json',state)
            return {'output':text,'input_tokens':n,'output_tokens':int(new.numel()),
                'hit_output_limit':bool(new.numel()>=512 and int(new[-1]) not in eos),
                'blank_output':not bool(text),'generation_seconds':time.monotonic()-before,'synthetic':False}
        collect_outputs(jobs,generate,output/'generations.private.jsonl')
        state.update(complete=True,phase='complete',elapsed_seconds=time.monotonic()-start,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(0))
        write_json(output/'state.json',state)
    except Exception as error:
        state.update(phase='failed',error_type=type(error).__name__,safe_error=safe_error(error,token))
        write_json(output/'state.json',state)
        raise RuntimeError('Baseline failed: '+safe_error(error,token)) from None


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jobs',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('/kaggle/working/multi_author_base_v1'));p.add_argument('--check',action='store_true')
    a=p.parse_args(argv);jobs,config=load_inputs(a.jobs,a.config)
    if a.check:
        print('Validated %d jobs / %d development examples; no models loaded or run.'%(len(jobs),config['development_examples']));return
    run_training(jobs,config,a.output)


if __name__=='__main__':main()
