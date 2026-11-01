"""Matched nine-output prose-only Gemma 3 4B diagnostic; no training."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import time

MODEL_ID='google/gemma-3-4b-it'
MODEL_REVISION='093f9f388b31de276ce2de164bdc2081324b9767'
PROTOCOL='neutral_reconstruction_prose_gemma4b_v1'
METHODS={'direct_rewrite','detailed_brief','compact_claims'}
DECODING={'do_sample':False,'num_beams':1,'max_new_tokens':512,
          'repetition_penalty':1.0,'context_limit':4096}


def sha(text):return hashlib.sha256(text.encode()).hexdigest()
def write_json(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def load_inputs(jobs_path,config_path):
    config=json.loads(config_path.read_text())
    if config['protocol']!=PROTOCOL or config['model_id']!=MODEL_ID or config['model_revision']!=MODEL_REVISION:
        raise ValueError('model_or_protocol_changed')
    if config['decoding']!=DECODING or config['seed']!=1729:raise ValueError('decoding_changed')
    if config['quantization']!='NF4 double quantization; FP32 compute':raise ValueError('precision_changed')
    if sha(jobs_path.read_text())!=config['jobs_sha256']:raise ValueError('jobs_hash_changed')
    jobs=[json.loads(line) for line in jobs_path.read_text().splitlines() if line.strip()]
    seen=set();groups={}
    for job in jobs:
        if set(job)!={'job_id','record_id','method','prompt','prompt_sha256'}:raise ValueError('generator_field_allowlist')
        if not all(isinstance(v,str) and v.strip() for v in job.values()):raise ValueError('nonempty_job_strings_required')
        if job['job_id'] in seen or job['method'] not in METHODS:raise ValueError('duplicate_job_or_unknown_method')
        seen.add(job['job_id'])
        if job['prompt_sha256']!=sha(job['prompt']):raise ValueError('prompt_hash_changed')
        methods=groups.setdefault(job['record_id'],set())
        if job['method'] in methods:raise ValueError('duplicate_method')
        methods.add(job['method'])
    if len(jobs)!=9 or len(groups)!=3 or any(g!=METHODS for g in groups.values()):raise ValueError('nine_matched_jobs_required')
    if sorted(groups)!=sorted(config['record_ids']):raise ValueError('record_coverage')
    return jobs,config


def classify_output(text):
    """Record contract cues without editing, extracting, rewriting or retrying text."""
    content=text.strip();flags=[]
    if not content:return {'status':'empty','format_flags':[]}
    if content.startswith('BLOCKED:'):
        if not content[len('BLOCKED:'):].strip():flags.append('blocked_without_reason')
        return {'status':'blocked','format_flags':flags}
    if content.startswith('```'):flags.append('markdown_fence')
    if content.startswith('#'):flags.append('heading_marker')
    if re.search(r'\n\s*\n',content):flags.append('multiple_paragraphs')
    try:
        value=json.loads(content)
        if isinstance(value,(dict,list)):flags.append('json_instead_of_prose')
    except ValueError:pass
    return {'status':'generated','format_flags':flags}


def check_budget(input_tokens):
    if input_tokens+DECODING['max_new_tokens']>DECODING['context_limit']:
        raise ValueError('context_budget_exceeded_no_truncation_allowed')


def load_hf_token():
    # Never substitute an environment value for a disabled Kaggle Secret.
    if Path('/kaggle').exists():
        try:
            from kaggle_secrets import UserSecretsClient
            token=UserSecretsClient().get_secret('HF_TOKEN')
        except Exception: raise RuntimeError('Enable the Kaggle Secret named HF_TOKEN') from None
    else: token=os.environ.get('HF_TOKEN')
    if not isinstance(token,str) or not token.strip(): raise RuntimeError('Configure HF_TOKEN outside the repository')
    return token


def run_training(jobs, config, output):
    """Inference only; all ML imports are lazy under the repository convention."""
    if output.exists() and any(output.iterdir()): raise ValueError('use_fresh_output_directory')
    output.mkdir(parents=True,exist_ok=True)
    state={'protocol':PROTOCOL,'complete':False,'training_called':False,'phase':'start','completed_jobs':0,'expected_jobs':9}
    write_json(output/'state.json',state)
    start=time.monotonic()
    def phase(name):
        state['phase']=name;write_json(output/'state.json',state);print('Phase: '+name,flush=True)
    try:
        import torch
        from huggingface_hub import HfApi
        from transformers import (AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig,
                                  GenerationConfig, LogitsProcessor, LogitsProcessorList)
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError('Expose exactly one T4 before importing torch')
        if 'T4' not in torch.cuda.get_device_name(0): raise RuntimeError('Select GPU T4 x2; only GPU 0 will be exposed')
        torch.cuda.set_device(0);torch.manual_seed(config['seed']);torch.cuda.reset_peak_memory_stats(0)
        phase('resolve_model_access');token=load_hf_token()
        revision=HfApi().model_info(MODEL_ID,revision=config['model_revision'],token=token,timeout=60).sha
        if revision!=config['model_revision']: raise RuntimeError('pinned_model_revision_mismatch')
        if not re.fullmatch(r'[0-9a-f]{40}',revision or ''): raise RuntimeError('immutable_revision_unavailable')
        phase('load_processor')
        processor=AutoProcessor.from_pretrained(MODEL_ID,revision=revision,token=token)
        tokenizer=processor.tokenizer
        def encode(job):
            return processor.apply_chat_template([{'role':'user','content':[{'type':'text','text':job['prompt']}]}],
                tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',truncation=False)
        audits=[]
        for job in jobs:
            encoded=encode(job);ids=encoded['input_ids'][0].tolist();check_budget(len(ids))
            audits.append({'job_id':job['job_id'],'input_tokens':len(ids),'input_ids_sha256':sha(json.dumps(ids))})
        write_json(output/'token_audit.json',audits)
        write_json(output/'runtime_metadata.json',{'model_id':MODEL_ID,'model_revision':revision,'config':config,
            'packages':{p:importlib.metadata.version(p) for p in ('torch','transformers','accelerate','bitsandbytes','huggingface_hub')},
            'gpu':torch.cuda.get_device_name(0),'visible_gpu_count':1,'training_called':False,
            'reference_policy':'Original paragraph appears only in direct jobs; reconstruction receives only the brief.',
            'attempts_per_job':1,'runtime_source_sha256':sha(Path(__file__).read_text()),
            'execution_complete_is_not_quality_approval':True})
        phase('load_quantized_model')
        model=Gemma3ForConditionalGeneration.from_pretrained(MODEL_ID,revision=revision,token=token,
            dtype=torch.float32,device_map={'':0},attn_implementation='sdpa',
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float32))
        token=None;model.eval();model.requires_grad_(False)
        generation=GenerationConfig(do_sample=False,num_beams=1,max_new_tokens=DECODING['max_new_tokens'],
            repetition_penalty=1.0,eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            use_cache=True)
        write_json(output/'generation_config.json',generation.to_dict())
        class FiniteScores(LogitsProcessor):
            def __call__(self,input_ids,scores):
                if bool(torch.isnan(scores).any() | torch.isposinf(scores).any() | ~torch.isfinite(scores).any(dim=-1).all()):
                    raise RuntimeError('nonfinite_generation_scores')
                return scores
        audit_by={r['job_id']:r for r in audits};results=[]
        with (output/'generations.private.jsonl').open('x') as stream:
            for job in jobs:
                phase('generate_'+job['job_id']);before=time.monotonic()
                batch=encode(job);ids=batch['input_ids'][0].tolist();n=len(ids)
                if sha(json.dumps(ids))!=audit_by[job['job_id']]['input_ids_sha256']: raise RuntimeError('tokenization_changed')
                batch={k:v.to(model.get_input_embeddings().weight.device) for k,v in batch.items()}
                with torch.inference_mode():
                    result=model.generate(**batch,generation_config=generation,
                        logits_processor=LogitsProcessorList([FiniteScores()]))
                new=result[0,n:];raw=tokenizer.decode(new,skip_special_tokens=True)
                eos=generation.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
                limited=bool(new.numel()>=DECODING['max_new_tokens'] and int(new[-1]) not in eos)
                parsed=classify_output(raw)
                row={k:job[k] for k in ('job_id','record_id','method','prompt_sha256')}
                row.update(raw_output=raw,parsed=parsed,input_tokens=n,output_tokens=int(new.numel()),
                    hit_output_limit=limited,generation_seconds=time.monotonic()-before)
                stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();results.append(row)
                state['completed_jobs']=len(results);write_json(output/'state.json',state)
                print('Saved %d/9 outputs'%len(results),flush=True)
        state.update(complete=True,execution_complete=True,phase='complete',
            response_contract_complete=all(r['parsed']['status']!='empty' and not r['parsed']['format_flags'] and not r['hit_output_limit'] for r in results),
            generated=sum(r['parsed']['status']=='generated' for r in results),
            blocked=sum(r['parsed']['status']=='blocked' for r in results),
            empty=sum(r['parsed']['status']=='empty' for r in results),
            format_review_outputs=sum(bool(r['parsed']['format_flags']) for r in results),
            output_limited=sum(r['hit_output_limit'] for r in results),
            elapsed_seconds=time.monotonic()-start,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(0))
        write_json(output/'state.json',state)
        print('All nine first outputs saved. Claim/style quality remains unassessed.',flush=True)
    except Exception as error:
        # Deliberately omit exception text: upstream errors can contain credentials.
        state.update(complete=False,failed_phase=state['phase'],phase='failed',error_type=type(error).__name__)
        write_json(output/'state.json',state)
        raise RuntimeError('Inference failed; inspect state.json and saved outputs. Error type: '+type(error).__name__) from None


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs',type=Path,required=True);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('/kaggle/working/reconstruction_prose_4b_v1'))
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args(argv);jobs,config=load_inputs(args.jobs,args.config)
    if args.check:
        print('Validated nine jobs / three cases; no models, weights or GPU accessed.');return
    run_training(jobs,config,args.output)


if __name__=='__main__':main()
