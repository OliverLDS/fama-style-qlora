"""Prepare rhetorical pairs and run an explicitly requested inference-only oracle."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import threading
import time

from fama_style.research.run_extension import read_split, synthetic, EXPECTED
from fama_style.research.signal import fit_profile
from fama_style.research.claims import sha
from .extract import paired, inspect_legacy
from .mapping import probe
from .oracle import jobs_for, generate_jobs, validate_outputs, score, PROTOCOL
from .schema import VERSION
from .semantics import NLI_MODEL, assess, calibration_controls

ROOT = Path(__file__).resolve().parents[3]
MODEL_ID = 'google/gemma-3-4b-it'
MODEL_REVISION = '093f9f388b31de276ce2de164bdc2081324b9767'
MAX_NEW_TOKENS = 512
CONTEXT_LIMIT = 4096


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def write_lines(path, rows):
    path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows), encoding='utf-8')


def fresh(path):
    if path.exists() and any(path.iterdir()): raise ValueError('Refusing to overwrite experiment outputs')
    path.mkdir(parents=True, exist_ok=True)


def manifest(directory, synthetic_run):
    # Hash only outputs/code; do not load any reserved-test records.
    paths = sorted(Path(__file__).resolve().parent.glob('*.py'))
    paths += sorted((ROOT/'src/fama_style/research').glob('*.py'))
    paths += [ROOT/'src/fama_style/prompting/corpus.py']
    write_json(directory/'manifest.json', {'protocol': PROTOCOL, 'schema': VERSION,
        'synthetic': synthetic_run, 'test_rows_used': 0,
        'input_split_hashes': None if synthetic_run else EXPECTED,
        'code_hashes': {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'files': {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir())
                  if p.is_file() and p.name!='manifest.json'}})


def prepare(train, validation, output_dir, dry_run=False):
    if any(row['split']!='train' for row in train) or any(row['split']!='validation' for row in validation):
        raise ValueError('Unexpected split in rhetorical experiment')
    if len({r['id'] for r in train+validation})!=len(train)+len(validation): raise ValueError('Duplicate or overlapping split IDs')
    fresh(output_dir)
    pairs = [{'id': row['id'], 'paper_key': row['paper_key'], 'split':'train',
              'annotation': paired(row['plain_english'], row['completion'])} for row in train]
    write_lines(output_dir/'paired_schemas.private.jsonl', pairs)
    analysis = probe(pairs)
    write_json(output_dir/'learnability_summary.json', analysis)
    write_json(output_dir/'legacy_inventory.json', inspect_legacy(train))
    jobs = jobs_for(validation)
    write_lines(output_dir/'oracle_jobs.private.jsonl', jobs)
    protocol = {'protocol': PROTOCOL, 'status':'prepared_not_executed' if not dry_run else 'synthetic_wiring_check',
        'training_pairs': len(train), 'validation_examples': len(validation), 'generation_jobs': len(jobs),
        'schema_version': VERSION, 'model_id': MODEL_ID, 'model_revision': MODEL_REVISION,
        'decoding': {'do_sample': False, 'max_new_tokens':MAX_NEW_TOKENS, 'use_cache':True, 'seed':42},
        'context_limit':CONTEXT_LIMIT, 'truncate':False, 'condition_order':'cyclic A/B/C per validation index',
        'claim_information':'B/C share source evidence units; A is the ordinary frozen prompt',
        'primary_comparison':'C versus B; A anchors the ordinary rewrite baseline',
        'oracle_reference_wording_in_generator':False, 'test_rows_used':0, 'human_review_required':False,
        'gpu_training_allowed':False, 'inference_executed':False,
        'semantic_assessment': 'Independent bidirectional document and sentence-unit NLI after Gemma is unloaded; synthetic controls must pass; over-context pairs abstain',
        'factorial_followup':{'status':'blocked_until_oracle_support', 'cells':[
            {'weights':'base','mapping':'none'}, {'weights':'base','mapping':'learned'},
            {'weights':'frozen_qlora','mapping':'none'}, {'weights':'frozen_qlora','mapping':'learned'}],
            'controls':'same source examples, model revision, quantization, budgets, prompts within mapping condition; existing adapter was not plan-conditioned'}}
    write_json(output_dir/'oracle_protocol.json', protocol)
    profile = fit_profile(train)
    write_json(output_dir/'style_profile.json', profile)
    if dry_run:
        sources = {row['id']: row['plain_english'] for row in validation}
        outputs = generate_jobs(jobs, lambda job: {'output':sources[job['id']], 'hit_output_limit':False, 'synthetic':True,
                                                    'input_tokens':None, 'output_tokens':None, 'generation_seconds':None})
        records, summary = score(validation, jobs, outputs, profile, synthetic=True)
        write_lines(output_dir/'synthetic_outputs.private.jsonl', outputs)
        write_json(output_dir/'synthetic_wiring_summary.json', summary)
    manifest(output_dir, dry_run)
    return jobs, profile, analysis


def secret_token():
    if Path('/kaggle').exists():
        try:
            from kaggle_secrets import UserSecretsClient
            value = UserSecretsClient().get_secret('HF_TOKEN')
        except Exception:
            raise RuntimeError('Enable the Kaggle Secret named HF_TOKEN') from None
    else:
        value = os.environ.get('HF_TOKEN')
    if not isinstance(value,str) or not value.strip(): raise RuntimeError('Configure HF_TOKEN outside code and repository')
    return value


def with_heartbeat(call, label):
    stop = threading.Event(); started = time.monotonic()
    def tick():
        while not stop.wait(30): print(f'{label}: working ({int(time.monotonic()-started)} seconds)', flush=True)
    thread = threading.Thread(target=tick, daemon=True); thread.start()
    try: return call()
    finally: stop.set(); thread.join(timeout=1)


def audit_input_ids(job, ids):
    if sha(job['prompt'])!=job['prompt_sha256']: raise ValueError('Changed prompt')
    if not ids or len(ids)+MAX_NEW_TOKENS>CONTEXT_LIMIT: raise ValueError('Prompt/context overflow; no truncation allowed')
    return {'id':job['id'],'condition':job['condition'],'input_tokens':len(ids),
            'prompt_sha256':job['prompt_sha256'],'input_ids_sha256':sha(json.dumps(ids))}


def run_training(validation, jobs, profile, output_dir):
    """Inference only; name follows the repository convention for lazy ML imports."""
    fresh(output_dir)
    status = {'complete':False, 'training_called':False, 'test_rows_used':0, 'protocol':PROTOCOL}
    started = time.monotonic()
    def phase(name, **extra):
        status.update(phase=name, **extra); write_json(output_dir/'run_status.json',status)
        print('Phase: '+name, flush=True)
    try:
        phase('loading_inference_runtime')
        os.environ['CUDA_VISIBLE_DEVICES']='0'
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
        os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
        os.environ['TOKENIZERS_PARALLELISM']='false'
        import torch
        from transformers import (AutoProcessor, AutoTokenizer, AutoModelForSequenceClassification,
            BitsAndBytesConfig, Gemma3ForConditionalGeneration, LogitsProcessor, LogitsProcessorList)
        from huggingface_hub import HfApi
        from transformers.utils import logging
        logging.set_verbosity_error()
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise RuntimeError('Use Kaggle T4 x2; expose only its first GPU before importing torch')
        torch.cuda.init(); torch.cuda.set_device(0); torch.manual_seed(42); torch.cuda.reset_peak_memory_stats(0)
        token = secret_token()
        phase('loading_pinned_processor')
        processor = with_heartbeat(lambda:AutoProcessor.from_pretrained(MODEL_ID,revision=MODEL_REVISION,token=token), 'Processor load')
        def inputs_for(job):
            if sha(job['prompt'])!=job['prompt_sha256']: raise ValueError('Changed prompt')
            return processor.apply_chat_template([{'role':'user','content':[{'type':'text','text':job['prompt']}]}],
                tokenize=True,add_generation_prompt=True,truncation=False,return_dict=True,return_tensors='pt')
        audits=[]
        for job in jobs:
            batch=inputs_for(job)
            audits.append(audit_input_ids(job,batch['input_ids'][0].tolist()))
        write_json(output_dir/'token_audit.json',audits)
        phase('loading_pinned_gemma')
        model=with_heartbeat(lambda:Gemma3ForConditionalGeneration.from_pretrained(MODEL_ID,revision=MODEL_REVISION,
            token=token,dtype=torch.float32,device_map={'':0},attn_implementation='sdpa',
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float32)), 'Gemma load')
        del token
        if any(t.device.type!='cuda' or t.device.index!=0 for t in list(model.parameters())+list(model.buffers())):
            raise RuntimeError('Unexpected model tensor placement')
        model.eval(); model.requires_grad_(False)
        class FiniteScores(LogitsProcessor):
            def __call__(self,input_ids,scores):
                if bool(torch.isnan(scores).any() | torch.isposinf(scores).any() | ~torch.isfinite(scores).any(dim=-1).all()):
                    raise RuntimeError('Nonfinite generation scores')
                return scores
        audit_by_key={(a['id'],a['condition']):a for a in audits}
        def generate(job):
            batch=inputs_for(job); n=int(batch['input_ids'].shape[-1]); audit=audit_by_key[(job['id'],job['condition'])]
            if sha(json.dumps(batch['input_ids'][0].tolist()))!=audit['input_ids_sha256']: raise ValueError('Prompt tokenization changed')
            batch={k:v.to(model.get_input_embeddings().weight.device) for k,v in batch.items()}
            phase('generation',completed_jobs=generate.completed,total_jobs=len(jobs))
            before=time.monotonic()
            with torch.inference_mode():
                out=with_heartbeat(lambda:model.generate(**batch,do_sample=False,max_new_tokens=MAX_NEW_TOKENS,
                    use_cache=True,pad_token_id=processor.tokenizer.pad_token_id,
                    logits_processor=LogitsProcessorList([FiniteScores()]),remove_invalid_values=False), 'Generation')
            new=out[0,n:]; eos=model.generation_config.eos_token_id; eos=eos if isinstance(eos,list) else [eos]
            result={'output':processor.tokenizer.decode(new,skip_special_tokens=True).strip(),
                    'input_tokens':n,'output_tokens':int(new.numel()),'generation_seconds':time.monotonic()-before,
                    'hit_output_limit':bool(new.numel()>=MAX_NEW_TOKENS and int(new[-1]) not in eos),'synthetic':False}
            generate.completed+=1
            return result
        generate.completed=0
        outputs=generate_jobs(jobs,generate,lambda rows:write_lines(output_dir/'generations.private.jsonl',rows))
        validate_outputs(jobs,outputs)
        gpu_memory={'peak_allocated_bytes':torch.cuda.max_memory_allocated(0),'peak_reserved_bytes':torch.cuda.max_memory_reserved(0)}
        del generate,model,processor,batch
        torch.cuda.empty_cache()
        phase('loading_independent_nli_on_cpu')
        nli_started=time.monotonic()
        revision=HfApi(token=False).model_info(NLI_MODEL,timeout=30).sha
        if not isinstance(revision,str) or not re.fullmatch('[0-9a-f]{40}',revision): raise ValueError('Unresolved NLI revision')
        tokenizer=with_heartbeat(lambda:AutoTokenizer.from_pretrained(NLI_MODEL,revision=revision,token=False), 'NLI tokenizer load')
        nli=with_heartbeat(lambda:AutoModelForSequenceClassification.from_pretrained(NLI_MODEL,revision=revision,token=False), 'NLI model load')
        nli.eval(); nli.requires_grad_(False); torch.set_num_threads(min(4,os.cpu_count() or 1))
        labels={int(k):v.lower() for k,v in nli.config.id2label.items()}
        if set(labels.values())!={'entailment','neutral','contradiction'}: raise ValueError('Unexpected NLI labels')
        limit=min(512,int(getattr(nli.config,'max_position_embeddings',512)),int(tokenizer.model_max_length))
        def predict(premise,hypothesis):
            batch=tokenizer(premise,hypothesis,truncation=False,return_tensors='pt')
            if int(batch['input_ids'].shape[-1])>limit:return None
            with torch.inference_mode(): probabilities=torch.softmax(nli(**batch).logits[0],dim=-1).tolist()
            return {labels[i]:float(p) for i,p in enumerate(probabilities)}
        calibration=with_heartbeat(lambda:calibration_controls(predict),'NLI synthetic controls')
        write_json(output_dir/'nli_controls.json',calibration)
        sources={row['id']:row['plain_english'] for row in validation}; semantic={}
        for index,row in enumerate(outputs,1):
            semantic[(row['id'],row['condition'])]=with_heartbeat(lambda:assess(sources[row['id']],row['output'],predict),'NLI assessment')
            phase('independent_semantic_assessment',completed_nli_outputs=index,total_jobs=len(outputs))
        records,summary=score(validation,jobs,outputs,profile,semantic=semantic,calibration=calibration)
        write_lines(output_dir/'metrics.private.jsonl',records)
        summary.update(model_id=MODEL_ID,model_revision=MODEL_REVISION,nli_model=NLI_MODEL,nli_revision=revision,
            nli_seconds=time.monotonic()-nli_started,total_seconds=time.monotonic()-started,memory=gpu_memory,
            versions={name:importlib.metadata.version(name) for name in ('torch','transformers','bitsandbytes','accelerate','huggingface-hub')})
        write_json(output_dir/'evaluation.json',summary)
        phase('complete',complete=True)
        manifest(output_dir,False)
    except Exception as exc:
        # Remote exception text/locals can contain authenticated URLs. Do not persist them.
        frames=[]; tb=exc.__traceback__
        while tb:
            if tb.tb_frame.f_code.co_filename==__file__:frames.append(tb.tb_lineno)
            tb=tb.tb_next
        phase('failed',complete=False,error_code='oracle_inference_failed',local_line_numbers=frames)
        raise RuntimeError('Oracle inference failed; inspect run_status.json and saved partial outputs. No training was called.') from None


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group()
    action.add_argument('--dry-run',action='store_true',help='Synthetic CPU wiring check, no model/Secret/network')
    action.add_argument('--execute-inference',action='store_true',help='Explicitly run the 63-job Kaggle inference experiment; never training')
    parser.add_argument('--train',type=Path,default=ROOT/'data/fama_corpus_v1/train.jsonl')
    parser.add_argument('--validation',type=Path,default=ROOT/'data/fama_corpus_v1/validation.jsonl')
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args(argv)
    if args.dry_run:
        train,validation=synthetic()
        for index,row in enumerate(train):row['paper_key']=f'SYN-{index}'
    else:train,validation=read_split(args.train,'train'),read_split(args.validation,'validation')
    fresh(args.output_dir)
    jobs,profile,analysis=prepare(train,validation,args.output_dir/'preparation',args.dry_run)
    if args.execute_inference:run_training(validation,jobs,profile,args.output_dir/'inference')
    print(json.dumps({'status':'synthetic_check_complete' if args.dry_run else 'inference_complete' if args.execute_inference else 'prepared_not_executed',
                      'training_pairs':len(train),'validation_examples':len(validation),'jobs':len(jobs),'training_called':False,'test_rows_used':0}))


if __name__=='__main__':main()
