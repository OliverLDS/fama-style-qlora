"""Count-only prose versus sentence-slot JSON: inference only, first attempts."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import re
import statistics
import time

from . import concise_plan as base
from .concise_plan import (MODEL_ID,MODEL_REVISION,NLI_MODEL,NLI_REVISION,CONTEXT_LIMIT,
    file_sha,read_lines,load_pack,synthetic_pack)
from .experiment import secret_token,with_heartbeat,write_json,write_lines,fresh
from .semantics import assess,calibration_controls
from fama_style.prompting.corpus import make_prompt
from fama_style.research.claims import sentences,sha,risk_flags
from fama_style.research.signal import distance,overlap

PROTOCOL='sentence_slot_realization_v1'
CONDITIONS=('A_ordinary','B_count_only','C_sentence_slots')
MAX_NEW_TOKENS=768


def jobs_for(pack):
    jobs=[]
    for index,row in enumerate(pack['rows']):
        # Rotate order rather than always giving a condition the same position.
        order=CONDITIONS[index%3:]+CONDITIONS[:index%3]
        for condition in order:
            prompt=make_prompt(row['source'])
            count=row['plan']['sentence_count']
            if condition!='A_ordinary':
                prompt+=f'\n\nPresentation control: write exactly {count} complete sentences. Preserve source claim order, all claims and qualifications. Do not add claims to fill slots. Faithfulness takes priority.'
            if condition=='C_sentence_slots':
                prompt+='\nOutput-format override: return only a JSON object with the single key "sentences", whose value is an array of exactly '+str(count)+' strings. Each string must contain exactly one complete sentence ending in a period, question mark or exclamation mark. No Markdown fences, commentary or other keys. The strings in order will be joined into one paragraph.'
            jobs.append({'id':row['id'],'condition':condition,'prompt':prompt,'prompt_sha256':sha(prompt),'source_sha256':sha(row['source'])})
    return jobs


def realize(raw,condition,count):
    """Strict format validation. No repair, regex extraction or silent salvage."""
    errors=[];items=None;text=raw
    if condition=='C_sentence_slots':
        try:
            def unique(pairs):
                result={}
                for key,value in pairs:
                    if key in result:raise ValueError('duplicate_key')
                    result[key]=value
                return result
            obj=json.loads(raw,object_pairs_hook=unique)
            if not isinstance(obj,dict) or set(obj)!={'sentences'}:raise ValueError('wrong_keys')
            items=obj['sentences']
            if not isinstance(items,list) or not items or any(not isinstance(s,str) or not s.strip() for s in items):
                raise ValueError('invalid_sentence_array')
            text=' '.join(s.strip() for s in items)
            if len(items)!=count:errors.append('wrong_slot_count')
            if any(len(sentences(s.strip()))!=1 or not re.search(r'[.!?]["”’\']?$',s.strip()) or '\n' in s for s in items):
                errors.append('slot_not_one_complete_surface_sentence')
        except (ValueError,TypeError):
            errors.append('invalid_json_contract')
    else:
        if raw.lstrip().startswith(('```','{','[')):errors.append('non_paragraph_format')
    return {'text':text,'format_valid':not errors,'format_errors':errors,'slot_count':len(items) if isinstance(items,list) else None,
            'sentence_count':len(sentences(text)),'count_match':len(sentences(text))==count,
            'semantic_input':'joined_slots' if items is not None and not errors else ('raw_invalid_format' if errors else 'paragraph')}


def audit_tokens(job,ids):
    if sha(job['prompt'])!=job['prompt_sha256']:raise ValueError('Changed prompt')
    if not ids or len(ids)+MAX_NEW_TOKENS>CONTEXT_LIMIT:raise ValueError('Context overflow; no truncation allowed')
    return {'id':job['id'],'condition':job['condition'],'input_tokens':len(ids),'prompt_sha256':job['prompt_sha256'],'input_ids_sha256':sha(json.dumps(ids))}


def evaluate(pack,jobs,outputs,semantic=None,pair_checks=None,calibration=None,synthetic=False):
    expected={(j['id'],j['condition']):j for j in jobs};seen=set();records=[]
    if jobs!=jobs_for(pack) or len(outputs)!=len(expected):raise ValueError('Changed jobs or incomplete outputs')
    rows={r['id']:r for r in pack['rows']}
    for out in outputs:
        key=(out['id'],out['condition']);job=expected.get(key)
        if key in seen or job is None or out['prompt_sha256']!=job['prompt_sha256']:raise ValueError('Duplicate/swapped output')
        if not out.get('output','').strip() or out.get('hit_output_limit') is not False or out.get('synthetic') is not synthetic:
            raise ValueError('Empty, cutoff or mislabeled output')
        seen.add(key);row=rows[out['id']];n=row['plan']['sentence_count'];r=realize(out['output'],out['condition'],n);text=r['text']
        nli=(semantic or {}).get(key);valid=r['format_valid'];passed=bool(nli and nli['status']=='proxy_consistent' and calibration and calibration['passed'])
        records.append({'id':out['id'],'condition':out['condition'],**r,'desired_sentence_count':n,
            'sentence_count_absolute_error':abs(r['sentence_count']-n),
            'nli':nli,'nli_status':nli['status'] if nli else 'not_run',
            'format_count_semantic_proxy_pass':valid and r['count_match'] and passed,
            'near_copy':overlap(row['source'],text)>=.9 if valid else None,
            'style_gain_proxy':distance(row['source'],pack['style_profile'])-distance(text,pack['style_profile']) if valid else None,
            'semantic_surface_flags':risk_flags(row['source'],text),
            'generation_seconds':out.get('generation_seconds',0),'input_tokens':out.get('input_tokens',0),'output_tokens':out.get('output_tokens',0)})
    aggregate={}
    for c in CONDITIONS:
        group=[r for r in records if r['condition']==c];valid=[r for r in group if r['format_valid']]
        aggregate[c]={'rows':len(group),'format_valid':len(valid),'count_matches':sum(r['format_valid'] and r['count_match'] for r in group),
            'format_invalid':len(group)-len(valid),
            'nli_unknown_valid_only':sum(r['nli_status']=='unknown_over_context' for r in valid),
            'mean_count_error_valid_only':statistics.mean(r['sentence_count_absolute_error'] for r in valid) if valid else None,
            'nli_proxy_passes_valid_only':sum(r['nli_status']=='proxy_consistent' for r in valid) if calibration and calibration['passed'] else None,
            'format_count_semantic_proxy_passes':sum(r['format_count_semantic_proxy_pass'] for r in group),
            'near_copies_valid_only':sum(r['near_copy'] for r in valid),
            'mean_style_gain_valid_only':statistics.mean(r['style_gain_proxy'] for r in valid) if valid else None,
            **{k:sum(r[k] or 0 for r in group) for k in ('generation_seconds','input_tokens','output_tokens')}}
    return records,{'protocol':PROTOCOL,'execution_kind':'synthetic_wiring_check' if synthetic else 'model_inference','conditions':aggregate,
        'pair_source_reference_nli':pair_checks,'nli_controls':calibration,'training_called':False,'held_out_rows_used':0,
        'oracle_is_deployable':False,'semantic_accuracy':None,'full_mapping_training_ready':False,'repair_attempts':0,
        'limitations':'Four previously inspected training pairs; count-only and output-format interventions; no generalization claim. Sentence boundaries and style are surface proxies. NLI can miss drift. Invalid formats remain failures in all-row denominators; valid-only metrics must report their denominators.'}


def run_training(pack,output,uploaded_pack_sha):
    """Inference only; lazy ML imports follow the repository entry-point convention."""
    fresh(output);jobs=jobs_for(pack);write_lines(output/'jobs.private.jsonl',jobs)
    write_json(output/'uploaded_pack_identity.json',{'sha256':uploaded_pack_sha,'format':'uploaded file bytes'})
    status={'complete':False,'protocol':PROTOCOL,'training_called':False,'held_out_rows_used':0,'completed_jobs':0,'total_jobs':len(jobs)}
    started=time.monotonic();outputs=[]
    def phase(name,**extra):
        status.update(phase=name,**extra);write_json(output/'run_status.json',status);print('Phase: '+name,flush=True)
    try:
        phase('loading_inference_runtime')
        import os
        import gc
        os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
        os.environ['HF_HUB_DISABLE_TELEMETRY']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
        import torch
        from transformers import (AutoProcessor,AutoTokenizer,AutoModelForSequenceClassification,
            Gemma3ForConditionalGeneration,BitsAndBytesConfig,LogitsProcessor,LogitsProcessorList)
        from transformers.utils import logging
        logging.set_verbosity_error()
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Expose one Kaggle GPU before importing torch')
        torch.cuda.init();torch.cuda.set_device(0);torch.manual_seed(42);torch.cuda.reset_peak_memory_stats(0)
        if 'T4' not in torch.cuda.get_device_name(0):raise RuntimeError('Select Kaggle T4 x2 for the pinned diagnostic')
        phase('checking_secret');token=secret_token()
        phase('loading_pinned_processor')
        processor=with_heartbeat(lambda:AutoProcessor.from_pretrained(MODEL_ID,revision=MODEL_REVISION,token=token),'Processor load')
        def tokenize(job):
            return processor.apply_chat_template([{'role':'user','content':[{'type':'text','text':job['prompt']}]}],
                tokenize=True,add_generation_prompt=True,truncation=False,return_dict=True,return_tensors='pt')
        audits=[]
        for job in jobs:audits.append(audit_tokens(job,tokenize(job)['input_ids'][0].tolist()))
        write_json(output/'token_audit.json',audits)
        phase('loading_pinned_gemma')
        model=with_heartbeat(lambda:Gemma3ForConditionalGeneration.from_pretrained(MODEL_ID,revision=MODEL_REVISION,
            token=token,dtype=torch.float32,device_map={'':0},attn_implementation='sdpa',
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float32)),'Gemma load')
        del token
        if any(t.device.type!='cuda' or t.device.index!=0 for t in list(model.parameters())+list(model.buffers())):
            raise RuntimeError('Unexpected model placement')
        model.eval();model.requires_grad_(False)
        class FiniteScores(LogitsProcessor):
            def __call__(self,input_ids,scores):
                if bool(torch.isnan(scores).any()|torch.isposinf(scores).any()|~torch.isfinite(scores).any(dim=-1).all()):
                    raise RuntimeError('Nonfinite generation scores')
                return scores
        by_key={(r['id'],r['condition']):r for r in audits}
        for job in jobs:
            phase('generation',completed_jobs=len(outputs))
            batch=tokenize(job);ids=batch['input_ids'][0].tolist();audit=audit_tokens(job,ids)
            if audit!=by_key[(job['id'],job['condition'])]:raise ValueError('Tokenized prompt changed')
            batch={k:v.to(model.get_input_embeddings().weight.device) for k,v in batch.items()};before=time.monotonic()
            with torch.inference_mode():
                generated=with_heartbeat(lambda:model.generate(**batch,do_sample=False,max_new_tokens=MAX_NEW_TOKENS,
                    use_cache=True,pad_token_id=processor.tokenizer.pad_token_id,remove_invalid_values=False,
                    logits_processor=LogitsProcessorList([FiniteScores()])),'Generation')
            new=generated[0,len(ids):];eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
            text=processor.tokenizer.decode(new,skip_special_tokens=True).strip()
            result={'id':job['id'],'condition':job['condition'],'prompt_sha256':job['prompt_sha256'],'output':text,
                'input_tokens':len(ids),'output_tokens':int(new.numel()),'generation_seconds':time.monotonic()-before,
                'hit_output_limit':bool(new.numel()>=MAX_NEW_TOKENS and int(new[-1]) not in eos),'synthetic':False}
            outputs.append(result);write_lines(output/'generations.private.jsonl',outputs)
            if not text or result['hit_output_limit']:raise RuntimeError('Empty/cutoff output; partial results remain incomplete')
            phase('generation',completed_jobs=len(outputs))
        gemma_memory={'peak_allocated_bytes':torch.cuda.max_memory_allocated(0),'peak_reserved_bytes':torch.cuda.max_memory_reserved(0)}
        del model,processor,batch,generated,new,tokenize
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(0)
        phase('loading_independent_nli_on_gpu');nli_started=time.monotonic()
        tokenizer=with_heartbeat(lambda:AutoTokenizer.from_pretrained(NLI_MODEL,revision=NLI_REVISION,token=False),'NLI tokenizer load')
        nli=with_heartbeat(lambda:AutoModelForSequenceClassification.from_pretrained(NLI_MODEL,revision=NLI_REVISION,
            token=False).to('cuda:0'),'NLI model load')
        nli.eval();nli.requires_grad_(False)
        labels={int(k):v.lower() for k,v in nli.config.id2label.items()}
        if set(labels.values())!={'entailment','neutral','contradiction'}:raise ValueError('Unexpected NLI labels')
        limit=min(512,int(getattr(nli.config,'max_position_embeddings',512)),int(tokenizer.model_max_length))
        cache={};cache_hits=0
        def predict(premise,hypothesis):
            nonlocal cache_hits
            key=(sha(premise),sha(hypothesis))
            if key in cache:cache_hits+=1;return cache[key]
            batch=tokenizer(premise,hypothesis,truncation=False,return_tensors='pt')
            if int(batch['input_ids'].shape[-1])>limit:cache[key]=None;return None
            with torch.inference_mode():
                probs=torch.softmax(nli(**{k:v.to('cuda:0') for k,v in batch.items()}).logits[0],dim=-1).tolist()
            cache[key]={labels[i]:float(p) for i,p in enumerate(probs)}
            return cache[key]
        calibration=with_heartbeat(lambda:calibration_controls(predict),'NLI controls');write_json(output/'nli_controls.json',calibration)
        semantic={};pair_checks={}
        for index,row in enumerate(pack['rows'],1):
            phase('pair_fidelity_assessment',completed_pairs=index-1)
            pair_checks[row['id']]=assess(row['source'],row['reference'],predict)
        source_by_id={row['id']:row['source'] for row in pack['rows']}
        for index,result in enumerate(outputs,1):
            phase('output_semantic_assessment',completed_nli_outputs=index-1)
            semantic[(result['id'],result['condition'])]=with_heartbeat(lambda:assess(source_by_id[result['id']],realize(result['output'],result['condition'],next(r['plan']['sentence_count'] for r in pack['rows'] if r['id']==result['id']))['text'],predict),'NLI assessment')
        records,summary=evaluate(pack,jobs,outputs,semantic,pair_checks,calibration)
        write_lines(output/'metrics.private.jsonl',records)
        summary.update(model_id=MODEL_ID,model_revision=MODEL_REVISION,nli_model=NLI_MODEL,nli_revision=NLI_REVISION,
            nli_device='cuda:0 after Gemma unload',nli_seconds=time.monotonic()-nli_started,total_seconds=time.monotonic()-started,
            nli_cache_entries=len(cache),nli_cache_hits=cache_hits,gemma_memory=gemma_memory,
            nli_memory={'peak_allocated_bytes':torch.cuda.max_memory_allocated(0),'peak_reserved_bytes':torch.cuda.max_memory_reserved(0)},
            generation={'do_sample':False,'seed':42,'max_new_tokens':MAX_NEW_TOKENS,'context_limit':CONTEXT_LIMIT,'truncate':False},
            versions={name:importlib.metadata.version(name) for name in ('torch','transformers','accelerate','bitsandbytes','huggingface-hub')})
        write_json(output/'evaluation.json',summary)
        phase('complete',complete=True,completed_jobs=len(outputs),completed_nli_outputs=len(outputs),completed_pairs=len(pack['rows']))
        write_json(output/'manifest.json',{'protocol':PROTOCOL,'training_called':False,'input_pack_sha256':uploaded_pack_sha,
            'files':{p.name:file_sha(p) for p in output.iterdir() if p.is_file()}})
    except Exception as error:
        failed_at=status.get('phase');locations=[];tb=error.__traceback__
        while tb:
            if tb.tb_frame.f_code.co_filename==__file__:locations.append(tb.tb_lineno)
            tb=tb.tb_next
        phase('failed',complete=False,failed_at=failed_at,error_code='concise_plan_inference_failed',local_line_numbers=locations)
        raise RuntimeError('Concise-plan inference failed; inspect run_status.json and saved partial files. No training was called.') from None


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_mutually_exclusive_group()
    actions.add_argument('--dry-run',action='store_true');actions.add_argument('--execute-inference',action='store_true')
    parser.add_argument('--pack',type=Path);parser.add_argument('--expected-pack-sha');parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.dry_run:
        fresh(args.output_dir);pack=synthetic_pack();jobs=jobs_for(pack);rows={r['id']:r for r in pack['rows']}
        outputs=[]
        for j in jobs:
            ref=rows[j['id']]['reference'];text=json.dumps({'sentences':sentences(ref)}) if j['condition']=='C_sentence_slots' else ref
            outputs.append({'id':j['id'],'condition':j['condition'],'prompt_sha256':j['prompt_sha256'],'output':text,'hit_output_limit':False,'synthetic':True})
        records,summary=evaluate(pack,jobs,outputs,synthetic=True)
        write_json(args.output_dir/'synthetic_evaluation.json',summary);write_lines(args.output_dir/'synthetic_jobs.jsonl',jobs)
        print('Synthetic wiring check complete: 12 jobs; no model calls.');return
    if args.pack is None or args.expected_pack_sha is None:parser.error('--pack and --expected-pack-sha required')
    pack=load_pack(args.pack,args.expected_pack_sha)
    if args.execute_inference:run_training(pack,args.output_dir,args.expected_pack_sha)
    else:
        fresh(args.output_dir);write_json(args.output_dir/'preflight.json',{'protocol':PROTOCOL,'pack_sha256':args.expected_pack_sha,'pairs':4,'jobs':12,'training_called':False})
        print('Preflight passed; no Secret access or model loading.')


if __name__=='__main__':main()
