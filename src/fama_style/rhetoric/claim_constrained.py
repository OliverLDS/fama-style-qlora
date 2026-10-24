"""Matched 2x2 claim-constraint and output-format inference diagnostic."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import re
import statistics
import time

from . import concise_plan as base
from . import sentence_slots as slot_base
from .concise_plan import (MODEL_ID,MODEL_REVISION,NLI_MODEL,NLI_REVISION,CONTEXT_LIMIT,
    file_sha,read_lines,load_pack,synthetic_pack)
from .experiment import secret_token,with_heartbeat,write_json,write_lines,fresh
from .semantics import assess,calibration_controls as original_controls
from fama_style.prompting.corpus import make_prompt
from fama_style.research.claims import sentences,sha,risk_flags,ledger
from fama_style.research.signal import distance,overlap

PROTOCOL='claim_constrained_realization_v1'
CONDITIONS=('A_count_paragraph','B_claims_paragraph','C_count_slots','D_claims_slots')
MAX_NEW_TOKENS=768


def constraint_ledger(source):
    """Source-only evidence, not an independent semantic parse or repaired input."""
    extracted=ledger(source);claims=[]
    bounds=(('upper_bound',r"\b(?:does not exceed|do not exceed|never more than|at most|no more than)\b"),
            ('lower_bound',r"\b(?:does not fall below|do not fall below|at least|no less than)\b"))
    for claim in extracted['claims']:
        text=claim['evidence']['text'];protected=[]
        for kind in ('quantities','negation','modality','scope','condition_cues','causality'):
            protected.extend({'kind':kind,'evidence':x['text']} for x in claim[kind])
        protected.extend({'kind':label,'evidence':m.group()} for label,pattern in bounds for m in re.finditer(pattern,text,re.I))
        claims.append({'claim_id':claim['claim_id'],'source_evidence':text,'protected_evidence':protected})
    return {'claims':claims,'status':'sentence_spans_and_surface_cues_not_atomic_or_independently_verified'}


def jobs_for(pack):
    jobs=[]
    for index,row in enumerate(pack['rows']):
        order=CONDITIONS[index%4:]+CONDITIONS[:index%4]
        count=row['plan']['sentence_count'];common=make_prompt(row['source'])
        common+=f'\n\nPresentation control: write exactly {count} complete sentences. Preserve source claim order. Faithfulness takes priority over sentence count.'
        for condition in order:
            prompt=common
            if condition in ('B_claims_paragraph','D_claims_slots'):
                prompt+='\n\nSource-only preservation ledger (evidence, not additional facts):\n'+json.dumps(constraint_ledger(row['source']),ensure_ascii=False,separators=(',',':'))
                prompt+='\nPreserve every ledger entry, including conditions, attribution, uncertainty and supporting details. Keep upper bounds as upper bounds and lower bounds as lower bounds; do not invert inequality direction or its negation. Preserve ranges, denominators, quantities and qualifiers together. Possibility is not certainty, association is not causation, and an observed frequency is not a universal claim. Do not invent explanations, mechanisms, biases, recommendations or conclusions. The ledger is a fallible surface aid; the complete source remains authoritative. Reorganize wording only; do not add sentences with unsupported claims to reach the count. Return no ledger or analysis.'
            if condition.endswith('_slots'):
                prompt+='\nOutput-format override: return only a JSON object with the single key "sentences", whose value is an array of exactly '+str(count)+' strings. Each string must contain exactly one complete sentence ending in a period, question mark or exclamation mark. No Markdown fences, commentary or other keys. The strings in order will be joined into one paragraph.'
            jobs.append({'id':row['id'],'condition':condition,'prompt':prompt,'prompt_sha256':sha(prompt),'source_sha256':sha(row['source'])})
    return jobs


def realize(raw,condition,count):
    mapped='C_sentence_slots' if condition.endswith('_slots') else 'A_ordinary'
    strict=slot_base.realize(raw,mapped,count);normalized=raw;wrapper_removed=False
    # Predeclared normalization: strip exactly one complete outer JSON fence only.
    if mapped=='C_sentence_slots' and raw.startswith('```json\n') and raw.endswith('\n```'):
        normalized=raw[len('```json\n'):-len('\n```')];wrapper_removed=True
    result=slot_base.realize(normalized,mapped,count)
    result.update(raw_format_valid=strict['format_valid'],raw_format_errors=strict['format_errors'],outer_fence_removed=wrapper_removed)
    if result['slot_count'] is not None:result['semantic_input']='joined_slots' if result['format_valid'] else 'joined_slots_format_failed'
    return result


def calibration_controls(predict):
    controls=original_controls(predict)
    pairs=[('upper_bound_reversal','Independence is useful only when dependence does not exceed the threshold.',
            'Independence is useful only when dependence does not fall below the threshold.'),
           ('quantity_cap','The maximum portfolio share is 4 percent.','The maximum portfolio share is 8 percent.'),
           ('uncertainty','The pattern may occur in some years.','The pattern occurs in all years.'),
           ('causal_direction','Higher costs cause lower trading volume.','Lower trading volume causes higher costs.')]
    results={name:assess(a,b,predict) for name,a,b in pairs}
    controls['original_controls_passed']=controls['passed']
    controls['targeted_mutation_checks']=results
    controls['targeted_mutations_rejected']=sum(r['status']=='proxy_flagged' for r in results.values())
    controls['passed']=controls['passed'] and controls['targeted_mutations_rejected']==len(pairs)
    controls['meaning']='Synthetic sanity checks including targeted semantic mutations; passing does not verify economic claims or checker accuracy.'
    return controls


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
            'format_count_semantic_proxy_pass':valid and r['count_match'] and passed if calibration and calibration['passed'] else None,
            'near_copy':overlap(row['source'],text)>=.9 if valid else None,
            'style_gain_proxy':distance(row['source'],pack['style_profile'])-distance(text,pack['style_profile']) if valid else None,
            'semantic_surface_flags':risk_flags(row['source'],text),
            'generation_seconds':out.get('generation_seconds',0),'input_tokens':out.get('input_tokens',0),'output_tokens':out.get('output_tokens',0)})
    aggregate={}
    for c in CONDITIONS:
        group=[r for r in records if r['condition']==c];valid=[r for r in group if r['format_valid']]
        aggregate[c]={'rows':len(group),'format_valid':len(valid),'count_matches':sum(r['format_valid'] and r['count_match'] for r in group),
            'format_invalid':len(group)-len(valid),
            'raw_format_valid':sum(r['raw_format_valid'] for r in group),
            'outer_fences_removed':sum(r['outer_fence_removed'] for r in group),
            'nli_unknown_valid_only':sum(r['nli_status']=='unknown_over_context' for r in valid),
            'mean_count_error_valid_only':statistics.mean(r['sentence_count_absolute_error'] for r in valid) if valid else None,
            'nli_proxy_passes_valid_only':sum(r['nli_status']=='proxy_consistent' for r in valid) if calibration and calibration['passed'] else None,
            'format_count_semantic_proxy_passes':sum(r['format_count_semantic_proxy_pass'] for r in group) if calibration and calibration['passed'] else None,
            'uncalibrated_nli_consistent_valid_only':sum(r['nli_status']=='proxy_consistent' for r in valid),
            'near_copies_valid_only':sum(r['near_copy'] for r in valid),
            'mean_style_gain_valid_only':statistics.mean(r['style_gain_proxy'] for r in valid) if valid else None,
            **{k:sum(r[k] or 0 for r in group) for k in ('generation_seconds','input_tokens','output_tokens')}}
    return records,{'protocol':PROTOCOL,'execution_kind':'synthetic_wiring_check' if synthetic else 'model_inference','conditions':aggregate,
        'pair_source_reference_nli':pair_checks,'nli_controls':calibration,'training_called':False,'held_out_rows_used':0,
        'oracle_is_deployable':False,'semantic_accuracy':None,'full_mapping_training_ready':False,'repair_attempts':0,'normalization':'Exactly one full outer json code fence; report raw and normalized compliance separately',
        'limitations':'Four previously inspected training pairs; matched constraints x format; source-only heuristic ledgers; no generalization claim. Sentence boundaries and style are surface proxies. NLI can miss drift. Invalid formats remain failures in all-row denominators; valid-only metrics must report their denominators.'}


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
            ref=rows[j['id']]['reference'];text=json.dumps({'sentences':sentences(ref)}) if j['condition'].endswith('_slots') else ref
            outputs.append({'id':j['id'],'condition':j['condition'],'prompt_sha256':j['prompt_sha256'],'output':text,'hit_output_limit':False,'synthetic':True})
        records,summary=evaluate(pack,jobs,outputs,synthetic=True)
        write_json(args.output_dir/'synthetic_evaluation.json',summary);write_lines(args.output_dir/'synthetic_jobs.jsonl',jobs)
        print('Synthetic wiring check complete: 16 jobs; no model calls.');return
    if args.pack is None or args.expected_pack_sha is None:parser.error('--pack and --expected-pack-sha required')
    pack=load_pack(args.pack,args.expected_pack_sha)
    if args.execute_inference:run_training(pack,args.output_dir,args.expected_pack_sha)
    else:
        fresh(args.output_dir);write_json(args.output_dir/'preflight.json',{'protocol':PROTOCOL,'pack_sha256':args.expected_pack_sha,'pairs':4,'jobs':16,'training_called':False})
        print('Preflight passed; no Secret access or model loading.')


if __name__=='__main__':main()
