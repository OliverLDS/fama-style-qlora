"""Three matched guidance conditions on base Gemma and a frozen planned adapter."""
import argparse
from collections import Counter
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import statistics
import time
from . import grounded_operations as grounded
from . import bound_grouping as grouping
from .concise_plan import TRAIN_SHA,MODEL_ID,MODEL_REVISION,NLI_MODEL,NLI_REVISION,file_sha,read_lines
from .experiment import fresh,write_json,write_lines,secret_token,with_heartbeat
from fama_style.training import plan_realizer as previous
from fama_style.evaluation import corpus_baseline as baseline
from fama_style.research.claims import sha
from fama_style.research import claim_verifier_v3 as verifier
from fama_style.research.signal import fit_profile

ROOT=Path(__file__).resolve().parents[3]
PROTOCOL='grounded_operation_realization_v1'
CONDITIONS=('A_ordinary_shared_evidence','B_shallow_shared_evidence','C_operations_shared_evidence')
MODELS=('base','planned_adapter')
CONFIG={'do_sample':False,'max_new_tokens':512,'context_limit':4096,'seed':42}
ADAPTER_FILES={'adapter_config.json':'9d2401646892d22c671d79d1bf4fa2c9bb392220ceb19c74e1f62953027c9974',
 'adapter_model.safetensors':'49925e79bccdce068a7d0b6e574b7a5b89494ce71ec83de1f971625a85a8f35d'}


def source_prefix(payload):
    # Only source evidence, shared byte-for-byte across all three conditions.
    return (previous.COMMON+'Source:\n'+payload['source']+
        '\n\nSource claim evidence (fallible; full source remains authoritative):\n'+json.dumps(payload['source_claim_index'],ensure_ascii=False,sort_keys=True)+
        '\nSource supplements:\n'+json.dumps(payload['source_facts'],ensure_ascii=False,sort_keys=True)+
        '\nExisting source relationships (not new findings):\n'+json.dumps(payload['source_relations'],ensure_ascii=False,sort_keys=True))


def jobs_for(pack):
    jobs=[]
    for model in MODELS:
        for i,row in enumerate(pack['rows']):
            prefix=source_prefix(row['payload'])
            for condition in CONDITIONS[i%3:]+CONDITIONS[:i%3]:
                prompt=prefix
                if condition==CONDITIONS[1]:
                    previous.validate_plan(row['shallow_plan'])
                    prompt+='\n\nPresentation plan:\n'+json.dumps(row['shallow_plan'],sort_keys=True)+(
                        '\nWord bands: short 1-16, medium 17-32, long 33-48, extended 49-72. '
                        'Aim for the count, word bands and semicolon ceiling only when source claims permit. Retain claim order.')
                elif condition==CONDITIONS[2]:
                    program=row['payload']['program'];catalog=grounded.CATALOG[program['operation']]
                    prompt+='\n\nRhetorical program:\n'+json.dumps(program,sort_keys=True)+(
                        '\nParagraph action: '+catalog['paragraph_action']+'\nSentence action: '+catalog['sentence_action']+
                        '\nStages are discourse moves, not compulsory one-sentence slots. Preserve source-supported relation direction and scope.')
                prompt+='\nReturn only the rewritten paragraph.'
                jobs.append({'id':row['id'],'model':model,'condition':condition,'prompt':prompt,'prompt_sha256':sha(prompt),
                    'source_sha256':sha(row['source']),'use':row['use']})
    return jobs


def job_key(row):return (row['model'],row['id'],row['condition'])
def runtime_paths():return sorted(set(list((ROOT/'src/fama_style').rglob('*.py'))+[ROOT/grouping.POLICY]))


def prepare_pack(output):
    base=ROOT/'research_runs/grounded_rhetorical_operations_v1';release=json.loads((base/'release_verification.json').read_text())
    package=base/release['active_package']
    if file_sha(package/'manifest.json')!=release['package_manifest_sha256']:raise ValueError('Changed operation release')
    grounded.check(package);cards=read_lines(package/'operation_cards.private.jsonl')
    pilot=ROOT/'research_runs/plan_realizer_pilot_v1';review=json.loads((pilot/'version_1_review/verification.json').read_text())
    if not all(review['checks'].values()):raise ValueError('Pilot was not verified')
    run=pilot/'kaggle_version_1/plan_realizer_pilot_v1'
    for name in ['adapter_planned/adapter_config.json','adapter_planned/adapter_model.safetensors','adapter_hashes.json','run_status.json']:
        if file_sha(run/name)!=review['artifact_hashes'][name]:raise ValueError('Changed pilot adapter/provenance')
    train_path=ROOT/'data/fama_corpus_v1/train.jsonl'
    if file_sha(train_path)!=TRAIN_SHA:raise ValueError('Changed training split')
    train=read_lines(train_path);status=json.loads((run/'run_status.json').read_text())
    trained={i for u in status['updates']['planned'] for i in u['ids']}
    if any(c['id'] not in trained for c in cards):raise ValueError('Incorrect adapter exposure declaration')
    rows=[]
    for c in cards:
        payload=grounded.generation_payload(c)
        rows.append({'id':c['id'],'split':'train','use':c['use'],'source':c['source'],'reference':c['reference'],
            'source_sha256':sha(c['source']),'reference_sha256':sha(c['reference']),'operation':c['program']['operation'],
            'payload':payload,'shallow_plan':previous.presentation_plan(c['reference']),
            'source_sentence_claim_groups':c['source_annotation']['sentence_claim_groups'],
            'target_sentence_claim_groups_evaluation_only':c['target_annotation']['sentence_claim_groups'],
            'uncertainty_evaluation_only':c['uncertainty'],'seen_by_adapter_during_training':True})
    pack={'protocol':PROTOCOL,'model_id':MODEL_ID,'model_revision':MODEL_REVISION,'train_sha256':TRAIN_SHA,'config':CONFIG,
        'rows':rows,'style_profile':fit_profile(train),'adapter_file_hashes':ADAPTER_FILES,
        'adapter_tensor_hashes':json.loads((run/'adapter_hashes.json').read_text())['trained']['planned'],
        'operation_package_manifest_sha256':release['package_manifest_sha256'],
        'runtime_code_files':{str(p.relative_to(ROOT)):file_sha(p) for p in runtime_paths()},
        'scope':'Five previously inspected training examples, all seen by the saved adapter. Four operation contrasts and one preservation control. All arms share source evidence; no development/test use. Oracle reference-derived programs, not a learned mapper.'}
    pack['jobs_sha256']=sha(json.dumps(jobs_for(pack),sort_keys=True,ensure_ascii=False))
    fresh(output);write_json(output/'operation_pack.json',pack);write_lines(output/'jobs.private.jsonl',jobs_for(pack))
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'status':'prepared_not_executed','pack_sha256':file_sha(output/'operation_pack.json'),'generation_calls':30,'training_called':False,'test_rows_read':0})
    print('Prepared 30 inference calls on base + saved adapter; no inference or training executed.')


def load_pack(path,expected_sha):
    if not re.fullmatch('[0-9a-f]{64}',expected_sha or '') or file_sha(path)!=expected_sha:raise ValueError('Pack hash mismatch')
    pack=json.loads(path.read_text())
    if pack['protocol']!=PROTOCOL or pack['model_id']!=MODEL_ID or pack['model_revision']!=MODEL_REVISION or pack['train_sha256']!=TRAIN_SHA or pack['config']!=CONFIG or pack['adapter_file_hashes']!=ADAPTER_FILES:raise ValueError('Changed identity')
    if len(pack['rows'])!=5 or {r['id'] for r in pack['rows']}!={s['id'] for s in grounded.SPECS}:raise ValueError('Changed selection')
    if Counter(r['use'] for r in pack['rows'])!=Counter({'contrast_hypothesis':4,'preservation_control':1}):raise ValueError('Changed contrast/control assignment')
    for r in pack['rows']:
        if r['split']!='train' or r['source_sha256']!=sha(r['source']) or r['reference_sha256']!=sha(r['reference']) or r['payload']['source']!=r['source'] or r['seen_by_adapter_during_training'] is not True:raise ValueError('Changed text/split/exposure')
        grounded.render_payload(r['payload'])
        if previous.presentation_plan(r['reference'])!=r['shallow_plan']:raise ValueError('Changed shallow oracle plan')
    for name,h in pack['runtime_code_files'].items():
        if file_sha(ROOT/name)!=h:raise ValueError('Frozen runtime changed')
    if sha(json.dumps(jobs_for(pack),sort_keys=True,ensure_ascii=False))!=pack['jobs_sha256']:raise ValueError('Changed jobs')
    return pack


def adapter_directory(root,pack):
    matches=list(Path(root).rglob('adapter_planned/adapter_config.json'))
    if len(matches)!=1:raise ValueError('Attach exactly one planned-adapter directory from pilot version 1')
    directory=matches[0].parent
    for name,h in pack['adapter_file_hashes'].items():
        if file_sha(directory/name)!=h:raise ValueError('Wrong adapter artifact')
    config=json.loads((directory/'adapter_config.json').read_text())
    if config['base_model_name_or_path']!=MODEL_ID or config['revision']!=MODEL_REVISION:raise ValueError('Wrong adapter base/revision')
    return directory


def audit_tokens(job,ids):
    if sha(job['prompt'])!=job['prompt_sha256'] or not ids or len(ids)+CONFIG['max_new_tokens']>CONFIG['context_limit']:raise ValueError('Changed/over-context input; no truncation')
    return {'id':job['id'],'model':job['model'],'condition':job['condition'],'input_tokens':len(ids),'input_ids_sha256':sha(json.dumps(ids))}


def operation_diagnostics(row,text,predict):
    """Sentence-location hypotheses, not semantic proof or a full rhetoric parser."""
    parsed=grouping.output_units(text);claims=row['payload']['source_claim_index'];matrices=[]
    if parsed['ambiguous']:
        return {'status':'abstain_ambiguous_boundaries','claim_sentence_scores':[],'stage_positions':None,'rhetorical_adherence_verified':False}
    for claim in claims:
        evidence=' '.join(s['quote'] for s in claim['source_spans'])
        scores=[predict(sentence,evidence) for sentence in parsed['units']]
        known=[(p['entailment'],i) for i,p in enumerate(scores) if p is not None]
        ranked=sorted(known,reverse=True)
        # Thresholds are diagnostic localization rules, not calibrated fidelity.
        best=ranked[0] if ranked else None;margin=best[0]-ranked[1][0] if len(ranked)>1 else None
        matrices.append({'claim_id':claim['claim_id'],'source_evidence_sha256':sha(evidence),'probabilities_by_output_sentence':scores,
            'best_position':best[1]+1 if best else None,'top_margin':margin,
            'localized_position':best[1]+1 if best and best[0]>=.75 and (margin is None or margin>=.1) else None})
    positions={c['claim_id']:c['localized_position'] for c in matrices};stages=[]
    for stage in row['payload']['program']['stages']:
        if stage.get('source_fact_refs'):
            stages.append({'role':stage['role'],'positions':None,'status':'supplemental_count_not_an_atomic_claim'});continue
        found=[positions[c] for c in stage['source_claim_refs']]
        stages.append({'role':stage['role'],'positions':found,'status':'localized' if all(p is not None for p in found) else 'unresolved'})
    repeated=[c for c,n in Counter(c for s in row['payload']['program']['stages'] for c in s['source_claim_refs']).items() if n>1]
    return {'status':'localization_hypotheses_only','claim_sentence_scores':matrices,'stage_positions':stages,
        'composite_claims_used_by_multiple_stages':repeated,'rhetorical_adherence_verified':False,
        'limitation':'NLI can assign incorrect locations or accept partial claims. Composite claims cannot resolve separate subclaim realization. Stage locations do not measure language actions or author style.'}


def evaluate(pack,outputs,evidence=None,synthetic=False):
    expected={job_key(j):j for j in jobs_for(pack)};rows={r['id']:r for r in pack['rows']};seen=set();records=[]
    for out in outputs:
        key=job_key(out);job=expected.get(key)
        if key in seen or job is None or out['prompt_sha256']!=job['prompt_sha256'] or out.get('synthetic') is not synthetic or type(out['hit_output_limit']) is not bool:raise ValueError('Unknown/changed/repeated output')
        seen.add(key);row=rows[out['id']];available=bool(out['output'].strip()) and not out['hit_output_limit'];v=(evidence or {}).get(key)
        if available and not synthetic and (not v or v['source_sha256']!=row['source_sha256'] or v['output_sha256']!=sha(out['output'])):raise ValueError('Missing or unbound full evidence')
        metric=previous.output_metrics(pack,{'source':row['source'],'reference':row['reference'],'plan':row['shallow_plan']},out,v['verifier'] if v and available else None)
        records.append({**metric,'use':row['use'],'operation':row['operation'],'operation_diagnostics':v['operation_diagnostics'] if v and available else None})
    if seen!=set(expected):raise ValueError('Incomplete output matrix')
    aggregate={}
    for use in ('contrast_hypothesis','preservation_control'):
        aggregate[use]={}
        for model in MODELS:
            for condition in CONDITIONS:
                group=[r for r in records if r['model']==model and r['condition']==condition and r['use']==use];valid=[r for r in group if r['available']]
                aggregate[use][model+'__'+condition]={'attempts':len(group),'available':len(valid),'near_copies':sum(r['source_similarity']>=.9 for r in valid),
                    'shallow_count_matches_all_attempts':sum(r['count_matches'] is True for r in valid),
                    'mean_source_similarity_available_only':statistics.mean(r['source_similarity'] for r in valid) if valid else None,
                    'verifier_statuses':dict(Counter(r['verifier']['status'] for r in valid if r['verifier'])),
                    **{k:sum(r.get(k,0) for r in group) for k in ('input_tokens','output_tokens','generation_seconds')}}
    return records,{'protocol':PROTOCOL,'status':'synthetic_wiring_check' if synthetic else 'inference_complete' if all(r['available'] for r in records) else 'execution_complete_with_generation_failures',
        'generation_calls':30,'groups':aggregate,'training_called':False,'validation_rows_read':0,'reserved_test_rows_read':0,'semantic_accuracy':None,'author_style_accuracy':None,
        'qualitative_assessment':'pending','scope':'Previously inspected TRAINING examples, all seen by the saved adapter; oracle plans, no learned mapper. All arms share source evidence. The preservation control is not a style success. Location hypotheses are not verified operation adherence.'}


def run_training(pack,output,pack_sha,adapter_dir):
    """Explicit inference entry point; all ML imports stay within this function."""
    fresh(output);jobs=jobs_for(pack);outputs=[];started=time.monotonic()
    status={'protocol':PROTOCOL,'complete':False,'training_called':False,'completed_jobs':0,'total_jobs':len(jobs)}
    def phase(name):
        status.update(phase=name,completed_jobs=len(outputs));write_json(output/'run_status.json',status);print('Phase: '+name,flush=True)
    try:
        import os
        os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
        os.environ['TOKENIZERS_PARALLELISM']='false'
        phase('loading_gemma_runtime')
        import torch
        from transformers import AutoProcessor,AutoTokenizer,AutoModelForSequenceClassification,Gemma3ForConditionalGeneration,BitsAndBytesConfig,LogitsProcessor,LogitsProcessorList
        from peft import PeftModel,get_peft_model_state_dict
        from transformers.utils import logging
        logging.set_verbosity_error()
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Expose one T4')
        torch.cuda.init();torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats(0)
        if 'T4' not in torch.cuda.get_device_name(0):raise RuntimeError('Select T4 x2')
        token=secret_token()
        processor=with_heartbeat(lambda:AutoProcessor.from_pretrained(MODEL_ID,revision=MODEL_REVISION,token=token),'Processor load')
        def tokenize(job):
            return processor.apply_chat_template([{'role':'user','content':[{'type':'text','text':job['prompt']}]}],tokenize=True,
                add_generation_prompt=True,truncation=False,return_dict=True,return_tensors='pt')
        audits=[audit_tokens(j,tokenize(j)['input_ids'][0].tolist()) for j in jobs];write_json(output/'token_audit.json',audits)
        model=with_heartbeat(lambda:Gemma3ForConditionalGeneration.from_pretrained(MODEL_ID,revision=MODEL_REVISION,token=token,
            dtype=torch.float32,device_map={'':0},attn_implementation='sdpa',quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float32)),'Gemma load')
        del token
        if any(t.device.type!='cuda' or t.device.index!=0 for t in list(model.parameters())+list(model.buffers())):raise RuntimeError('Unexpected placement')
        model.eval();model.requires_grad_(False)
        class FiniteScores(LogitsProcessor):
            def __call__(self,input_ids,scores):
                if bool(torch.isnan(scores).any()|torch.isposinf(scores).any()|~torch.isfinite(scores).any(dim=-1).all()):raise RuntimeError('Nonfinite scores')
                return scores
        for model_name in MODELS:
            if model_name=='planned_adapter':
                phase('load_frozen_planned_adapter')
                model=PeftModel.from_pretrained(model,adapter_dir,is_trainable=False)
                tensors={n:hashlib.sha256(v.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest() for n,v in get_peft_model_state_dict(model,save_embedding_layers=False).items()}
                if tensors!=pack['adapter_tensor_hashes']:raise ValueError('Adapter tensor identity mismatch')
                write_json(output/'adapter_identity.json',{'file_hashes':pack['adapter_file_hashes'],'tensor_hashes_match':True,'training_called':False})
            model.eval();model.requires_grad_(False)
            baseline.validate_model_placement(model)
            if any(p.requires_grad for p in model.parameters()):raise RuntimeError('Inference weights must be frozen')
            for job,audit in [(j,a) for j,a in zip(jobs,audits) if j['model']==model_name]:
                phase('generating_'+str(len(outputs)+1)+'_of_'+str(len(jobs)));torch.manual_seed(CONFIG['seed'])
                batch=tokenize(job);ids=batch['input_ids'][0].tolist()
                if audit_tokens(job,ids)!=audit:raise ValueError('Tokenizer changed')
                batch={k:v.to('cuda:0') for k,v in batch.items()};before=time.monotonic()
                with torch.inference_mode():
                    generated=with_heartbeat(lambda:model.generate(**batch,do_sample=False,max_new_tokens=CONFIG['max_new_tokens'],
                        use_cache=True,pad_token_id=processor.tokenizer.pad_token_id,remove_invalid_values=False,
                        logits_processor=LogitsProcessorList([FiniteScores()])),'Generation')
                new=generated[0,len(ids):];eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
                text=processor.tokenizer.decode(new,skip_special_tokens=True).strip()
                out={**{k:job[k] for k in ('id','model','condition','prompt_sha256')},'output':text,'synthetic':False,
                    'hit_output_limit':bool(new.numel()>=CONFIG['max_new_tokens'] and int(new[-1]) not in eos),
                    'input_tokens':len(ids),'output_tokens':int(new.numel()),'generation_seconds':time.monotonic()-before}
                out['generation_failure']='token_cutoff' if out['hit_output_limit'] else 'empty_output' if not text else None
                outputs.append(out);write_lines(output/'generations.private.jsonl',outputs)
                status.update(completed_jobs=len(outputs),failed_generation_calls=sum(r['generation_failure'] is not None for r in outputs))
                if out['generation_failure']:print('Recorded failed sample: '+out['generation_failure'],flush=True)
        generation_memory={'peak_allocated_bytes':torch.cuda.max_memory_allocated(0),'peak_reserved_bytes':torch.cuda.max_memory_reserved(0)}
        del model,processor,batch,generated,new,tokenize;gc.collect();torch.cuda.empty_cache()
        phase('loading_semantic_diagnostics')
        tokenizer=with_heartbeat(lambda:AutoTokenizer.from_pretrained(NLI_MODEL,revision=NLI_REVISION,token=False),'NLI tokenizer')
        nli=with_heartbeat(lambda:AutoModelForSequenceClassification.from_pretrained(NLI_MODEL,revision=NLI_REVISION,token=False,dtype=torch.float32).to('cuda:0'),'NLI model')
        nli.eval();nli.requires_grad_(False);labels={int(k):v.lower() for k,v in nli.config.id2label.items()}
        if set(labels.values())!={'entailment','neutral','contradiction'}:raise ValueError('Wrong NLI labels')
        limit=min(512,int(tokenizer.model_max_length),int(getattr(nli.config,'max_position_embeddings',512)));cache={}
        def predict(a,b):
            key=(sha(a),sha(b))
            if key in cache:return cache[key]
            batch=tokenizer(a,b,truncation=False,return_tensors='pt')
            if batch['input_ids'].shape[-1]>limit:cache[key]=None;return None
            with torch.inference_mode():p=torch.softmax(nli(**{k:v.to('cuda:0') for k,v in batch.items()}).logits[0],dim=-1).tolist()
            cache[key]={labels[i]:float(x) for i,x in enumerate(p)};return cache[key]
        rows={r['id']:r for r in pack['rows']};evidence={};verdict_rows=[]
        for i,candidate in enumerate(outputs):
            phase('diagnostics_'+str(i+1)+'_of_'+str(len(outputs)))
            if candidate['hit_output_limit'] or not candidate['output'].strip():continue
            row=rows[candidate['id']]
            diagnostic={**{k:candidate[k] for k in ('id','model','condition')},
                'output_sha256':sha(candidate['output']),'source_sha256':sha(row['source']),
                'verifier':verifier.verify(row['source'],candidate['output'],predict),
                'operation_diagnostics':operation_diagnostics(row,candidate['output'],predict)}
            evidence[job_key(candidate)]=diagnostic;verdict_rows.append(diagnostic)
            write_lines(output/'verifier_results.private.jsonl',verdict_rows)
        records,summary=evaluate(pack,outputs,evidence)
        write_lines(output/'jobs.private.jsonl',jobs);write_lines(output/'metrics.private.jsonl',records);write_json(output/'evaluation.json',summary)
        write_json(output/'runtime.json',{'model_id':MODEL_ID,'revision':MODEL_REVISION,'nli_model':NLI_MODEL,'nli_revision':NLI_REVISION,
            'decoding':CONFIG,'elapsed_seconds':time.monotonic()-started,'generation_memory':generation_memory,
            'packages':{k:importlib.metadata.version(k) for k in ('torch','transformers','peft','accelerate','bitsandbytes','huggingface-hub')}})
        status.update(complete=True,all_candidates_complete=all(r['available'] for r in records));phase(summary['status'])
        write_json(output/'manifest.json',{'protocol':PROTOCOL,'pack_sha256':pack_sha,'runtime_code_files':pack['runtime_code_files'],
            'files':{p.name:file_sha(p) for p in output.iterdir() if p.is_file()}})
    except Exception as exc:
        status.update(complete=False,phase='failed',completed_jobs=len(outputs),exception_type=type(exc).__name__);write_json(output/'run_status.json',status)
        print('Inference failed; exception details suppressed to protect credentials.',flush=True)
        raise RuntimeError('Inference incomplete; inspect saved status and partial artifacts') from None


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);a=p.add_mutually_exclusive_group()
    a.add_argument('--prepare-pack',action='store_true');a.add_argument('--dry-run',action='store_true');a.add_argument('--execute-inference',action='store_true')
    p.add_argument('--pack',type=Path);p.add_argument('--expected-pack-sha');p.add_argument('--adapter-root',type=Path);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args(argv)
    if args.prepare_pack:prepare_pack(args.output_dir);return
    if args.pack is None:p.error('--pack required')
    pack=load_pack(args.pack,args.expected_pack_sha)
    adapter=adapter_directory(args.adapter_root,pack) if args.adapter_root else None
    if args.execute_inference:
        if adapter is None:p.error('--adapter-root required for inference')
        run_training(pack,args.output_dir,args.expected_pack_sha,adapter);return
    fresh(args.output_dir)
    if args.dry_run:
        outputs=[{**{k:j[k] for k in ('id','model','condition','prompt_sha256')},'output':'Synthetic placeholder.','synthetic':True,'hit_output_limit':False} for j in jobs_for(pack)]
        _,summary=evaluate(pack,outputs,synthetic=True);write_json(args.output_dir/'synthetic_evaluation.json',summary)
    else:write_json(args.output_dir/'preflight.json',{'status':'preflight_passed','generation_calls':30,'adapter_file_identity_checked':adapter is not None,'training_called':False})
    print('CPU check passed; no Secret, model weights load or GPU accessed.')


if __name__=='__main__':main()
