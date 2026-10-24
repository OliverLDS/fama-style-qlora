"""Matched 21-example, count-free soft-guidance inference; no trainer/adapter."""
import argparse
from collections import Counter
import gc
import importlib.metadata
import json
from pathlib import Path
import re
import statistics
import time

from .soft_guidance import jobs_for,soft_plan,VALIDATION_SHA,CONDITIONS
from .concise_plan import TRAIN_SHA,MODEL_ID,MODEL_REVISION,NLI_MODEL,NLI_REVISION,file_sha,read_lines
from .experiment import fresh,write_json,write_lines,secret_token,with_heartbeat
from .claim_constrained import calibration_controls
from fama_style.research import claim_verifier_v3 as verifier
from fama_style.research.claims import sha,sentences,tokens
from fama_style.research.signal import fit_profile,distance,overlap,features

ROOT=Path(__file__).resolve().parents[3]
PROTOCOL='soft_guidance_validation_v2'
MAX_NEW_TOKENS=768
CONTEXT_LIMIT=4096
VERIFIER_FOLDER=ROOT/'docs/claim_verifier_calibration_v3'


def runtime_paths():
    paths=[ROOT/'src/fama_style/__init__.py',ROOT/'src/fama_style/prompting/__init__.py',ROOT/'src/fama_style/prompting/corpus.py']
    return paths+sorted((ROOT/'src/fama_style/research').glob('*.py'))+sorted((ROOT/'src/fama_style/rhetoric').glob('*.py'))


def prepare_pack(output):
    training=ROOT/'data/fama_corpus_v1/train.jsonl';validation=ROOT/'data/fama_corpus_v1/validation.jsonl'
    if file_sha(training)!=TRAIN_SHA or file_sha(validation)!=VALIDATION_SHA:raise ValueError('Frozen train/validation bytes changed')
    train,val=read_lines(training),read_lines(validation)
    if len(train)!=247 or any(r['split']!='train' for r in train):raise ValueError('Wrong training split')
    if len(val)!=21 or any(r['split']!='validation' for r in val):raise ValueError('Wrong validation split')
    if {r['id'] for r in train}&{r['id'] for r in val} or {r['paper_key'] for r in train}&{r['paper_key'] for r in val}:raise ValueError('Overlapping papers or IDs')
    base=ROOT/'research_runs/claim_verifier_calibration_v3';run=base/'kaggle_version_1/verifier_calibration_v3'
    review_path=base/'version_1_review/verification.json';review=json.loads(review_path.read_text())
    if not review['checks'] or not all(review['checks'].values()):raise ValueError('Calibration review did not pass')
    for name,digest in review['artifact_hashes'].items():
        if file_sha(run/name)!=digest:raise ValueError('Reviewed calibration changed')
    result_path=run/'evaluation.json';result=json.loads(result_path.read_text());manifest=json.loads((run/'manifest.json').read_text())
    protocol,_=verifier.validate_protocol(VERIFIER_FOLDER)
    if (result['protocol']!=verifier.PROTOCOL or result['execution_kind']!='pinned_nli_inference'
        or result['readiness_gate']!=verifier.gate(result['splits']['held_out'],protocol['readiness_gate'])
        or result['readiness_gate']['passed'] is not True or manifest['code_files']!=protocol['code_files']
        or manifest['protocol_sha256']!=file_sha(VERIFIER_FOLDER/'protocol.json')):raise ValueError('Unusable verifier calibration')
    runtime=json.loads((run/'runtime.json').read_text())
    if runtime['model_id']!=NLI_MODEL or runtime['model_revision']!=NLI_REVISION:raise ValueError('Changed NLI revision')
    pack={'protocol':PROTOCOL,'parent_training_sha256':TRAIN_SHA,'validation_sha256':VALIDATION_SHA,
        'style_profile':fit_profile(train),
        'rows':[{'id':r['id'],'paper_key':r['paper_key'],'split':'validation','source':r['plain_english'],'reference':r['completion']} for r in val],
        'execution_gate':{'ready':True,'reason':'Reviewed v3 synthetic gate passed; exploratory inference only, not domain reliability'},
        'verifier_evidence':{'notebook':'oliverlds/verifier_calibration_v3','version':1,'protocol':verifier.PROTOCOL,
            'evaluation_sha256':file_sha(result_path),'review_sha256':file_sha(review_path),'held_out_metrics':result['splits']['held_out'],
            'protocol_sha256':manifest['protocol_sha256'],'code_files':protocol['code_files'],'fixture_files':protocol['fixture_files']},
        'generation':{'model_id':MODEL_ID,'revision':MODEL_REVISION,'max_new_tokens':MAX_NEW_TOKENS,'context_limit':CONTEXT_LIMIT,'do_sample':False,'seed':42,'truncate':False},
        'scope':'Existing 21 development examples. Fixed training-derived soft prior, not a learned full-taxonomy mapper. References only for post-generation diagnostics. Reserved paper test split never read.'}
    jobs=jobs_for(pack)
    pack['runtime_code_files']={str(p.relative_to(ROOT)):file_sha(p) for p in runtime_paths()}
    pack['jobs_sha256']=sha(json.dumps(jobs,sort_keys=True,ensure_ascii=False))
    fresh(output);path=output/'soft_guidance_pack.json';write_json(path,pack)
    write_lines(output/'jobs.private.jsonl',jobs)
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'status':'prepared_not_executed','pairs':21,'jobs':42,'pack_sha256':file_sha(path),
        'training_called':False,'model_calls':0,'reserved_test_rows_used':0,
        'input_hashes':{str(p.relative_to(ROOT)):file_sha(p) for p in (training,validation,result_path,review_path)},
        'files':{p.name:file_sha(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({'status':'prepared_not_executed','pairs':21,'jobs':42,'pack_sha256':file_sha(path)}));return pack


def load_pack(path,expected_sha):
    if not re.fullmatch('[0-9a-f]{64}',expected_sha) or file_sha(path)!=expected_sha:raise ValueError('Input pack hash mismatch')
    pack=json.loads(path.read_text())
    if pack['protocol']!=PROTOCOL or pack['parent_training_sha256']!=TRAIN_SHA or pack['validation_sha256']!=VALIDATION_SHA:raise ValueError('Wrong frozen data/protocol')
    if pack['execution_gate']['ready'] is not True:raise ValueError('Verifier gate did not pass')
    for name,h in pack['runtime_code_files'].items():
        if file_sha(ROOT/name)!=h:raise ValueError('Frozen inference implementation changed')
    profile=pack['style_profile']
    if profile['fit_split']!='train' or profile['training_rows']!=247:raise ValueError('Wrong style-profile split')
    evidence=pack['verifier_evidence'];protocol,_=verifier.validate_protocol(VERIFIER_FOLDER)
    if evidence['protocol_sha256']!=file_sha(VERIFIER_FOLDER/'protocol.json') or evidence['code_files']!=protocol['code_files'] or evidence['fixture_files']!=protocol['fixture_files']:raise ValueError('Verifier evidence changed')
    if not verifier.gate(evidence['held_out_metrics'],protocol['readiness_gate'])['passed']:raise ValueError('Synthetic calibration is not ready')
    if pack['generation']!={'model_id':MODEL_ID,'revision':MODEL_REVISION,'max_new_tokens':MAX_NEW_TOKENS,'context_limit':CONTEXT_LIMIT,'do_sample':False,'seed':42,'truncate':False}:raise ValueError('Changed generation controls')
    if pack['jobs_sha256']!=sha(json.dumps(jobs_for(pack),sort_keys=True,ensure_ascii=False)):raise ValueError('Prepared prompts changed')
    if any(not r['source'].strip() or not r['reference'].strip() for r in pack['rows']):raise ValueError('Empty pair')
    return pack


def audit_tokens(job,ids):
    if sha(job['prompt'])!=job['prompt_sha256'] or not ids or len(ids)+MAX_NEW_TOKENS>CONTEXT_LIMIT:raise ValueError('Changed/overflowing prompt; no truncation allowed')
    return {'id':job['id'],'condition':job['condition'],'prompt_sha256':job['prompt_sha256'],'input_tokens':len(ids),'input_ids_sha256':sha(json.dumps(ids))}


def paragraph_format(text):
    return not bool(re.match(r'\s*(?:```|[\[{]|#+\s|[-*]\s|\d+\.\s)',text) or re.search(r'\n\s*\n',text))


def evaluate(pack,jobs,outputs,semantic=None,pair_checks=None,canary_passed=False,synthetic=False):
    if jobs!=jobs_for(pack) or len(outputs)!=len(jobs):raise ValueError('Changed jobs or incomplete outputs')
    expected={(j['id'],j['condition']):j for j in jobs};seen=set();rows={r['id']:r for r in pack['rows']};records=[]
    if canary_passed and not synthetic and set(semantic or {})!=set(expected):raise ValueError('Incomplete verifier results')
    for out in outputs:
        key=(out['id'],out['condition']);job=expected.get(key)
        if key in seen or job is None or out['prompt_sha256']!=job['prompt_sha256']:raise ValueError('Duplicate/swapped output')
        if not out.get('output','').strip() or out.get('hit_output_limit') is not False or out.get('synthetic') is not synthetic:raise ValueError('Empty/cutoff/mislabeled output')
        seen.add(key);row=rows[out['id']];text=out['output'];verdict=(semantic or {}).get(key);valid=paragraph_format(text)
        status=verdict['status'] if verdict else 'not_run';sim=overlap(row['source'],text)
        if verdict is not None and status not in ('model_consistent','risk_signal','abstain'):raise ValueError('Unknown verifier status')
        gain=distance(row['source'],pack['style_profile'])-distance(text,pack['style_profile'])
        usable=canary_passed and not synthetic and verdict is not None
        records.append({'id':row['id'],'condition':out['condition'],'output_sha256':sha(text),'format_valid':valid,
            'verifier_status':status,'verifier':verdict,'source_similarity':sim,'near_copy':sim>=.9,
            'exact_copy':' '.join(row['source'].split())==' '.join(text.split()),'token_normalized_copy':tokens(row['source'])==tokens(text),
            'style_gain_proxy':gain,'source_features':features(row['source']),'output_features':features(text),
            'source_sentence_count':len(sentences(row['source'])),'output_sentence_count':len(sentences(text)),
            'reference_similarity_diagnostic_only':overlap(row['reference'],text),
            'valid_semantic_proxy_pass':valid and status=='model_consistent' if usable else None,
            'joint_proxy_success':valid and status=='model_consistent' and sim<.9 and gain>0 if usable else None,
            'input_tokens':out.get('input_tokens',0),'output_tokens':out.get('output_tokens',0),'generation_seconds':out.get('generation_seconds',0)})
    aggregate={}
    for c in CONDITIONS:
        group=[r for r in records if r['condition']==c];valid=[r for r in group if r['format_valid']]
        aggregate[c]={'rows':len(group),'valid_formats':len(valid),'verifier_status_counts':dict(Counter(r['verifier_status'] for r in group)),
            'valid_semantic_proxy_passes':sum(bool(r['valid_semantic_proxy_pass']) for r in group) if canary_passed and not synthetic else None,
            'joint_proxy_successes':sum(bool(r['joint_proxy_success']) for r in group) if canary_passed and not synthetic else None,
            'near_copies':sum(r['near_copy'] for r in group),'exact_copies':sum(r['exact_copy'] for r in group),
            'mean_style_gain_proxy_all_outputs':statistics.mean(r['style_gain_proxy'] for r in group),
            'mean_style_gain_proxy_valid_only':statistics.mean(r['style_gain_proxy'] for r in valid) if valid else None,
            'mean_source_similarity':statistics.mean(r['source_similarity'] for r in group),
            **{k:sum(r[k] or 0 for r in group) for k in ('input_tokens','output_tokens','generation_seconds')}}
    lookup={(r['id'],r['condition']):r for r in records};paired=[]
    for row in pack['rows']:
        a,b=(lookup[(row['id'],c)] for c in CONDITIONS)
        paired.append({'id':row['id'],'A_verifier_status':a['verifier_status'],'B_verifier_status':b['verifier_status'],
            'A_format_valid':a['format_valid'],'B_format_valid':b['format_valid'],
            'style_gain_difference_B_minus_A':b['style_gain_proxy']-a['style_gain_proxy'],
            'source_similarity_difference_B_minus_A':b['source_similarity']-a['source_similarity'],
            'A_joint_proxy_success':a['joint_proxy_success'],'B_joint_proxy_success':b['joint_proxy_success']})
    return records,{'protocol':PROTOCOL,'execution_kind':'synthetic_wiring_check' if synthetic else 'model_inference',
        'conditions':aggregate,'paired':paired,'pair_source_reference_verifier':pair_checks,'runtime_canary_passed':canary_passed,
        'training_called':False,'reserved_test_rows_used':0,'development_validation_rows_used':21,'model_selection_test_rows_used':0,
        'training_ready':False,'semantic_accuracy':None,'author_style_accuracy':None,'oracle':False,
        'limitations':'Previously used development set from one paper. Fixed source-bound soft prior, not a learned full-taxonomy mapper. Verifier consistency and style distance are fallible proxies; unknowns and bad formats remain in all-example denominators. No hard sentence quotas or automatic repairs.'}


def run_training(pack,output,uploaded_pack_sha):
    """Inference only. Check public verifier canary before accessing the Gemma Secret."""
    fresh(output);jobs=jobs_for(pack);write_lines(output/'jobs.private.jsonl',jobs)
    write_json(output/'uploaded_pack_identity.json',{'sha256':uploaded_pack_sha,'format':'uploaded file bytes'})
    status={'complete':False,'protocol':PROTOCOL,'training_called':False,'reserved_test_rows_used':0,'completed_jobs':0,'total_jobs':len(jobs)}
    started=time.monotonic();outputs=[]
    def phase(name,**extra):
        status.update(phase=name,**extra);write_json(output/'run_status.json',status);print('Phase: '+name,flush=True)
    try:
        phase('loading_inference_runtime')
        import os
        os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
        os.environ['HF_HUB_DISABLE_TELEMETRY']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
        import torch
        from transformers import (AutoProcessor,AutoTokenizer,AutoModelForSequenceClassification,
            Gemma3ForConditionalGeneration,BitsAndBytesConfig,LogitsProcessor,LogitsProcessorList)
        from transformers.utils import logging
        logging.set_verbosity_error()
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Expose one Kaggle GPU before importing torch')
        torch.cuda.init();torch.cuda.set_device(0);torch.manual_seed(42)
        if 'T4' not in torch.cuda.get_device_name(0):raise RuntimeError('Select Kaggle T4 x2')
        cache={};cache_hits=0
        def load_nli():
            tokenizer=with_heartbeat(lambda:AutoTokenizer.from_pretrained(NLI_MODEL,revision=NLI_REVISION,token=False),'NLI tokenizer load')
            nli=with_heartbeat(lambda:AutoModelForSequenceClassification.from_pretrained(NLI_MODEL,revision=NLI_REVISION,
                token=False,dtype=torch.float32).to('cuda:0'),'NLI model load')
            nli.eval();nli.requires_grad_(False);labels={int(k):v.lower() for k,v in nli.config.id2label.items()}
            if set(labels.values())!={'entailment','neutral','contradiction'}:raise ValueError('Unexpected NLI labels')
            limit=min(512,int(tokenizer.model_max_length),int(getattr(nli.config,'max_position_embeddings',512)))
            def predict(premise,hypothesis):
                nonlocal cache_hits
                key=(sha(premise),sha(hypothesis))
                if key in cache:cache_hits+=1;return cache[key]
                batch=tokenizer(premise,hypothesis,truncation=False,return_tensors='pt')
                if int(batch['input_ids'].shape[-1])>limit:cache[key]=None;return None
                with torch.inference_mode():
                    probs=torch.softmax(nli(**{k:v.to('cuda:0') for k,v in batch.items()}).logits[0],dim=-1).tolist()
                cache[key]={labels[i]:float(v) for i,v in enumerate(probs)};return cache[key]
            return tokenizer,nli,predict
        phase('runtime_verifier_canary');torch.cuda.reset_peak_memory_stats(0);canary_started=time.monotonic()
        nli_tokenizer,nli,predict=load_nli()
        canary=verifier.evaluate(VERIFIER_FOLDER,output/'runtime_calibration',predict,'pinned_nli_inference')
        canary_seconds=time.monotonic()-canary_started
        if canary['readiness_gate']['passed'] is not True:raise RuntimeError('Runtime verifier canary failed; no Gemma generation performed')
        del nli_tokenizer,nli,predict;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(0)
        phase('checking_secret');token=secret_token()
        processor=with_heartbeat(lambda:AutoProcessor.from_pretrained(MODEL_ID,revision=MODEL_REVISION,token=token),'Processor load')
        def tokenize(job):
            return processor.apply_chat_template([{'role':'user','content':[{'type':'text','text':job['prompt']}]}],
                tokenize=True,add_generation_prompt=True,truncation=False,return_dict=True,return_tensors='pt')
        audits=[audit_tokens(job,tokenize(job)['input_ids'][0].tolist()) for job in jobs];write_json(output/'token_audit.json',audits)
        phase('loading_pinned_gemma')
        model=with_heartbeat(lambda:Gemma3ForConditionalGeneration.from_pretrained(MODEL_ID,revision=MODEL_REVISION,
            token=token,dtype=torch.float32,device_map={'':0},attn_implementation='sdpa',
            quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float32)),'Gemma load')
        del token
        if any(t.device.type!='cuda' or t.device.index!=0 for t in list(model.parameters())+list(model.buffers())):raise RuntimeError('Unexpected model placement')
        model.eval();model.requires_grad_(False)
        class FiniteScores(LogitsProcessor):
            def __call__(self,input_ids,scores):
                if bool(torch.isnan(scores).any()|torch.isposinf(scores).any()|~torch.isfinite(scores).any(dim=-1).all()):raise RuntimeError('Nonfinite generation scores')
                return scores
        by_key={(r['id'],r['condition']):r for r in audits}
        for job in jobs:
            phase('generation',completed_jobs=len(outputs));batch=tokenize(job);ids=batch['input_ids'][0].tolist()
            if audit_tokens(job,ids)!=by_key[(job['id'],job['condition'])]:raise ValueError('Tokenized prompt changed')
            batch={k:v.to(model.get_input_embeddings().weight.device) for k,v in batch.items()};before=time.monotonic()
            with torch.inference_mode():
                generated=with_heartbeat(lambda:model.generate(**batch,do_sample=False,max_new_tokens=MAX_NEW_TOKENS,use_cache=True,
                    pad_token_id=processor.tokenizer.pad_token_id,remove_invalid_values=False,
                    logits_processor=LogitsProcessorList([FiniteScores()])),'Generation')
            new=generated[0,len(ids):];eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
            text=processor.tokenizer.decode(new,skip_special_tokens=True).strip()
            result={'id':job['id'],'condition':job['condition'],'prompt_sha256':job['prompt_sha256'],'output':text,
                'input_tokens':len(ids),'output_tokens':int(new.numel()),'generation_seconds':time.monotonic()-before,
                'hit_output_limit':bool(new.numel()>=MAX_NEW_TOKENS and int(new[-1]) not in eos),'synthetic':False}
            outputs.append(result);write_lines(output/'generations.private.jsonl',outputs)
            if not text or result['hit_output_limit']:raise RuntimeError('Empty/cutoff output; partial results remain incomplete')
        gemma_memory={'peak_allocated_bytes':torch.cuda.max_memory_allocated(0),'peak_reserved_bytes':torch.cuda.max_memory_reserved(0)}
        del model,processor,batch,generated,new,tokenize;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(0)
        phase('loading_verifier_after_gemma_unload');nli_started=time.monotonic();nli_tokenizer,nli,predict=load_nli()
        semantic={};pair_checks={};sources={r['id']:r['source'] for r in pack['rows']}
        for index,row in enumerate(pack['rows'],1):
            phase('pair_fidelity_assessment',completed_pairs=index-1)
            pair_checks[row['id']]=with_heartbeat(lambda:verifier.verify(row['source'],row['reference'],predict),'Pair verifier')
        for index,result in enumerate(outputs,1):
            phase('output_semantic_assessment',completed_verifier_outputs=index-1)
            semantic[(result['id'],result['condition'])]=with_heartbeat(lambda:verifier.verify(sources[result['id']],result['output'],predict),'Output verifier')
        records,summary=evaluate(pack,jobs,outputs,semantic,pair_checks,canary_passed=True)
        write_lines(output/'metrics.private.jsonl',records)
        write_lines(output/'nli_probability_cache.private.jsonl',[{'premise_sha256':a,'hypothesis_sha256':b,'probabilities':p} for (a,b),p in cache.items()])
        summary.update(model_id=MODEL_ID,model_revision=MODEL_REVISION,nli_model=NLI_MODEL,nli_revision=NLI_REVISION,
            canary_seconds=canary_seconds,verifier_seconds=time.monotonic()-nli_started,total_seconds=time.monotonic()-started,
            nli_cache_entries=len(cache),nli_cache_hits=cache_hits,gemma_memory=gemma_memory,
            nli_memory={'peak_allocated_bytes':torch.cuda.max_memory_allocated(0),'peak_reserved_bytes':torch.cuda.max_memory_reserved(0)},
            generation=pack['generation'],versions={n:importlib.metadata.version(n) for n in ('torch','transformers','accelerate','bitsandbytes','huggingface-hub')})
        write_json(output/'evaluation.json',summary)
        phase('complete',complete=True,completed_jobs=42,completed_pairs=21,completed_verifier_outputs=42)
        write_json(output/'manifest.json',{'protocol':PROTOCOL,'training_called':False,'input_pack_sha256':uploaded_pack_sha,
            'files':{str(p.relative_to(output)):file_sha(p) for p in output.rglob('*') if p.is_file()}})
    except Exception as error:
        failed_at=status.get('phase');locations=[];tb=error.__traceback__
        while tb:
            if tb.tb_frame.f_code.co_filename==__file__:locations.append(tb.tb_lineno)
            tb=tb.tb_next
        phase('failed',complete=False,failed_at=failed_at,error_code='soft_guidance_inference_failed',local_line_numbers=locations)
        raise RuntimeError('Soft-guidance inference failed; inspect run_status.json and saved partial files. No training was called.') from None


def synthetic_pack():
    examples=[{'split':'train','plain_english':'Firms report sales. Investors read reports.','completion':'Firms report sales; investors read reports.'}]
    return {'protocol':PROTOCOL,'style_profile':fit_profile(examples),
        'rows':[{'id':'synthetic-'+str(i),'paper_key':'synthetic-validation','split':'validation','source':'Firms report sales. Investors read reports.',
                 'reference':'Firms report sales; investors read reports.'} for i in range(21)]}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);actions=p.add_mutually_exclusive_group()
    actions.add_argument('--prepare-pack',action='store_true');actions.add_argument('--dry-run',action='store_true');actions.add_argument('--execute-inference',action='store_true')
    p.add_argument('--pack',type=Path);p.add_argument('--expected-pack-sha');p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args(argv)
    if a.prepare_pack:prepare_pack(a.output_dir);return
    if a.dry_run:
        pack=synthetic_pack();jobs=jobs_for(pack);by_id={r['id']:r for r in pack['rows']};fresh(a.output_dir)
        outputs=[{'id':j['id'],'condition':j['condition'],'prompt_sha256':j['prompt_sha256'],'output':by_id[j['id']]['source'],'hit_output_limit':False,'synthetic':True} for j in jobs]
        records,summary=evaluate(pack,jobs,outputs,synthetic=True)
        write_json(a.output_dir/'synthetic_evaluation.json',summary);write_lines(a.output_dir/'synthetic_jobs.jsonl',jobs)
        print('Synthetic wiring check: 42 jobs; no model calls or training.');return
    if a.pack is None or a.expected_pack_sha is None:p.error('--pack and --expected-pack-sha required')
    pack=load_pack(a.pack,a.expected_pack_sha)
    if a.execute_inference:run_training(pack,a.output_dir,a.expected_pack_sha)
    else:
        fresh(a.output_dir);write_json(a.output_dir/'preflight.json',{'protocol':PROTOCOL,'pack_sha256':a.expected_pack_sha,'pairs':21,'jobs':42,'model_calls':0,'training_called':False})
        print('Preflight passed; no Secret access, model download or inference.')


if __name__=='__main__':main()
