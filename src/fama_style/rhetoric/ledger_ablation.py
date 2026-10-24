"""Two-case, 2x2 ledger-format/grouping ablation; inference only, post-hoc."""
import argparse
from collections import Counter
import copy
import gc
import importlib.metadata
import json
from pathlib import Path
import re
import statistics
import time

from . import soft_guidance_experiment as parent
from .soft_guidance import jobs_for as parent_jobs
from .claim_constrained import constraint_ledger
from .concise_plan import TRAIN_SHA,MODEL_ID,MODEL_REVISION,NLI_MODEL,NLI_REVISION,file_sha,read_lines
from .experiment import fresh,write_json,write_lines,secret_token,with_heartbeat
from fama_style.prompting.corpus import make_prompt
from fama_style.research import claim_verifier_v3 as verifier
from fama_style.research.claims import sha,sentences,tokens
from fama_style.research.signal import distance,overlap,features

ROOT=Path(__file__).resolve().parents[3]
PROTOCOL='ledger_grouping_ablation_v1'
MAX_NEW_TOKENS=768
CONTEXT_LIMIT=4096
VERIFIER_FOLDER=ROOT/'docs/claim_verifier_calibration_v3'
CONDITIONS=('A_sentence_only','B_sentence_grouping','C_cue_rich','D_cue_rich_grouping')
FACTORS={c:{'ledger':'sentence_evidence' if c.startswith(('A_','B_')) else 'cue_rich',
    'grouping':c.startswith(('B_','D_'))} for c in CONDITIONS}
audit_tokens=parent.audit_tokens
paragraph_format=parent.paragraph_format
runtime_paths=parent.runtime_paths


def jobs_for(pack):
    rows=pack['rows']
    if len(rows)!=2 or len({r['id'] for r in rows})!=2 or any(r['split']!='validation' for r in rows):raise ValueError('Expected two unique development rows')
    jobs=[]
    for index,row in enumerate(rows):
        rich=constraint_ledger(row['source']);simple=copy.deepcopy(rich)
        for claim in simple['claims']:del claim['protected_evidence']
        rich_json=json.dumps(rich,ensure_ascii=False,separators=(',',':'))
        simple_json=json.dumps(simple,ensure_ascii=False,separators=(',',':'))
        for c in CONDITIONS if index==0 else CONDITIONS[::-1]:
            factors=FACTORS[c];prompt=row['parent_prompts']['grouping' if factors['grouping'] else 'no_grouping']
            if prompt.count(rich_json)!=1:raise ValueError('Original prompt ledger mismatch')
            if not prompt.startswith(make_prompt(row['source'])):raise ValueError('Original source prefix mismatch')
            if factors['ledger']=='sentence_evidence':prompt=prompt.replace(rich_json,simple_json,1)
            jobs.append({'id':row['id'],'condition':c,'prompt':prompt,'prompt_sha256':sha(prompt),'source_sha256':sha(row['source']),**factors,'oracle':False})
    return jobs


def prepare_pack(output):
    base=ROOT/'research_runs/soft_guidance_validation_v2';parent_path=base/'input_pack/soft_guidance_pack.json'
    release=json.loads((base/'release_verification.json').read_text());prior=parent.load_pack(parent_path,release['pack_sha256'])
    run=base/'kaggle_version_1/soft_guidance_validation_v2';review_path=base/'version_1_review/verification.json';review=json.loads(review_path.read_text())
    if not review['checks'] or not all(review['checks'].values()):raise ValueError('Prior run review failed')
    for name,digest in review['artifact_hashes'].items():
        if file_sha(run/name)!=digest:raise ValueError('Reviewed result changed')
    metrics=read_lines(run/'metrics.private.jsonl');ranked=sorted((r for r in metrics if r['condition']=='B_claims_soft_plan'),key=lambda r:(r['style_gain_proxy'],r['id']))[:2]
    chosen={r['id'] for r in ranked};original_jobs={(j['id'],j['condition']):j for j in parent_jobs(prior)}
    outputs={(r['id'],r['condition']):r for r in read_lines(run/'generations.private.jsonl')}
    rows=[]
    for row in prior['rows']:
        if row['id'] not in chosen:continue
        rows.append({**row,'parent_prompts':{'no_grouping':original_jobs[(row['id'],'A_claims_only')]['prompt'],'grouping':original_jobs[(row['id'],'B_claims_soft_plan')]['prompt']},
            'previous_rich_output_sha256':{'C_cue_rich':sha(outputs[(row['id'],'A_claims_only')]['output']),'D_cue_rich_grouping':sha(outputs[(row['id'],'B_claims_soft_plan')]['output'])}})
    pack={k:copy.deepcopy(prior[k]) for k in ('parent_training_sha256','validation_sha256','style_profile','execution_gate','verifier_evidence','generation')}
    pack.update(protocol=PROTOCOL,rows=rows,parent_pack_sha256=file_sha(parent_path),
        selection={'post_hoc':True,'rule':'Two lowest B style-gain proxies in reviewed 21-example run; ascending score then ID; both exposed failures',
            'review_sha256':file_sha(review_path),'parent_metrics_sha256':file_sha(run/'metrics.private.jsonl'),'held_out_test_rows_read':0},
        scope='Post-hoc two-case mechanism diagnostic; same master instruction, source evidence, model, decoding and grouping texts. Only protected_evidence fields are removed in the simple-ledger arms.')
    jobs=jobs_for(pack);pack['runtime_code_files']={str(p.relative_to(ROOT)):file_sha(p) for p in runtime_paths()};pack['jobs_sha256']=sha(json.dumps(jobs,sort_keys=True,ensure_ascii=False))
    fresh(output);path=output/'ledger_ablation_pack.json';write_json(path,pack);write_lines(output/'jobs.private.jsonl',jobs)
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'status':'prepared_not_executed','pairs':2,'jobs':8,'pack_sha256':file_sha(path),
        'training_called':False,'model_calls':0,'reserved_test_rows_read':0,'parent_pack_sha256':file_sha(parent_path),
        'files':{p.name:file_sha(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({'status':'prepared_not_executed','pairs':2,'jobs':8,'pack_sha256':file_sha(path)}));return pack


def load_pack(path,expected_sha):
    if not re.fullmatch('[0-9a-f]{64}',expected_sha) or file_sha(path)!=expected_sha:raise ValueError('Input pack hash mismatch')
    pack=json.loads(path.read_text())
    if pack['protocol']!=PROTOCOL or pack['parent_training_sha256']!=TRAIN_SHA or pack['validation_sha256']!=parent.VALIDATION_SHA:raise ValueError('Wrong frozen protocol/data')
    if pack['execution_gate']['ready'] is not True or pack['style_profile']['fit_split']!='train' or pack['style_profile']['training_rows']!=247:raise ValueError('Wrong gate/profile')
    for name,h in pack['runtime_code_files'].items():
        if file_sha(ROOT/name)!=h:raise ValueError('Frozen implementation changed')
    protocol,_=verifier.validate_protocol(VERIFIER_FOLDER);evidence=pack['verifier_evidence']
    if evidence['protocol_sha256']!=file_sha(VERIFIER_FOLDER/'protocol.json') or evidence['code_files']!=protocol['code_files'] or evidence['fixture_files']!=protocol['fixture_files']:raise ValueError('Verifier changed')
    if not verifier.gate(evidence['held_out_metrics'],protocol['readiness_gate'])['passed']:raise ValueError('Verifier gate failed')
    if pack['generation']!={'model_id':MODEL_ID,'revision':MODEL_REVISION,'max_new_tokens':768,'context_limit':4096,'do_sample':False,'seed':42,'truncate':False}:raise ValueError('Changed generation controls')
    if pack['jobs_sha256']!=sha(json.dumps(jobs_for(pack),sort_keys=True,ensure_ascii=False)):raise ValueError('Prepared prompts changed')
    return pack


def artifact_checks(source,text):
    cues={x['evidence'].lower() for c in constraint_ledger(source)['claims'] for x in c['protected_evidence'] if x['kind']!='quantities' and len(x['evidence'].split())<=3}
    quoted=[]
    for cue in sorted(cues):
        pattern=r"(?<!\w)[‘\"']"+re.escape(cue)+r"[’\"'](?!\w)"
        if re.search(pattern,text,re.I) and not re.search(pattern,source,re.I):quoted.append(cue)
    terms=('modality','qualifier','quantifier','ledger','annotation','source evidence','claim_id','protected_evidence')
    labels=[t for t in terms if re.search(r'\b'+re.escape(t)+r'\b',text,re.I) and not re.search(r'\b'+re.escape(t)+r'\b',source,re.I)]
    year_phrases=re.findall(r'\b(?:quantitative|statistical|numerical)\s+evidence\s+(?:of|for|from)\s+(?:the\s+year\s+)?(?:18|19|20)\d{2}\b',text,re.I)
    year_phrases=[p for p in year_phrases if p.lower() not in source.lower()]
    return {'artifact_warning':bool(quoted or labels or year_phrases),'newly_quoted_protected_cues':quoted,'new_metalinguistic_terms':labels,
        'new_numeric_evidence_phrases':year_phrases,'semantic_proof':False,'meaning':'Predeclared surface artifact screen, not confirmed added-claim detection'}


def evaluate(pack,jobs,outputs,semantic=None,pair_checks=None,canary_passed=False,synthetic=False):
    if jobs!=jobs_for(pack) or len(outputs)!=8:raise ValueError('Changed jobs or incomplete generation set')
    expected={(j['id'],j['condition']):j for j in jobs};seen=set();rows={r['id']:r for r in pack['rows']};records=[]
    if canary_passed and not synthetic and set(semantic or {})!=set(expected):raise ValueError('Incomplete verifier results')
    for out in outputs:
        key=(out['id'],out['condition']);job=expected.get(key)
        if key in seen or job is None or out['prompt_sha256']!=job['prompt_sha256']:raise ValueError('Duplicate/swapped output')
        if not out.get('output','').strip() or out.get('hit_output_limit') is not False or out.get('synthetic') is not synthetic:raise ValueError('Empty/cutoff/mislabeled output')
        seen.add(key);row=rows[out['id']];text=out['output'];v=(semantic or {}).get(key);valid=paragraph_format(text);status=v['status'] if v else 'not_run'
        if v and status not in ('model_consistent','risk_signal','abstain'):raise ValueError('Unknown verifier status')
        usable=canary_passed and not synthetic and v is not None
        records.append({'id':out['id'],'condition':out['condition'],**FACTORS[out['condition']],**artifact_checks(row['source'],text),
            'format_valid':valid,'verifier_status':status,'verifier':v,'output_sha256':sha(text),
            'source_sentence_count':len(sentences(row['source'])),'output_sentence_count':len(sentences(text)),
            'source_features':features(row['source']),'output_features':features(text),'reference_features_diagnostic_only':features(row['reference']),
            'style_gain_proxy':distance(row['source'],pack['style_profile'])-distance(text,pack['style_profile']),
            'source_similarity':overlap(row['source'],text),'near_copy':overlap(row['source'],text)>=.9,
            'exact_copy':' '.join(row['source'].split())==' '.join(text.split()),
            'valid_semantic_proxy_pass':valid and status=='model_consistent' if usable else None,
            'previous_rich_output_hash_matches':sha(text)==row.get('previous_rich_output_sha256',{}).get(out['condition']) if out['condition'].startswith(('C_','D_')) else None,
            'input_tokens':out.get('input_tokens',0),'output_tokens':out.get('output_tokens',0),'generation_seconds':out.get('generation_seconds',0)})
    conditions={}
    for c in CONDITIONS:
        group=[r for r in records if r['condition']==c]
        conditions[c]={'rows':2,'artifact_warning_outputs':sum(r['artifact_warning'] for r in group),'single_sentence_outputs':sum(r['output_sentence_count']==1 for r in group),
            'mean_words_per_sentence':statistics.mean(r['output_features']['words_per_sentence'] for r in group),
            'mean_style_gain_proxy':statistics.mean(r['style_gain_proxy'] for r in group),'mean_source_similarity':statistics.mean(r['source_similarity'] for r in group),
            'near_copies':sum(r['near_copy'] for r in group),'valid_formats':sum(r['format_valid'] for r in group),
            'verifier_status_counts':dict(Counter(r['verifier_status'] for r in group)),
            'valid_semantic_proxy_passes':sum(bool(r['valid_semantic_proxy_pass']) for r in group) if canary_passed and not synthetic else None,
            'rich_control_hash_matches':sum(r['previous_rich_output_hash_matches'] is True for r in group) if c.startswith(('C_','D_')) else None,
            **{k:sum(r[k] or 0 for r in group) for k in ('input_tokens','output_tokens','generation_seconds')}}
    lookup={(r['id'],r['condition']):r for r in records};contrasts=[]
    for row in pack['rows']:
        for label,a,b in (('remove_cues_without_grouping','C_cue_rich','A_sentence_only'),('remove_cues_with_grouping','D_cue_rich_grouping','B_sentence_grouping'),
                          ('add_grouping_sentence_evidence','A_sentence_only','B_sentence_grouping'),('add_grouping_cue_rich','C_cue_rich','D_cue_rich_grouping')):
            before,after=lookup[(row['id'],a)],lookup[(row['id'],b)]
            contrasts.append({'id':row['id'],'contrast':label,'before':a,'after':b,
                'artifact_before':before['artifact_warning'],'artifact_after':after['artifact_warning'],
                'words_per_sentence_delta':after['output_features']['words_per_sentence']-before['output_features']['words_per_sentence'],
                'style_gain_delta':after['style_gain_proxy']-before['style_gain_proxy'],
                'verifier_before':before['verifier_status'],'verifier_after':after['verifier_status']})
    return records,{'protocol':PROTOCOL,'execution_kind':'synthetic_wiring_check' if synthetic else 'model_inference',
        'conditions':conditions,'within_case_contrasts':contrasts,'runtime_canary_passed':canary_passed,'pair_source_reference_verifier':pair_checks,
        'training_called':False,'reserved_test_rows_used':0,'development_validation_rows_used':2,'training_ready':False,
        'selection_post_hoc':True,'semantic_accuracy':None,'author_style_accuracy':None,
        'limitations':'Two already-observed failures, eight first attempts; causal prompt-factor diagnostic, not held-out performance. Artifact markers, NLI and style metrics are imperfect proxies. No repairs or sentence quotas.'}


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
        phase('complete',complete=True,completed_jobs=8,completed_pairs=2,completed_verifier_outputs=8)
        write_json(output/'manifest.json',{'protocol':PROTOCOL,'training_called':False,'input_pack_sha256':uploaded_pack_sha,
            'files':{str(p.relative_to(output)):file_sha(p) for p in output.rglob('*') if p.is_file()}})
    except Exception as error:
        failed_at=status.get('phase');locations=[];tb=error.__traceback__
        while tb:
            if tb.tb_frame.f_code.co_filename==__file__:locations.append(tb.tb_lineno)
            tb=tb.tb_next
        phase('failed',complete=False,failed_at=failed_at,error_code='ledger_ablation_inference_failed',local_line_numbers=locations)
        raise RuntimeError('Ledger ablation inference failed; inspect run_status.json and saved partial files. No training was called.') from None


def synthetic_pack():
    prior=parent.synthetic_pack();original={(j['id'],j['condition']):j for j in parent_jobs(prior)}
    rows=[]
    for r in prior['rows'][:2]:
        rows.append({**r,'parent_prompts':{'no_grouping':original[(r['id'],'A_claims_only')]['prompt'],'grouping':original[(r['id'],'B_claims_soft_plan')]['prompt']},
            'previous_rich_output_sha256':{'C_cue_rich':sha(r['source']),'D_cue_rich_grouping':sha(r['source'])}})
    return {'protocol':PROTOCOL,'rows':rows,'style_profile':prior['style_profile']}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);actions=p.add_mutually_exclusive_group()
    actions.add_argument('--prepare-pack',action='store_true');actions.add_argument('--dry-run',action='store_true');actions.add_argument('--execute-inference',action='store_true')
    p.add_argument('--pack',type=Path);p.add_argument('--expected-pack-sha');p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args(argv)
    if a.prepare_pack:prepare_pack(a.output_dir);return
    if a.dry_run:
        pack=synthetic_pack();jobs=jobs_for(pack);rows={r['id']:r for r in pack['rows']};fresh(a.output_dir)
        outputs=[{'id':j['id'],'condition':j['condition'],'prompt_sha256':j['prompt_sha256'],'output':rows[j['id']]['source'],'hit_output_limit':False,'synthetic':True} for j in jobs]
        records,summary=evaluate(pack,jobs,outputs,synthetic=True)
        write_json(a.output_dir/'synthetic_evaluation.json',summary);write_lines(a.output_dir/'synthetic_jobs.jsonl',jobs)
        print('Synthetic wiring check: 8 jobs; no model calls or training.');return
    if a.pack is None or a.expected_pack_sha is None:p.error('--pack and --expected-pack-sha required')
    pack=load_pack(a.pack,a.expected_pack_sha)
    if a.execute_inference:run_training(pack,a.output_dir,a.expected_pack_sha)
    else:
        fresh(a.output_dir);write_json(a.output_dir/'preflight.json',{'protocol':PROTOCOL,'pack_sha256':a.expected_pack_sha,'pairs':2,'jobs':8,'model_calls':0,'training_called':False})
        print('Preflight passed; no Secret access, model download or inference.')


if __name__=='__main__':main()
