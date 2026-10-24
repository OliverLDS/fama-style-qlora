"""Two-case source-sentence replacement scripts; inference only, exposed cases."""
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

from . import ledger_ablation as parent
from .concise_plan import TRAIN_SHA,MODEL_ID,MODEL_REVISION,NLI_MODEL,NLI_REVISION,file_sha,read_lines
from .experiment import fresh,write_json,write_lines,secret_token,with_heartbeat
from fama_style.research import claim_verifier_v3 as verifier
from fama_style.research.claims import sha,sentences,tokens
from fama_style.research.signal import distance,overlap,features
from fama_style.rhetoric.semantics import checked_prediction

ROOT=Path(__file__).resolve().parents[3]
PROTOCOL='sentence_preserving_edit_v1'
MAX_NEW_TOKENS=768
CONTEXT_LIMIT=4096
VERIFIER_FOLDER=ROOT/'docs/claim_verifier_calibration_v3'
CONDITIONS=('A_baseline','B_edit_script')
VARIANTS=('A_baseline','B_proposed','B_guarded')
audit_tokens=parent.audit_tokens
paragraph_format=parent.paragraph_format
runtime_paths=parent.runtime_paths
artifact_checks=parent.artifact_checks


def source_units(source):return {'c'+str(i):text for i,text in enumerate(sentences(source),1)}


def jobs_for(pack):
    if len(pack['rows'])!=2 or len({r['id'] for r in pack['rows']})!=2 or any(r['split']!='validation' for r in pack['rows']):raise ValueError('Expected two exposed development cases')
    jobs=[]
    for index,row in enumerate(pack['rows']):
        baseline=row['baseline_prompt'];units=source_units(row['source'])
        if not baseline.startswith(parent.make_prompt(row['source'])):raise ValueError('Source prefix mismatch')
        for condition in CONDITIONS if index==0 else CONDITIONS[::-1]:
            prompt=baseline
            if condition=='B_edit_script':
                prompt+='\n\nSource-sentence editing protocol: preserve the source sentence identities and their order. Make only local wording or clause edits that improve expression without adding, dropping or strengthening claims. Do not merge source sentences, insert a conclusion, or move information between sentences. Keep all actors, attributions, quantities, conditions and uncertainty attached to their original sentence. A sentence may remain unchanged when no safe useful edit is available.'
                prompt+='\nOutput-format override: return only a JSON object with the key "edits" and an array of replacement objects, each with exactly "claim_id" and "replacement". Use only these source IDs: '+', '.join(units)+'. Each replacement must be one complete sentence ending in sentence punctuation. Include only sentences you change; all others remain verbatim. Use each ID at most once. No new IDs, deletion, reordering, Markdown or explanations. Return {"edits":[]} if no edits are needed. The program applies replacements at their source positions and joins them as one paragraph.'
            jobs.append({'id':row['id'],'condition':condition,'prompt':prompt,'prompt_sha256':sha(prompt),'source_sha256':sha(row['source']),'oracle':False})
    return jobs


def prepare_pack(output):
    base=ROOT/'research_runs/ledger_grouping_ablation_v1';path=base/'input_pack/ledger_ablation_pack.json'
    release=json.loads((base/'release_verification.json').read_text());prior=parent.load_pack(path,release['pack_sha256'])
    run=base/'kaggle_version_1/ledger_grouping_ablation_v1';review_path=base/'version_1_review/verification.json';review=json.loads(review_path.read_text())
    if not review['checks'] or not all(review['checks'].values()):raise ValueError('Prior review failed')
    for name,digest in review['artifact_hashes'].items():
        if file_sha(run/name)!=digest:raise ValueError('Reviewed ablation changed')
    old_jobs={(j['id'],j['condition']):j for j in parent.jobs_for(prior)};outputs={(r['id'],r['condition']):r for r in read_lines(run/'generations.private.jsonl')}
    rows=[{k:copy.deepcopy(row[k]) for k in ('id','paper_key','split','source','reference')} for row in prior['rows']]
    for row in rows:
        row['baseline_prompt']=old_jobs[(row['id'],'A_sentence_only')]['prompt']
        row['previous_baseline_output_sha256']=sha(outputs[(row['id'],'A_sentence_only')]['output'])
    pack={k:copy.deepcopy(prior[k]) for k in ('parent_training_sha256','validation_sha256','style_profile','execution_gate','verifier_evidence','generation')}
    pack.update(protocol=PROTOCOL,rows=rows,parent_pack_sha256=file_sha(path),selection={'post_hoc':True,'rule':'Same two exposed failure cases, no new selection',
        'review_sha256':file_sha(review_path),'reserved_test_rows_read':0},scope='Structured sentence replacement versus identical clean baseline; proposed and guarded versions are derived from one script, not extra generations.')
    jobs=jobs_for(pack);pack['runtime_code_files']={str(p.relative_to(ROOT)):file_sha(p) for p in runtime_paths()};pack['jobs_sha256']=sha(json.dumps(jobs,sort_keys=True,ensure_ascii=False))
    fresh(output);path=output/'sentence_edit_pack.json';write_json(path,pack);write_lines(output/'jobs.private.jsonl',jobs)
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'status':'prepared_not_executed','pairs':2,'jobs':4,'pack_sha256':file_sha(path),
        'model_calls':0,'training_called':False,'reserved_test_rows_read':0,'files':{p.name:file_sha(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({'status':'prepared_not_executed','pairs':2,'jobs':4,'pack_sha256':file_sha(path)}));return pack


def load_pack(path,expected_sha):
    if not re.fullmatch('[0-9a-f]{64}',expected_sha) or file_sha(path)!=expected_sha:raise ValueError('Input pack hash mismatch')
    pack=json.loads(path.read_text())
    if pack['protocol']!=PROTOCOL or pack['parent_training_sha256']!=TRAIN_SHA or pack['validation_sha256']!=parent.parent.VALIDATION_SHA:raise ValueError('Wrong protocol/data')
    if pack['execution_gate']['ready'] is not True or pack['style_profile']['fit_split']!='train' or pack['style_profile']['training_rows']!=247:raise ValueError('Wrong gate/profile')
    for name,h in pack['runtime_code_files'].items():
        if file_sha(ROOT/name)!=h:raise ValueError('Implementation changed')
    protocol,_=verifier.validate_protocol(VERIFIER_FOLDER);evidence=pack['verifier_evidence']
    if evidence['protocol_sha256']!=file_sha(VERIFIER_FOLDER/'protocol.json') or evidence['code_files']!=protocol['code_files'] or evidence['fixture_files']!=protocol['fixture_files']:raise ValueError('Verifier changed')
    if not verifier.gate(evidence['held_out_metrics'],protocol['readiness_gate'])['passed']:raise ValueError('Verifier gate failed')
    if pack['generation']!={'model_id':MODEL_ID,'revision':MODEL_REVISION,'max_new_tokens':768,'context_limit':4096,'do_sample':False,'seed':42,'truncate':False}:raise ValueError('Generation controls changed')
    if pack['jobs_sha256']!=sha(json.dumps(jobs_for(pack),sort_keys=True,ensure_ascii=False)):raise ValueError('Prompts changed')
    return pack


def apply_edits(source,edits):
    units=source_units(source);replacements={r['claim_id']:r['replacement'] for r in edits}
    return ' '.join(replacements.get(key,text) for key,text in units.items())


def parse_edits(source,raw):
    normalized=raw.strip();fence_removed=False;errors=[];edits=[];units=source_units(source)
    if normalized.startswith('```json\n') and normalized.endswith('\n```'):
        normalized=normalized[len('```json\n'):-len('\n```')];fence_removed=True
    def unique(pairs):
        obj={}
        for k,v in pairs:
            if k in obj:raise ValueError('duplicate_json_key')
            obj[k]=v
        return obj
    try:
        obj=json.loads(normalized,object_pairs_hook=unique)
        if not isinstance(obj,dict) or set(obj)!={'edits'} or not isinstance(obj['edits'],list):raise ValueError('invalid_script_contract')
        seen=set()
        for item in obj['edits']:
            if not isinstance(item,dict) or set(item)!={'claim_id','replacement'}:raise ValueError('invalid_edit_keys')
            identifier,text=item['claim_id'],item['replacement']
            if not isinstance(identifier,str) or identifier not in units or identifier in seen:raise ValueError('unknown_or_duplicate_source_id')
            if not isinstance(text,str) or not text.strip():raise ValueError('empty_replacement')
            text=text.strip()
            if '\n' in text or len(sentences(text))!=1 or re.search(r'[.!?]\s+[a-z]',text) or not re.search(r'[.!?]["”’\']?$',text):raise ValueError('replacement_not_one_surface_sentence')
            seen.add(identifier);edits.append({'claim_id':identifier,'replacement':text})
        paragraph=apply_edits(source,edits)
        if len(sentences(paragraph))!=len(units):raise ValueError('assembled_sentence_count_mismatch')
    except (ValueError,TypeError) as error:
        errors.append(str(error) if isinstance(error,ValueError) else 'invalid_script_types')
    return {'format_valid':not errors,'format_errors':errors,'outer_fence_removed':fence_removed,
        'raw_contract_valid':not errors and not fence_removed,'edits':edits if not errors else [],
        'proposed_paragraph':apply_edits(source,edits) if not errors else None,'source_sentence_count':len(units)}


def assess_edits(source,parsed,predict):
    if not parsed['format_valid']:
        return {**parsed,'guarded_paragraph':None,'per_edit':[],'proposed_nontrivial_edits':0,'accepted_nontrivial_edits':0,'rejected_risk_edits':0,'abstained_edits':0,'noop_edits':0}
    units=source_units(source);accepted=[];checks=[]
    for item in parsed['edits']:
        original=units[item['claim_id']];text=item['replacement'];changed=' '.join(original.split())!=' '.join(text.split())
        if not changed:
            checks.append({**item,'decision':'noop','local_verifier':None,'contextual_support':None});continue
        local=verifier.verify(original,text,predict);context=checked_prediction(predict,source,text)
        if local['status']=='risk_signal':decision='risk_signal'
        elif local['status']=='abstain' or context is None:decision='abstain'
        elif context['entailment']<.8 or context['contradiction']>.1:decision='risk_signal'
        else:decision='accept_proxy';accepted.append(item)
        checks.append({**item,'decision':decision,'local_verifier':local,'contextual_support':context})
    guarded=apply_edits(source,accepted)
    if len(sentences(guarded))!=len(units):raise ValueError('Guarded sentence-count mismatch')
    return {**parsed,'guarded_paragraph':guarded,'per_edit':checks,
        'proposed_nontrivial_edits':sum(r['decision']!='noop' for r in checks),'accepted_nontrivial_edits':len(accepted),
        'rejected_risk_edits':sum(r['decision']=='risk_signal' for r in checks),'abstained_edits':sum(r['decision']=='abstain' for r in checks),
        'noop_edits':sum(r['decision']=='noop' for r in checks)}


def evaluate(pack,jobs,outputs,semantic=None,pair_checks=None,patch_checks=None,canary_passed=False,synthetic=False):
    if jobs!=jobs_for(pack) or len(outputs)!=4:raise ValueError('Changed jobs or incomplete generations')
    expected={(j['id'],j['condition']):j for j in jobs};seen=set();raw={};rows={r['id']:r for r in pack['rows']}
    for out in outputs:
        key=(out['id'],out['condition']);job=expected.get(key)
        if key in seen or job is None or out['prompt_sha256']!=job['prompt_sha256']:raise ValueError('Duplicate/swapped output')
        if not out.get('output','').strip() or out.get('hit_output_limit') is not False or out.get('synthetic') is not synthetic:raise ValueError('Empty/cutoff/mislabeled output')
        seen.add(key);raw[key]=out
    if set(patch_checks or {})!=set(rows):raise ValueError('Missing script assessment')
    records=[];used_semantic=set()
    for row in pack['rows']:
        patch=patch_checks[row['id']];script=raw[(row['id'],'B_edit_script')];parsed=parse_edits(row['source'],script['output'])
        if any(patch[k]!=parsed[k] for k in parsed):raise ValueError('Script parse/assessment mismatch')
        if parsed['format_valid']:
            evidence=[{k:r[k] for k in ('claim_id','replacement')} for r in patch['per_edit']]
            if evidence!=parsed['edits'] or any(r['decision'] not in ('accept_proxy','risk_signal','abstain','noop') for r in patch['per_edit']):raise ValueError('Invalid edit decisions')
            accepted=[{k:r[k] for k in ('claim_id','replacement')} for r in patch['per_edit'] if r['decision']=='accept_proxy']
            if patch['guarded_paragraph']!=apply_edits(row['source'],accepted):raise ValueError('Guarded output does not follow accepted edits')
        for variant in VARIANTS:
            is_baseline=variant=='A_baseline';out=raw[(row['id'],'A_baseline' if is_baseline else 'B_edit_script')]
            text=out['output'] if is_baseline else patch['proposed_paragraph'] if variant=='B_proposed' else patch['guarded_paragraph']
            valid=paragraph_format(text) if is_baseline else patch['format_valid']
            key=(row['id'],variant);verdict=(semantic or {}).get(key) if text is not None else None
            if canary_passed and not synthetic and text is not None and verdict is None:raise ValueError('Missing paragraph verifier result')
            if verdict:used_semantic.add(key)
            status=verdict['status'] if verdict else 'invalid_script' if text is None else 'not_run'
            sim=overlap(row['source'],text) if text is not None else None
            gain=distance(row['source'],pack['style_profile'])-distance(text,pack['style_profile']) if text is not None else None
            usable=canary_passed and not synthetic
            changed=patch['proposed_nontrivial_edits'] if variant=='B_proposed' else patch['accepted_nontrivial_edits'] if variant=='B_guarded' else None
            records.append({'id':row['id'],'variant':variant,'format_valid':valid,'paragraph':text,'verifier_status':status,'verifier':verdict,
                'source_sentence_count':len(source_units(row['source'])),'output_sentence_count':len(sentences(text)) if text is not None else None,
                'source_similarity':sim,'near_copy':sim>=.9 if sim is not None else None,'exact_copy':' '.join(row['source'].split())==' '.join(text.split()) if text is not None else None,
                'token_normalized_copy':tokens(row['source'])==tokens(text) if text is not None else None,
                'style_gain_proxy':gain,'words_per_sentence':features(text)['words_per_sentence'] if text is not None else None,
                'artifact_screen':artifact_checks(row['source'],text) if text is not None else None,
                'nontrivial_edit_count':changed,'valid_semantic_proxy_pass':valid and status=='model_consistent' if usable else None,
                'joint_proxy_success':bool(valid and text is not None and status=='model_consistent' and sim<.9 and gain>0 and (is_baseline or changed>0)) if usable else None,
                'previous_baseline_hash_matches':sha(text)==row.get('previous_baseline_output_sha256') if is_baseline else None})
    if canary_passed and not synthetic and used_semantic!=set(semantic or {}):raise ValueError('Unexpected semantic results')
    aggregate={}
    for variant in VARIANTS:
        group=[r for r in records if r['variant']==variant];available=[r for r in group if r['paragraph'] is not None]
        aggregate[variant]={'rows':2,'available_paragraphs':len(available),'valid_formats':sum(r['format_valid'] for r in group),
            'verifier_status_counts':dict(Counter(r['verifier_status'] for r in group)),
            'valid_semantic_proxy_passes':sum(bool(r['valid_semantic_proxy_pass']) for r in group) if canary_passed and not synthetic else None,
            'joint_proxy_successes':sum(bool(r['joint_proxy_success']) for r in group) if canary_passed and not synthetic else None,
            'near_copies_available_only':sum(bool(r['near_copy']) for r in available),'exact_copies_available_only':sum(bool(r['exact_copy']) for r in available),
            'token_normalized_copies_available_only':sum(bool(r['token_normalized_copy']) for r in available),
            'mean_style_gain_proxy_available_only':statistics.mean(r['style_gain_proxy'] for r in available) if available else None,
            'mean_words_per_sentence_available_only':statistics.mean(r['words_per_sentence'] for r in available) if available else None,
            'source_sentence_count_matches':sum(r['output_sentence_count']==r['source_sentence_count'] for r in available),
            'baseline_hash_matches':sum(r['previous_baseline_hash_matches'] is True for r in group) if variant=='A_baseline' else None}
    return records,{'protocol':PROTOCOL,'execution_kind':'synthetic_wiring_check' if synthetic else 'model_inference','variants':aggregate,
        'generated_jobs':4,'evaluated_variants':3,'script_contracts':{r['id']:{k:v for k,v in patch_checks[r['id']].items() if k not in ('edits','per_edit','proposed_paragraph','guarded_paragraph')} for r in pack['rows']},
        'generation_cost':{c:{'jobs':2,**{k:sum(out.get(k,0) or 0 for out in outputs if out['condition']==c) for k in ('input_tokens','output_tokens','generation_seconds')}} for c in CONDITIONS},
        'runtime_canary_passed':canary_passed,'pair_source_reference_verifier':pair_checks,'training_called':False,'reserved_test_rows_used':0,'development_validation_rows_used':2,
        'training_ready':False,'semantic_accuracy':None,'selection_post_hoc':True,
        'limitations':'Two exposed cases; four generations. Proposed and guarded text share one script per case and are not independent generations. Rejected/unknown edits retain source text; copying is a material tradeoff. Sentence structure is constrained by application, not evidence of learned style. Unit consistency does not certify paragraph meaning.'}


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
        semantic={};pair_checks={};patch_checks={};sources={r['id']:r['source'] for r in pack['rows']}
        rows={r['id']:r for r in pack['rows']}
        for index,row in enumerate(pack['rows'],1):
            phase('pair_fidelity_assessment',completed_pairs=index-1)
            pair_checks[row['id']]=with_heartbeat(lambda:verifier.verify(row['source'],row['reference'],predict),'Pair verifier')
        for index,result in enumerate(outputs,1):
            phase('output_semantic_assessment',completed_verifier_outputs=index-1)
            row=rows[result['id']]
            if result['condition']=='A_baseline':
                semantic[(row['id'],'A_baseline')]=with_heartbeat(lambda:verifier.verify(row['source'],result['output'],predict),'Baseline verifier')
            else:
                parsed=parse_edits(row['source'],result['output'])
                patch=with_heartbeat(lambda:assess_edits(row['source'],parsed,predict),'Edit verifier')
                patch_checks[row['id']]=patch
                if parsed['format_valid']:
                    for variant,text in (('B_proposed',patch['proposed_paragraph']),('B_guarded',patch['guarded_paragraph'])):
                        semantic[(row['id'],variant)]=with_heartbeat(lambda:verifier.verify(row['source'],text,predict),'Paragraph verifier')
        records,summary=evaluate(pack,jobs,outputs,semantic,pair_checks,patch_checks,canary_passed=True)
        write_lines(output/'metrics.private.jsonl',records)
        write_json(output/'edit_assessments.private.json',patch_checks)
        write_lines(output/'nli_probability_cache.private.jsonl',[{'premise_sha256':a,'hypothesis_sha256':b,'probabilities':p} for (a,b),p in cache.items()])
        summary.update(model_id=MODEL_ID,model_revision=MODEL_REVISION,nli_model=NLI_MODEL,nli_revision=NLI_REVISION,
            canary_seconds=canary_seconds,verifier_seconds=time.monotonic()-nli_started,total_seconds=time.monotonic()-started,
            nli_cache_entries=len(cache),nli_cache_hits=cache_hits,gemma_memory=gemma_memory,
            nli_memory={'peak_allocated_bytes':torch.cuda.max_memory_allocated(0),'peak_reserved_bytes':torch.cuda.max_memory_reserved(0)},
            generation=pack['generation'],versions={n:importlib.metadata.version(n) for n in ('torch','transformers','accelerate','bitsandbytes','huggingface-hub')})
        write_json(output/'evaluation.json',summary)
        phase('complete',complete=True,completed_jobs=4,completed_pairs=2,completed_verifier_outputs=4)
        write_json(output/'manifest.json',{'protocol':PROTOCOL,'training_called':False,'input_pack_sha256':uploaded_pack_sha,
            'files':{str(p.relative_to(output)):file_sha(p) for p in output.rglob('*') if p.is_file()}})
    except Exception as error:
        failed_at=status.get('phase');locations=[];tb=error.__traceback__
        while tb:
            if tb.tb_frame.f_code.co_filename==__file__:locations.append(tb.tb_lineno)
            tb=tb.tb_next
        phase('failed',complete=False,failed_at=failed_at,error_code='sentence_edit_inference_failed',local_line_numbers=locations)
        raise RuntimeError('Sentence-edit inference failed; inspect run_status.json and saved partial files. No training was called.') from None


def synthetic_pack():
    prior=parent.synthetic_pack();old={(j['id'],j['condition']):j for j in parent.jobs_for(prior)}
    rows=[{k:copy.deepcopy(r[k]) for k in ('id','paper_key','split','source','reference')} for r in prior['rows']]
    for r in rows:
        r['baseline_prompt']=old[(r['id'],'A_sentence_only')]['prompt'];r['previous_baseline_output_sha256']=sha(r['source'])
    return {'protocol':PROTOCOL,'rows':rows,'style_profile':prior['style_profile']}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);actions=p.add_mutually_exclusive_group()
    actions.add_argument('--prepare-pack',action='store_true');actions.add_argument('--dry-run',action='store_true');actions.add_argument('--execute-inference',action='store_true')
    p.add_argument('--pack',type=Path);p.add_argument('--expected-pack-sha');p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args(argv)
    if a.prepare_pack:prepare_pack(a.output_dir);return
    if a.dry_run:
        pack=synthetic_pack();jobs=jobs_for(pack);rows={r['id']:r for r in pack['rows']};fresh(a.output_dir)
        outputs=[{'id':j['id'],'condition':j['condition'],'prompt_sha256':j['prompt_sha256'],'output':rows[j['id']]['source'] if j['condition']=='A_baseline' else '{"edits":[]}', 'hit_output_limit':False,'synthetic':True} for j in jobs]
        patches={r['id']:assess_edits(r['source'],parse_edits(r['source'],'{"edits":[]}'),lambda a,b:None) for r in pack['rows']}
        records,summary=evaluate(pack,jobs,outputs,patch_checks=patches,synthetic=True)
        write_json(a.output_dir/'synthetic_evaluation.json',summary);write_lines(a.output_dir/'synthetic_jobs.jsonl',jobs)
        print('Synthetic wiring check: 4 jobs; no model calls or training.');return
    if a.pack is None or a.expected_pack_sha is None:p.error('--pack and --expected-pack-sha required')
    pack=load_pack(a.pack,a.expected_pack_sha)
    if a.execute_inference:run_training(pack,a.output_dir,a.expected_pack_sha)
    else:
        fresh(a.output_dir);write_json(a.output_dir/'preflight.json',{'protocol':PROTOCOL,'pack_sha256':a.expected_pack_sha,'pairs':2,'jobs':4,'model_calls':0,'training_called':False})
        print('Preflight passed; no Secret access, model download or inference.')


if __name__=='__main__':main()
