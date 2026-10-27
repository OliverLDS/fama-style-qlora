"""Source-bound sentence grouping: plain, identity-plan and oracle-plan inference."""
import argparse
from collections import Counter
import gc
import importlib.metadata
import json
from pathlib import Path
import re
import statistics
import time

from . import soft_guidance_experiment as parent
from . import name_free_ablation as previous
from .concise_plan import TRAIN_SHA,MODEL_ID,MODEL_REVISION,NLI_MODEL,NLI_REVISION,file_sha,read_lines
from .experiment import fresh,write_json,write_lines,secret_token,with_heartbeat
from fama_style.research import claim_verifier_v3 as verifier
from fama_style.research.claims import sha,sentences
from fama_style.research.signal import features
from fama_style.research.taxonomy_signal import validate_grouping
from fama_style.research.style_drift import score_candidate,frontier

ROOT=Path(__file__).resolve().parents[3]
PROTOCOL='bound_grouping_realization_v1'
SEEDS=(41,42,43)
CONDITIONS=('A_concrete_only','B_identity_groups','C_oracle_groups')
DECODING={'do_sample':True,'temperature':0.6,'top_p':0.9,'top_k':0,'max_new_tokens':768,'context_limit':4096}
POLICY=Path('docs/style_drift_tradeoff_v1/policy.json')
COMMON_INSTRUCTION=previous.COMMON_INSTRUCTION
CONCRETE_STYLE=previous.CONCRETE_STYLE


def output_units(text):
    """Experiment-only boundary check; preserve the historical splitter unchanged."""
    import re
    parts=[];cursor=0;ambiguous=False
    pattern=r'[.!?][)\]”’"\']*\s+(?=[(\[“‘"\']*[A-Z0-9])'
    for match in re.finditer(pattern,text):
        prefix=text[:match.start()+1]
        if re.search(r'(?:\b(?:e\.g|i\.e|U\.S|Dr|Prof|Fig|al))\.$',prefix):
            ambiguous=True;continue
        parts.append(text[cursor:match.end()].strip());cursor=match.end()
    if text[cursor:].strip():parts.append(text[cursor:].strip())
    if parts!=sentences(text):ambiguous=True
    return {'units':parts,'ambiguous':ambiguous,'legacy_count':len(sentences(text))}


def plan_groups(row,condition):
    if condition=='B_identity_groups':return [[u['unit_id']] for u in row['source_units']]
    if condition=='C_oracle_groups':return row['groups']
    return None


def jobs_for(pack):
    rows=pack['rows']
    if len(rows)!=3 or len({r['id'] for r in rows})!=3 or any(r['split']!='train' for r in rows):
        raise ValueError('Expected three unique training diagnostic pairs')
    result=[]
    for index,r in enumerate(rows):
        units=r['source_units'];groups=r['groups']
        if sha(r['source'])!=r['source_sha256'] or [u['unit_id'] for u in units]!=list(range(1,len(units)+1)):
            raise ValueError('Source units changed')
        if ' '.join(' '.join(u['text'].split()) for u in units)!=' '.join(r['source'].split()):raise ValueError('Source units do not reconstruct source')
        flat=[i for g in groups for i in g]
        if any(type(i) is not int for i in flat) or flat!=list(range(1,len(units)+1)) or any(not g for g in groups):raise ValueError('Groups reorder, omit or duplicate source units')
        base=CONCRETE_STYLE+COMMON_INSTRUCTION+'\n\nPassage:\n'+r['source']
        for seed_index,seed in enumerate(SEEDS):
            order=CONDITIONS[(index+seed_index)%3:]+CONDITIONS[:(index+seed_index)%3]
            for condition in order:
                prompt=base;grouping=plan_groups(r,condition)
                if grouping is not None:
                    prompt+=('\n\nFor this grouping diagnostic, use the supplied sentence groups as a presentation request. '
                        'Aim for one output sentence per group, in group order. Reword naturally and retain the common semantic tolerance. '
                        'Grouping grants no new causal, inferential or evidential relationship. '
                        'Return the paragraph only, without unit IDs or a plan.\n'
                        +json.dumps({'source_units':units,'sentence_groups':grouping},ensure_ascii=False,separators=(',',':')))
                result.append({'id':r['id'],'condition':condition,'seed':seed,'prompt':prompt,
                    'prompt_sha256':sha(prompt),'source_sha256':sha(r['source']),'oracle':condition=='C_oracle_groups'})
    return result


def group_diagnostics(row,condition,text,predict=None):
    parsed=output_units(text);groups=plan_groups(row,condition)
    result={'requested_groups':groups,'output_sentence_count':len(parsed['units']),
        'legacy_output_sentence_count':parsed['legacy_count'],'segmentation_ambiguous':parsed['ambiguous'],
        'count_adherence':None,'positional_group_checks':[],'group_binding_verified':False}
    if groups is None:return {**result,'status':'no_plan_requested'}
    if parsed['ambiguous']:return {**result,'status':'sentence_boundaries_unresolved'}
    result['count_adherence']=len(parsed['units'])==len(groups)
    if not result['count_adherence']:return {**result,'status':'count_mismatch_no_positional_alignment'}
    if predict is None:return {**result,'status':'count_matched_semantics_not_run'}
    source={u['unit_id']:u['text'] for u in row['source_units']}
    for i,(group,sentence) in enumerate(zip(groups,parsed['units']),1):
        evidence=' '.join(source[n] for n in group)
        result['positional_group_checks'].append({'output_position':i,'source_unit_ids':group,
            'source_group_sha256':sha(evidence),'output_sentence_sha256':sha(sentence),
            'verifier':verifier.verify(evidence,sentence,predict)})
    return {**result,'status':'positional_checks_recorded_not_semantic_proof'}


def runtime_paths():
    return parent.runtime_paths()+[ROOT/POLICY]


def prepare_pack(output):
    root=ROOT/'research_runs/taxonomy_training_signal_v1/audit_v3'
    release=json.loads((root/'hypothesis_release.json').read_text())
    for name,h in release['files'].items():
        if file_sha(root/name)!=h:raise ValueError('Grouping evidence changed')
    cards=read_lines(root/release['active_grouping_cards'])
    training_path=ROOT/'data/fama_corpus_v1/train.jsonl'
    if file_sha(training_path)!=TRAIN_SHA:raise ValueError('Frozen training data changed')
    training=read_lines(training_path)
    if len(training)!=247 or any(r['split']!='train' for r in training):raise ValueError('Wrong training split')
    training_by_id={r['id']:r for r in training};rows=[]
    for card in cards:
        r=training_by_id[card['training_id']]
        checked=validate_grouping(r['plain_english'],r['completion'],card['groups'],card['source_sha256'],card['target_sha256'])
        if checked['source_units']!=card['source_units']:raise ValueError('Grouping source spans changed')
        if output_units(r['plain_english'])['ambiguous'] or output_units(r['completion'])['ambiguous']:raise ValueError('Resolve diagnostic pair boundaries before preparation')
        rows.append({'id':r['id'],'paper_key':r['paper_key'],'split':'train','source':r['plain_english'],'reference':r['completion'],
            'source_sha256':card['source_sha256'],'reference_sha256':card['target_sha256'],
            'source_units':card['source_units'],'groups':card['groups'],'semantic_obligations_evaluation_only':card['semantic_obligations']})
    previous_root=ROOT/'research_runs/name_free_style_ablation_v1'
    prior_release=json.loads((previous_root/'release_verification.json').read_text())
    prior=previous.load_pack(previous_root/'input_pack/name_free_style_pack.json',prior_release['pack_sha256'])
    pack={'protocol':PROTOCOL,'model_id':MODEL_ID,'model_revision':MODEL_REVISION,'train_sha256':TRAIN_SHA,
        'rows':rows,'style_profile':prior['style_profile'],'policy':prior['policy'],'decoding':DECODING,
        'selection':{'rule':'All three assistant-proposed source-bound grouping hypotheses from the completed training-only audit; not selected on this generation run.',
                     'post_hoc_training_diagnostic':True,'reserved_test_rows_read':0,'validation_rows_read':0},
        'hypothesis_release_sha256':file_sha(root/'hypothesis_release.json'),
        'scope':'Primary C versus B isolates source grouping assignments; A versus B also changes the evidence scaffold. C uses reference-derived groupings and is oracle-only. No full taxonomy mapper is trained.',
        'runtime_code_files':{str(p.relative_to(ROOT)):file_sha(p) for p in runtime_paths()}}
    pack['jobs_sha256']=sha(json.dumps(jobs_for(pack),sort_keys=True,ensure_ascii=False))
    fresh(output);write_json(output/'bound_grouping_pack.json',pack);write_lines(output/'jobs.private.jsonl',jobs_for(pack))
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'status':'prepared_not_executed','pairs':3,'jobs':27,
        'pack_sha256':file_sha(output/'bound_grouping_pack.json'),'training_called':False,'reserved_test_rows_read':0})
    print('Prepared three grouping pairs / 27 inference jobs; no models accessed.');return pack


def load_pack(path,expected_sha):
    if not re.fullmatch('[0-9a-f]{64}',expected_sha or '') or file_sha(path)!=expected_sha:raise ValueError('Pack hash mismatch')
    pack=json.loads(path.read_text())
    if (pack['protocol']!=PROTOCOL or pack['model_id']!=MODEL_ID or pack['model_revision']!=MODEL_REVISION
        or pack['train_sha256']!=TRAIN_SHA
        or pack['decoding']!=DECODING):raise ValueError('Changed experiment identity')
    if pack['policy']!=json.loads((ROOT/POLICY).read_text()):raise ValueError('Changed scoring policy')
    if pack['style_profile']['fit_split']!='train' or pack['style_profile']['training_rows']!=247:raise ValueError('Non-training style profile')
    for row in pack['rows']:
        validate_grouping(row['source'],row['reference'],row['groups'],row['source_sha256'],row['reference_sha256'])
    for name,h in pack['runtime_code_files'].items():
        if file_sha(ROOT/name)!=h:raise ValueError('Frozen runtime changed')
    if pack['jobs_sha256']!=sha(json.dumps(jobs_for(pack),sort_keys=True,ensure_ascii=False)):raise ValueError('Changed jobs')
    return pack


def evaluate(pack,outputs,semantic=None,synthetic=False,group_checks=None):
    jobs=jobs_for(pack);expected={(j['id'],j['condition'],j['seed']):j for j in jobs};seen=set();records=[]
    rows={r['id']:r for r in pack['rows']}
    if len(outputs)!=len(jobs):raise ValueError('Incomplete output set')
    if not synthetic and set(semantic or {})!=set(expected):raise ValueError('Missing semantic evidence')
    if not synthetic and set(group_checks or {})!=set(expected):raise ValueError('Missing grouping diagnostics')
    for out in outputs:
        key=(out['id'],out['condition'],out['seed']);j=expected.get(key)
        if key in seen or j is None or out['prompt_sha256']!=j['prompt_sha256']:raise ValueError('Duplicate or swapped output')
        seen.add(key)
        if not out['output'].strip() or out.get('synthetic') is not synthetic or out['hit_output_limit'] is not False:raise ValueError('Invalid generation')
        row=rows[out['id']];scores=score_candidate(row['source'],out['output'],pack['style_profile'],(semantic or {}).get(key),pack['policy'])
        basic=group_diagnostics(row,out['condition'],out['output'])
        supplied=(group_checks or {}).get(key,basic)
        for field in ('requested_groups','output_sentence_count','legacy_output_sentence_count','segmentation_ambiguous','count_adherence'):
            if supplied[field]!=basic[field]:raise ValueError('Grouping diagnostics do not match output/plan')
        if not synthetic and basic['count_adherence'] is True and len(supplied['positional_group_checks'])!=len(basic['requested_groups']):
            raise ValueError('Missing positional semantic evidence')
        if supplied['positional_group_checks']:
            units={u['unit_id']:u['text'] for u in row['source_units']}
            output_sentences=output_units(out['output'])['units']
            if basic['count_adherence'] is not True:raise ValueError('Positional evidence without resolved alignment')
            for i,(group,check) in enumerate(zip(basic['requested_groups'],supplied['positional_group_checks']),1):
                if (check['output_position']!=i or check['source_unit_ids']!=group
                    or check['source_group_sha256']!=sha(' '.join(units[n] for n in group))
                    or check['output_sentence_sha256']!=sha(output_sentences[i-1])):
                    raise ValueError('Group semantic evidence is not bound to the requested units/output')
        records.append({'id':out['id'],'condition':out['condition'],'seed':out['seed'],'candidate_id':sha(out['id']+'|'+out['output'])[:20],
            'output_sha256':sha(out['output']),'format_valid':parent.paragraph_format(out['output']),
            'output_features':features(out['output']),'verifier':(semantic or {}).get(key),
            'group_diagnostics':supplied,**scores})
    # Do not select winners using references or discard semantic warnings. Show all candidates.
    fronts={r['id']:frontier([{'candidate_id':'source-control','style_alignment_benefit_proxy':0,'semantic_cost_proxy':0}]+
        [m for m in records if m['id']==r['id'] and m['format_valid']]) for r in pack['rows']}
    counts={c:{'outputs':sum(r['condition']==c for r in records),
        'unique_paragraphs':len({r['output_sha256'] for r in records if r['condition']==c}),
        'mean_style_benefit_proxy':statistics.mean(r['style_alignment_benefit_proxy'] for r in records if r['condition']==c),
        'verifier_statuses':dict(Counter(r['old_verifier_status'] for r in records if r['condition']==c)),
        'count_adherence':sum(r['group_diagnostics']['count_adherence'] is True for r in records if r['condition']==c) if c!='A_concrete_only' else None,
        'unresolved_boundaries':sum(r['group_diagnostics']['segmentation_ambiguous'] for r in records if r['condition']==c),
        'positional_verifier_status_counts':dict(Counter(g['verifier']['status'] for r in records if r['condition']==c for g in r['group_diagnostics']['positional_group_checks']))} for c in CONDITIONS}
    return records,{'protocol':PROTOCOL,'status':'synthetic_wiring_check' if synthetic else 'inference_complete',
        'conditions':counts,'within_case_frontiers':fronts,'training_called':False,'reserved_test_rows_used':0,
        'qualitative_style_assessment':'pending','semantic_accuracy':None,'author_style_accuracy':None,
        'primary_comparison':'C_oracle_groups versus B_identity_groups',
        'limitations':'Three selected training examples, 27 sampled outputs. Oracle grouping is not deployment or mapper performance. Count matching alone is not group adherence; positional NLI is not semantic proof. All outputs require qualitative assessment.'}


def audit_tokens(job,ids):
    if sha(job['prompt'])!=job['prompt_sha256'] or not ids or len(ids)+DECODING['max_new_tokens']>DECODING['context_limit']:
        raise ValueError('Changed or over-context prompt; no truncation')
    return {'id':job['id'],'condition':job['condition'],'seed':job['seed'],'input_tokens':len(ids),'input_ids_sha256':sha(json.dumps(ids))}


def run_training(pack,output,pack_sha):
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
        for job,audit in zip(jobs,audits):
            phase('generating_'+str(len(outputs)+1)+'_of_'+str(len(jobs)));torch.manual_seed(job['seed'])
            batch=tokenize(job);ids=batch['input_ids'][0].tolist()
            if audit_tokens(job,ids)!=audit:raise ValueError('Tokenizer changed')
            batch={k:v.to('cuda:0') for k,v in batch.items()};before=time.monotonic()
            with torch.inference_mode():
                generated=with_heartbeat(lambda:model.generate(**batch,**{k:v for k,v in DECODING.items() if k!='context_limit'},
                    use_cache=True,pad_token_id=processor.tokenizer.pad_token_id,remove_invalid_values=False,
                    logits_processor=LogitsProcessorList([FiniteScores()])),'Generation')
            new=generated[0,len(ids):];eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
            text=processor.tokenizer.decode(new,skip_special_tokens=True).strip()
            out={**{k:job[k] for k in ('id','condition','seed','prompt_sha256')},'output':text,'synthetic':False,
                'hit_output_limit':bool(new.numel()>=DECODING['max_new_tokens'] and int(new[-1]) not in eos),
                'input_tokens':len(ids),'output_tokens':int(new.numel()),'generation_seconds':time.monotonic()-before}
            outputs.append(out);write_lines(output/'generations.private.jsonl',outputs)
            if not text or out['hit_output_limit']:raise RuntimeError('Empty or cutoff output; incomplete run')
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
        sources={r['id']:r['source'] for r in pack['rows']};semantic={};verdict_rows=[];group_checks={};group_rows=[]
        for i,out in enumerate(outputs):
            phase('diagnostics_'+str(i+1)+'_of_'+str(len(jobs)))
            verdict=verifier.verify(sources[out['id']],out['output'],predict)
            semantic[(out['id'],out['condition'],out['seed'])]=verdict
            verdict_rows.append({**{k:out[k] for k in ('id','condition','seed')},'source_sha256':sha(sources[out['id']]),'output_sha256':sha(out['output']),'verifier':verdict})
            write_lines(output/'verifier_results.private.jsonl',verdict_rows)
            status['completed_verifications']=len(verdict_rows)
            diagnostic=group_diagnostics(next(r for r in pack['rows'] if r['id']==out['id']),out['condition'],out['output'],predict)
            group_checks[(out['id'],out['condition'],out['seed'])]=diagnostic
            group_rows.append({**{k:out[k] for k in ('id','condition','seed')},'output_sha256':sha(out['output']),'group_diagnostics':diagnostic})
            write_lines(output/'group_verifier_results.private.jsonl',group_rows)
        records,summary=evaluate(pack,outputs,semantic,group_checks=group_checks)
        write_lines(output/'jobs.private.jsonl',jobs);write_lines(output/'metrics.private.jsonl',records);write_json(output/'evaluation.json',summary)
        write_json(output/'runtime.json',{'model_id':MODEL_ID,'revision':MODEL_REVISION,'nli_model':NLI_MODEL,'nli_revision':NLI_REVISION,
            'decoding':DECODING,'elapsed_seconds':time.monotonic()-started,'generation_memory':generation_memory,
            'packages':{k:importlib.metadata.version(k) for k in ('torch','transformers','accelerate','bitsandbytes','huggingface-hub')}})
        status.update(complete=True);phase('complete')
        write_json(output/'manifest.json',{'protocol':PROTOCOL,'pack_sha256':pack_sha,'runtime_code_files':pack['runtime_code_files'],
            'files':{p.name:file_sha(p) for p in output.iterdir() if p.is_file()}})
    except Exception as exc:
        status.update(complete=False,phase='failed',exception_type=type(exc).__name__);write_json(output/'run_status.json',status)
        print('Inference failed; exception details suppressed to protect credentials.',flush=True)
        raise RuntimeError('Inference incomplete; inspect saved status and partial artifacts') from None


def synthetic_pack():
    old=previous.synthetic_pack();rows=[]
    source='Firms invest. Banks lend. Investors save.';target='Firms invest and banks lend. Investors save.'
    for i in range(3):
        checked=validate_grouping(source,target,[[1,2],[3]],sha(source),sha(target))
        rows.append({'id':'synthetic-'+str(i),'split':'train','source':source,'reference':target,
            'source_sha256':sha(source),'reference_sha256':sha(target),'source_units':checked['source_units'],'groups':checked['groups']})
    return {'rows':rows,'style_profile':old['style_profile'],'policy':old['policy']}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);a=p.add_mutually_exclusive_group()
    a.add_argument('--prepare-pack',action='store_true');a.add_argument('--dry-run',action='store_true');a.add_argument('--execute-inference',action='store_true')
    p.add_argument('--pack',type=Path);p.add_argument('--expected-pack-sha');p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args(argv)
    if args.prepare_pack:prepare_pack(args.output_dir);return
    if args.dry_run:
        pack=synthetic_pack();jobs=jobs_for(pack);rows={r['id']:r for r in pack['rows']}
        outputs=[{**{k:j[k] for k in ('id','condition','seed','prompt_sha256')},'output':rows[j['id']]['source'],'synthetic':True,'hit_output_limit':False} for j in jobs]
        _,summary=evaluate(pack,outputs,synthetic=True);fresh(args.output_dir);write_json(args.output_dir/'synthetic_evaluation.json',summary)
        print('Synthetic CPU check: 27 jobs; zero model calls or training.');return
    if args.pack is None:p.error('--pack required')
    pack=load_pack(args.pack,args.expected_pack_sha)
    if args.execute_inference:run_training(pack,args.output_dir,args.expected_pack_sha)
    else:
        fresh(args.output_dir);write_json(args.output_dir/'preflight.json',{'protocol':PROTOCOL,'status':'preflight_passed','jobs':len(jobs_for(pack)),'model_calls':0,'training_called':False})
        print('Preflight passed; no Secret access, model download or generation.')


if __name__=='__main__':main()
