"""Six-paper, base-only oracle packing diagnostic. No optimizer or training."""
import argparse
from collections import Counter
import gc
import importlib.metadata
import json
from pathlib import Path
import re
import statistics
import time
from . import elementary_edits as edits
from . import operation_realization as prior
from .concise_plan import TRAIN_SHA, MODEL_ID, MODEL_REVISION, NLI_MODEL, NLI_REVISION, file_sha, read_lines
from .experiment import fresh, write_json, write_lines, secret_token, with_heartbeat
from fama_style.evaluation import corpus_baseline as baseline
from fama_style.research import claim_verifier_v3 as verifier
from fama_style.research.claims import sha, risk_flags
from fama_style.research.signal import fit_profile, distance, features, overlap

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = 'elementary_packing_realization_v1'
CONDITIONS = ('A_ordinary_shared_evidence', 'B_elementary_edits_shared_evidence')
MODELS = ('base',)
CONFIG = {'do_sample': False, 'max_new_tokens': 768, 'context_limit': 4096, 'seed': 42}
COMMON = (prior.previous.grouping.CONCRETE_STYLE + '\nRewrite as one formal economics paragraph, returning only the paragraph. '
          'Keep the claims, actors, attribution, quantities, conditions, negation, uncertainty and scope. '
          'Change presentation and expression where useful. Do not introduce unsupported causal or inferential connections. '
          'Source annotations are fallible; the full source is authoritative. Interpretive relations are not established facts.\n')


def compact_source(payload):
    """Identical source-only scaffold; offsets avoid repeating protected quotations."""
    def bounds(spans):
        return [{'start': s['start'], 'end': s['end']} for s in spans]
    units = []; cursor = 0
    for unit in payload['source_units']:
        start = payload['source'].find(unit['text'], cursor)
        if start < 0: raise ValueError('Unbound source sentence')
        cursor = start + len(unit['text'])
        units.append({'unit_id': unit['unit_id'], 'start': start, 'end': cursor})
    claims = [{'claim_id': c['claim_id'], 'composite': c['composite'], 'epistemic_status': c['epistemic_status'],
               'source_spans': bounds(c['source_spans']),
               'protected': [{'kind': p['kind'], 'source_spans': bounds(p['source_spans'])} for p in c['protected']]}
              for c in payload['claim_index']]
    return {'source': payload['source'], 'source_units': units, 'claim_index': claims,
            'source_relations': payload['source_relations'], 'offset_convention': 'Python character offsets, end exclusive'}


def source_prefix(payload):
    return COMMON + json.dumps(compact_source(payload), ensure_ascii=False, sort_keys=True)


def jobs_for(pack):
    result = []
    for i, row in enumerate(pack['rows']):
        prefix = source_prefix(row['payload'])
        for condition in CONDITIONS if i % 2 == 0 else CONDITIONS[::-1]:
            prompt = prefix
            if condition == CONDITIONS[1]:
                prompt += '\nSentence-packing edit plan:\n' + json.dumps({'operations': edits.CATALOG, 'edits': row['payload']['edits']}, sort_keys=True)
                prompt += '\nKeep each composite claim intact. Output slots specify sentences, not additional claims. Preserve every untouched sentence. Resolve pronouns without changing attribution.'
            prompt += '\nReturn only the rewritten paragraph.'
            result.append({'id': row['id'], 'model': 'base', 'condition': condition, 'prompt': prompt,
                           'prompt_sha256': sha(prompt), 'source_sha256': sha(row['source']), 'use': row['use']})
    return result


def job_key(row): return (row['model'], row['id'], row['condition'])
def runtime_paths(): return prior.runtime_paths()


def select_six(candidates):
    split = [r for r in candidates if any(o['op'] == 'SPLIT' for o in r['generation_payload']['edits']) and not r['diagnostic_probe']]
    probes = [r for r in candidates if r['diagnostic_probe']]
    if len(split) != 1 or len(probes) != 1: raise ValueError('Expected one split and one reserved probe')
    chosen = split + probes
    if len({r['paper_key'] for r in chosen}) != 2: raise ValueError('Forced cases must be paper-distinct')
    for paper in sorted({r['paper_key'] for r in candidates} - {r['paper_key'] for r in chosen}):
        pool = [r for r in candidates if r['paper_key'] == paper and not r['diagnostic_probe']]
        chosen.append(min(pool, key=lambda r: (len(r['prompt']), r['case_id'])))
    if len(chosen) != 6: raise ValueError('Expected six distinct papers')
    return sorted(chosen, key=lambda r: r['case_id'])


def prepare_pack(compiled, package, output):
    key = edits.base.check_package(package)
    m = json.loads((compiled / 'manifest.json').read_text())
    if m['code_sha256'] != file_sha(Path(edits.__file__)): raise ValueError('Changed elementary compiler')
    for name, digest in m['files'].items():
        if file_sha(compiled / name) != digest: raise ValueError('Changed elementary artifacts')
    selected = select_six(read_lines(compiled / 'oracle_candidates.private.jsonl'))
    infos = {r['case_id']: r for r in key['records']}
    train = read_lines(ROOT / 'data/fama_corpus_v1/train.jsonl')
    rows = []
    for r in selected:
        info = infos[r['case_id']]; record = info['record']
        if info['split'] != 'train' or r['pair_sha256'] != record['pair_sha256'] or r['generation_payload']['source'] != record['source']:
            raise ValueError('Changed candidate identity')
        rows.append({'id': r['case_id'], 'source_id': info['source_id'], 'paper_key': info['paper_key'], 'split': 'train',
                     'use': 'reserved_diagnostic_probe' if r['diagnostic_probe'] else 'new_oracle_candidate',
                     'source': record['source'], 'reference': record['reference'], 'source_sha256': sha(record['source']),
                     'reference_sha256': sha(record['reference']), 'payload': r['generation_payload'],
                     'annotation_sha256': r['annotation_sha256']})
    pack = {'protocol': PROTOCOL, 'model_id': MODEL_ID, 'model_revision': MODEL_REVISION, 'train_sha256': TRAIN_SHA,
            'nli_model': NLI_MODEL, 'nli_revision': NLI_REVISION, 'config': CONFIG, 'rows': rows,
            'style_profile': fit_profile(train), 'runtime_code_files': {str(p.relative_to(ROOT)): file_sha(p) for p in runtime_paths()},
            'selection': {'rule': 'One split and the reserved probe forced; one shortest compiled prompt per remaining paper, case ID ties.',
                          'model_outputs_used': False, 'compiled_manifest_sha256': file_sha(compiled / 'manifest.json'),
                          'package_manifest_sha256': file_sha(package / 'manifest.json')},
            'scope': 'Six previously annotated TRAINING paragraphs. Oracle reference-derived grouping; no learned mapping, adapter or generalization claim.'}
    pack['jobs_sha256'] = sha(json.dumps(jobs_for(pack), sort_keys=True, ensure_ascii=False))
    fresh(output); write_json(output / 'elementary_pack.json', pack); write_lines(output / 'jobs.private.jsonl', jobs_for(pack))
    write_json(output / 'manifest.json', {'protocol': PROTOCOL, 'status': 'prepared_not_executed', 'pack_sha256': file_sha(output / 'elementary_pack.json'),
                                        'generation_calls': 12, 'training_called': False, 'validation_rows_read': 0, 'reserved_test_rows_read': 0})
    print('Prepared six papers / 12 base-model inference calls; no upload or execution.')


def load_pack(path, expected_sha):
    if not re.fullmatch('[0-9a-f]{64}', expected_sha or '') or file_sha(path) != expected_sha: raise ValueError('Pack hash mismatch')
    p = json.loads(path.read_text())
    for key, value in {'protocol': PROTOCOL, 'model_id': MODEL_ID, 'model_revision': MODEL_REVISION, 'train_sha256': TRAIN_SHA,
                       'nli_model': NLI_MODEL, 'nli_revision': NLI_REVISION, 'config': CONFIG}.items():
        if p[key] != value: raise ValueError('Changed experiment identity')
    rows = p['rows']
    if len(rows) != 6 or len({r['id'] for r in rows}) != 6 or len({r['paper_key'] for r in rows}) != 6: raise ValueError('Wrong selection')
    if Counter(r['use'] for r in rows) != Counter({'new_oracle_candidate': 5, 'reserved_diagnostic_probe': 1}): raise ValueError('Wrong probe assignment')
    for r in rows:
        if r['split'] != 'train' or r['source_sha256'] != sha(r['source']) or r['reference_sha256'] != sha(r['reference']) or r['payload']['source'] != r['source']:
            raise ValueError('Changed text or split')
        # Recheck the compiled payload's entire claim partition, including KEEP slots.
        ids = [c['claim_id'] for c in r['payload']['claim_index']]
        flat = [c for op in r['payload']['edits'] for slot in op['output_slots'] for c in slot['claim_refs']]
        if len(ids) != len(set(ids)) or sorted(flat) != sorted(ids) or any(op['op'] not in edits.CATALOG for op in r['payload']['edits']):
            raise ValueError('Changed edit coverage')
    for name, digest in p['runtime_code_files'].items():
        if file_sha(ROOT / name) != digest: raise ValueError('Frozen runtime changed')
    if p['jobs_sha256'] != sha(json.dumps(jobs_for(p), sort_keys=True, ensure_ascii=False)): raise ValueError('Changed jobs')
    return p


def audit_tokens(job, ids):
    if sha(job['prompt']) != job['prompt_sha256'] or not ids or len(ids) + CONFIG['max_new_tokens'] > CONFIG['context_limit']:
        raise ValueError('Changed or over-context prompt; no truncation')
    return {'id': job['id'], 'model': job['model'], 'condition': job['condition'], 'input_tokens': len(ids), 'input_ids_sha256': sha(json.dumps(ids))}


def operation_diagnostics(row, text, predict):
    # Reuse frozen NLI localization, with elementary slots taking the place of stages.
    slots = [slot for op in row['payload']['edits'] for slot in op['output_slots']]
    shim = {'payload': {'source_claim_index': row['payload']['claim_index'],
                        'program': {'stages': [{'role': 'sentence_' + str(i + 1), 'source_claim_refs': slot['claim_refs']} for i, slot in enumerate(slots)]}}}
    result = prior.operation_diagnostics(shim, text, predict)
    if result['status'] != 'localization_hypotheses_only':
        return {**result, 'slot_matches_proxy': None, 'slot_order_and_grouping_proxy': None}
    positions = {c['claim_id']: c['localized_position'] for c in result['claim_sentence_scores']}
    hits = [all(positions[c] == i + 1 for c in s['claim_refs']) if all(positions[c] is not None for c in s['claim_refs']) else None for i, s in enumerate(slots)]
    return {**result, 'slot_matches_proxy': hits,
            'slot_order_and_grouping_proxy': all(hits) if all(h is not None for h in hits) else None,
            'localization_only_not_semantic_or_style_proof': True}


def evaluate(pack, outputs, evidence=None, synthetic=False):
    expected = {job_key(j): j for j in jobs_for(pack)}; rows = {r['id']: r for r in pack['rows']}; seen = set(); metrics = []
    for out in outputs:
        key = job_key(out)
        if key not in expected or key in seen or out['prompt_sha256'] != expected[key]['prompt_sha256'] or out.get('synthetic') is not synthetic or type(out['hit_output_limit']) is not bool:
            raise ValueError('Unknown, duplicate or changed output')
        seen.add(key); row = rows[out['id']]; available = bool(out['output'].strip()) and not out['hit_output_limit']; v = (evidence or {}).get(key)
        if available and not synthetic and (not v or v['source_sha256'] != row['source_sha256'] or v['output_sha256'] != sha(out['output'])):
            raise ValueError('Missing or unbound diagnostic evidence')
        parsed = prior.grouping.output_units(out['output']) if available else None
        desired = sum(len(op['output_slots']) for op in row['payload']['edits'])
        metrics.append({**out, 'use': row['use'], 'paper_key': row['paper_key'], 'available': available,
                        'source_sha256': row['source_sha256'], 'output_sha256': sha(out['output']),
                        'source_similarity': overlap(row['source'], out['output']) if available else None,
                        'reference_similarity_evaluation_only': overlap(row['reference'], out['output']) if available else None,
                        'style_distance_gain_proxy': distance(row['source'], pack['style_profile']) - distance(out['output'], pack['style_profile']) if available else None,
                        'features': features(out['output']) if available else None,
                        'surface_flags': risk_flags(row['source'], out['output']) if available else None,
                        'desired_sentence_count': desired, 'output_sentence_count': len(parsed['units']) if parsed and not parsed['ambiguous'] else None,
                        'count_matches': len(parsed['units']) == desired if parsed and not parsed['ambiguous'] else None,
                        'verifier': v['verifier'] if v and available else None,
                        'operation_diagnostics': v['operation_diagnostics'] if v and available else None,
                        'claim_drift_severity': 'unassessed', 'acceptable_claim_drift': None, 'author_style_accuracy': None})
    if seen != set(expected): raise ValueError('Incomplete output matrix')
    groups = {}
    for use in ('new_oracle_candidate', 'reserved_diagnostic_probe'):
        groups[use] = {}
        for condition in CONDITIONS:
            group = [r for r in metrics if r['use'] == use and r['condition'] == condition]; valid = [r for r in group if r['available']]
            groups[use][condition] = {'attempts': len(group), 'available': len(valid), 'count_matches_all_attempts': sum(r['count_matches'] is True for r in valid),
                                    'near_copies_available': sum(r['source_similarity'] >= .9 for r in valid),
                                    'mean_style_distance_gain_available_only': statistics.mean(r['style_distance_gain_proxy'] for r in valid) if valid else None,
                                    'verifier_statuses': dict(Counter(r['verifier']['status'] for r in valid if r['verifier'])),
                                    'generation_seconds': sum(r.get('generation_seconds', 0) for r in group)}
    paired = []
    for row in pack['rows']:
        a, b = [next(r for r in metrics if r['id'] == row['id'] and r['condition'] == c) for c in CONDITIONS]
        both = a['available'] and b['available']
        paired.append({'id': row['id'], 'use': row['use'], 'both_available': both,
                       'style_proxy_gain_B_minus_A': b['style_distance_gain_proxy'] - a['style_distance_gain_proxy'] if both else None,
                       'source_similarity_B_minus_A': b['source_similarity'] - a['source_similarity'] if both else None,
                       'counts_A_B': [a['count_matches'], b['count_matches']],
                       'verifier_A_B': [r['verifier']['status'] if r['verifier'] else None for r in (a, b)],
                       'claim_drift_acceptability': 'not_inferred_from_binary_verifier_or_surface_style'})
    return metrics, {'protocol': PROTOCOL, 'status': 'synthetic_wiring_check' if synthetic else 'inference_complete' if all(r['available'] for r in metrics) else 'execution_complete_with_generation_failures',
                     'generation_calls': len(expected), 'groups': groups, 'paired': paired, 'training_called': False,
                     'validation_rows_read': 0, 'reserved_test_rows_read': 0, 'semantic_accuracy': None, 'author_style_accuracy': None,
                     'decision': 'Assess organization, style and drift separately from saved evidence. No automatic acceptance gate; minor drift may be worthwhile, but NLI does not establish its severity.',
                     'scope': 'Training-only, reference-grouping oracle; base model only. Neither learned mapping nor held-out generalization.'}


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
        for model_name in MODELS:
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
                'verifier':with_heartbeat(lambda:verifier.verify(row['source'],candidate['output'],predict),'Semantic diagnostics'),
                'operation_diagnostics':with_heartbeat(lambda:operation_diagnostics(row,candidate['output'],predict),'Claim localization')}
            evidence[job_key(candidate)]=diagnostic;verdict_rows.append(diagnostic)
            write_lines(output/'verifier_results.private.jsonl',verdict_rows)
        records,summary=evaluate(pack,outputs,evidence)
        write_lines(output/'jobs.private.jsonl',jobs);write_lines(output/'metrics.private.jsonl',records);write_json(output/'evaluation.json',summary)
        write_json(output/'runtime.json',{'model_id':MODEL_ID,'revision':MODEL_REVISION,'nli_model':NLI_MODEL,'nli_revision':NLI_REVISION,
            'decoding':CONFIG,'elapsed_seconds':time.monotonic()-started,'generation_memory':generation_memory,
            'packages':{k:importlib.metadata.version(k) for k in ('torch','transformers','accelerate','bitsandbytes','huggingface-hub')}})
        status.update(complete=True,all_candidates_complete=all(r['available'] for r in records));phase(summary['status'])
        write_json(output/'manifest.json',{'protocol':PROTOCOL,'pack_sha256':pack_sha,'runtime_code_files':pack['runtime_code_files'],
            'files':{p.name:file_sha(p) for p in output.iterdir() if p.is_file()}})
    except Exception as exc:
        status.update(complete=False,phase='failed',completed_jobs=len(outputs),exception_type=type(exc).__name__);write_json(output/'run_status.json',status)
        print('Inference failed; exception details suppressed to protect credentials.',flush=True)
        raise RuntimeError('Inference incomplete; inspect saved status and partial artifacts') from None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--prepare-pack', action='store_true'); mode.add_argument('--dry-run', action='store_true'); mode.add_argument('--execute-inference', action='store_true')
    p.add_argument('--compiled-dir', type=Path); p.add_argument('--package', type=Path)
    p.add_argument('--pack', type=Path); p.add_argument('--expected-pack-sha'); p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args(argv)
    if a.prepare_pack:
        if a.compiled_dir is None or a.package is None: p.error('--compiled-dir and --package required')
        prepare_pack(a.compiled_dir, a.package, a.output_dir); return
    if a.pack is None: p.error('--pack required')
    pack = load_pack(a.pack, a.expected_pack_sha)
    if a.execute_inference:
        run_training(pack, a.output_dir, a.expected_pack_sha); return
    fresh(a.output_dir)
    if a.dry_run:
        outputs = [{**{k: j[k] for k in ('id', 'model', 'condition', 'prompt_sha256')}, 'output': 'Synthetic placeholder.',
                    'synthetic': True, 'hit_output_limit': False} for j in jobs_for(pack)]
        _, summary = evaluate(pack, outputs, synthetic=True)
        write_json(a.output_dir / 'synthetic_evaluation.json', summary)
    else:
        write_json(a.output_dir / 'preflight.json', {'status': 'preflight_passed', 'generation_calls': 12, 'training_called': False, 'token_audit': 'pending_Kaggle_tokenizer'})
    print('CPU check passed; no Secret, weights, model calls or GPU accessed.')


if __name__ == '__main__':
    main()
