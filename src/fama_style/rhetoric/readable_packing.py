"""Two-case inference-only ablation of offset/JSON versus readable oracle edits."""
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
from . import elementary_realization as parent
from .concise_plan import TRAIN_SHA, MODEL_ID, MODEL_REVISION, NLI_MODEL, NLI_REVISION, file_sha, read_lines
from .experiment import fresh, write_json, write_lines, secret_token, with_heartbeat
from fama_style.evaluation import corpus_baseline as baseline
from fama_style.research import claim_verifier_v3 as verifier
from fama_style.research.claims import sha, risk_flags
from fama_style.research.signal import distance, features, overlap

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = 'readable_packing_ablation_v1'
CONDITIONS = ('A_frozen_offsets_json', 'B_readable_bindings_and_plan')
MODELS = ('base',)
CONFIG = copy.deepcopy(parent.CONFIG)
CASE_IDS = ('Xc33a4080a76e0609', 'Xee3c1c3763b95849')
SUFFIX = ('\nKeep each composite claim intact. Output slots specify sentences, not additional claims. '
          'Preserve every untouched sentence. Resolve pronouns without changing attribution.')


def quoted(text, spans):
    """Expand existing source offsets exactly; no text repair or semantic inference."""
    result = []
    for s in spans:
        if type(s['start']) is not int or type(s['end']) is not int or not 0 <= s['start'] < s['end'] <= len(text):
            raise ValueError('Invalid source evidence bounds')
        quote = text[s['start']:s['end']]
        if quote != s['quote']: raise ValueError('Changed source evidence')
        occurrences = [m.start() for m in re.finditer(re.escape(quote), text)]
        if s['start'] not in occurrences: raise ValueError('Unresolved source occurrence')
        result.append({'quote': quote, 'occurrence': occurrences.index(s['start']) + 1})
    return result


def readable_source(payload):
    compact = parent.compact_source(payload)
    units = payload['source_units']; bounds = compact['source_units']; claims = []
    for c in payload['claim_index']:
        claims.append({'claim_id': c['claim_id'], 'composite': c['composite'], 'epistemic_status': c['epistemic_status'],
                       'source_unit_ids': [u['unit_id'] for u in bounds if any(s['start'] < u['end'] and u['start'] < s['end'] for s in c['source_spans'])],
                       'source_evidence': quoted(payload['source'], c['source_spans']),
                       'protected': [{'kind': p['kind'], 'source_evidence': quoted(payload['source'], p['source_spans'])} for p in c['protected']]})
    return {'source': payload['source'], 'source_units': copy.deepcopy(units), 'claim_index': claims,
            'source_relations': copy.deepcopy(payload['source_relations'])}


def readable_plan(payload):
    known = {c['claim_id'] for c in payload['claim_index']}
    source_ids = {u['unit_id'] for u in payload['source_units']}
    flat = [c for op in payload['edits'] for slot in op['output_slots'] for c in slot['claim_refs']]
    if sorted(flat) != sorted(known): raise ValueError('Plan must include each whole claim exactly once')
    count = sum(len(op['output_slots']) for op in payload['edits'])
    lines = ['Write exactly ' + str(count) + ' complete sentences in one paragraph.']
    index = 1
    for op in payload['edits']:
        if op['op'] not in parent.edits.CATALOG or not op['source_unit_ids'] or not set(op['source_unit_ids']) <= source_ids:
            raise ValueError('Unknown operation or source sentence')
        if not op['output_slots']: raise ValueError('Empty operation')
        lines.append(op['op'] + ' on source sentence(s) ' + ', '.join(op['source_unit_ids']) + ': ' + parent.edits.CATALOG[op['op']])
        for slot in op['output_slots']:
            if not slot['claim_refs']: raise ValueError('Empty output sentence')
            lines.append('Output sentence ' + str(index) + ': express whole claim(s) ' + ', '.join(slot['claim_refs']) + '.')
            index += 1
    return '\n'.join(lines)


def readable_prefix(payload):
    source = readable_source(payload)
    lines = [parent.COMMON, 'Full source:', source['source'], '\nLabeled source sentences:']
    lines.extend(u['unit_id'] + ': ' + u['text'] for u in source['source_units'])
    lines.append('\nSource claim evidence (whole claims; annotations remain fallible):')
    for c in source['claim_index']:
        lines.append(c['claim_id'] + ' [' + c['epistemic_status'] + '; composite=' + str(c['composite']).lower()
                     + '; source ' + ', '.join(c['source_unit_ids']) + ']: '
                     + json.dumps(c['source_evidence'], ensure_ascii=False))
        for p in c['protected']:
            lines.append(c['claim_id'] + ' protected ' + p['kind'] + ': ' + json.dumps(p['source_evidence'], ensure_ascii=False))
    lines.extend(['\nExisting source relationships (recorded certainty unchanged):',
                  json.dumps(source['source_relations'], ensure_ascii=False, sort_keys=True)])
    return '\n'.join(lines)


def frozen_prompt(row):
    return next(j['prompt'] for j in parent.jobs_for({'rows': [row]}) if j['condition'] == parent.CONDITIONS[1])


def jobs_for(pack):
    jobs = []
    for i, row in enumerate(pack['rows']):
        for condition in CONDITIONS if i % 2 == 0 else CONDITIONS[::-1]:
            if condition == CONDITIONS[0]:
                prompt = frozen_prompt(row)
            else:
                prompt = readable_prefix(row['payload'])
                prompt += '\nSentence-packing edit plan:\n' + readable_plan(row['payload']) + SUFFIX + '\nReturn only the rewritten paragraph.'
            jobs.append({'id': row['id'], 'model': 'base', 'condition': condition, 'prompt': prompt,
                         'prompt_sha256': sha(prompt), 'source_sha256': sha(row['source']), 'use': row['use']})
    return jobs


def job_key(row): return parent.job_key(row)
def runtime_paths(): return parent.runtime_paths()
def audit_tokens(job, ids): return parent.audit_tokens(job, ids)
def operation_diagnostics(row, text, predict): return parent.operation_diagnostics(row, text, predict)


def prepare_pack(parent_pack, output):
    old = parent.load_pack(parent_pack, file_sha(parent_pack))
    rows = [copy.deepcopy(next(r for r in old['rows'] if r['id'] == case)) for case in CASE_IDS]
    # Validate exact readable evidence on CPU before freezing jobs.
    for row in rows:
        readable_source(row['payload']); readable_plan(row['payload'])
    pack = {'protocol': PROTOCOL, 'model_id': MODEL_ID, 'model_revision': MODEL_REVISION, 'nli_model': NLI_MODEL,
            'nli_revision': NLI_REVISION, 'train_sha256': TRAIN_SHA, 'config': CONFIG, 'rows': rows,
            'style_profile': copy.deepcopy(old['style_profile']), 'parent_pack_sha256': file_sha(parent_pack),
            'parent_row_hashes': {r['id']: sha(json.dumps(r, ensure_ascii=False, sort_keys=True)) for r in rows},
            'frozen_prompt_hashes': {r['id']: sha(frozen_prompt(r)) for r in rows},
            'selection': {'rule': 'Previously inspected hierarchy merge and market-efficiency split; no reserved probe.',
                          'selected_after_parent_outputs': True, 'validation_rows_read': 0, 'reserved_test_rows_read': 0},
            'runtime_code_files': {str(p.relative_to(ROOT)): file_sha(p) for p in runtime_paths()},
            'scope': 'Retrospective two-case TRAINING oracle diagnostic; joint binding/plan display ablation, not new data or generalization.'}
    pack['jobs_sha256'] = sha(json.dumps(jobs_for(pack), ensure_ascii=False, sort_keys=True))
    fresh(output); write_json(output / 'readable_pack.json', pack); write_lines(output / 'jobs.private.jsonl', jobs_for(pack))
    write_json(output / 'manifest.json', {'protocol': PROTOCOL, 'status': 'prepared_not_executed', 'pack_sha256': file_sha(output / 'readable_pack.json'),
                                        'generation_calls': 4, 'training_called': False, 'model_calls_executed': 0})
    print('Prepared two cases / four inference calls. No upload or execution.')


def load_pack(path, expected_sha):
    if not re.fullmatch('[0-9a-f]{64}', expected_sha or '') or file_sha(path) != expected_sha: raise ValueError('Pack hash mismatch')
    pack = json.loads(path.read_text())
    for k, v in {'protocol': PROTOCOL, 'model_id': MODEL_ID, 'model_revision': MODEL_REVISION, 'nli_model': NLI_MODEL,
                 'nli_revision': NLI_REVISION, 'config': CONFIG, 'train_sha256': TRAIN_SHA}.items():
        if pack[k] != v: raise ValueError('Changed experiment identity')
    if [r['id'] for r in pack['rows']] != list(CASE_IDS): raise ValueError('Changed cases or order')
    for row in pack['rows']:
        if row['split'] != 'train' or row['use'] != 'new_oracle_candidate' or sha(row['source']) != row['source_sha256'] or sha(row['reference']) != row['reference_sha256'] or row['source'] != row['payload']['source']:
            raise ValueError('Changed row identity')
        if pack['parent_row_hashes'][row['id']] != sha(json.dumps(row, ensure_ascii=False, sort_keys=True)):
            raise ValueError('Changed inherited row')
        if sha(frozen_prompt(row)) != pack['frozen_prompt_hashes'][row['id']]: raise ValueError('Historical prompt changed')
        readable_source(row['payload']); readable_plan(row['payload'])
    for name, digest in pack['runtime_code_files'].items():
        if file_sha(ROOT / name) != digest: raise ValueError('Frozen runtime changed')
    if pack['jobs_sha256'] != sha(json.dumps(jobs_for(pack), ensure_ascii=False, sort_keys=True)): raise ValueError('Changed jobs')
    return pack


def evaluate(pack, outputs, evidence=None, synthetic=False):
    expected = {job_key(j): j for j in jobs_for(pack)}; rows = {r['id']: r for r in pack['rows']}; seen = set(); metrics = []
    for out in outputs:
        key = job_key(out)
        if key not in expected or key in seen or out['prompt_sha256'] != expected[key]['prompt_sha256'] or out.get('synthetic') is not synthetic or type(out['hit_output_limit']) is not bool:
            raise ValueError('Unknown, duplicate or changed output')
        seen.add(key); row = rows[out['id']]; available = bool(out['output'].strip()) and not out['hit_output_limit']; v = (evidence or {}).get(key)
        if available and not synthetic and (not v or v['source_sha256'] != row['source_sha256'] or v['output_sha256'] != sha(out['output'])):
            raise ValueError('Missing or unbound diagnostic evidence')
        parsed = parent.prior.grouping.output_units(out['output']) if available else None
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
    for condition in CONDITIONS:
        group = [r for r in metrics if r['condition'] == condition]
        valid = [r for r in group if r['available']]
        groups[condition] = {'attempts': len(group), 'available': len(valid),
                             'sentence_count_matches_all_attempts': sum(r['count_matches'] is True for r in valid),
                             'near_copies': sum(r['source_similarity'] >= .9 for r in valid),
                             'verifier_statuses': dict(Counter(r['verifier']['status'] for r in valid if r['verifier'])),
                             'mean_style_distance_gain_available_only': statistics.mean(r['style_distance_gain_proxy'] for r in valid) if valid else None}
    pairs = []
    for row in pack['rows']:
        a, b = [next(r for r in metrics if r['id'] == row['id'] and r['condition'] == c) for c in CONDITIONS]
        available = a['available'] and b['available']
        pairs.append({'id': row['id'], 'both_available': available, 'sentence_counts_A_B': [a['output_sentence_count'], b['output_sentence_count']],
                      'count_matches_A_B': [a['count_matches'], b['count_matches']],
                      'style_proxy_B_minus_A': b['style_distance_gain_proxy'] - a['style_distance_gain_proxy'] if available else None,
                      'source_overlap_B_minus_A': b['source_similarity'] - a['source_similarity'] if available else None,
                      'verifier_A_B': [r['verifier']['status'] if r['verifier'] else None for r in (a, b)],
                      'acceptable_claim_drift': None})
    return metrics, {'protocol': PROTOCOL, 'status': 'synthetic_wiring_check' if synthetic else 'inference_complete' if all(r['available'] for r in metrics) else 'execution_complete_with_generation_failures',
                     'generation_calls': len(expected), 'groups': groups, 'paired': pairs, 'training_called': False,
                     'validation_rows_read': 0, 'reserved_test_rows_read': 0, 'semantic_accuracy': None, 'author_style_accuracy': None,
                     'scope': 'Two retrospectively selected training examples; oracle plans. Readable source bindings and plan wording change together; their separate effects are not identified.',
                     'decision': 'Inspect actual merge/split realization, unsupported additions, attribution and conditions before weighing style benefits against drift; no binary NLI acceptance gate.'}


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
    p = argparse.ArgumentParser(description=__doc__); mode = p.add_mutually_exclusive_group()
    mode.add_argument('--prepare-pack', action='store_true'); mode.add_argument('--dry-run', action='store_true'); mode.add_argument('--execute-inference', action='store_true')
    p.add_argument('--parent-pack', type=Path); p.add_argument('--pack', type=Path); p.add_argument('--expected-pack-sha')
    p.add_argument('--output-dir', type=Path, required=True); a = p.parse_args(argv)
    if a.prepare_pack:
        if a.parent_pack is None: p.error('--parent-pack required')
        prepare_pack(a.parent_pack, a.output_dir); return
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
        write_json(a.output_dir / 'preflight.json', {'status': 'preflight_passed', 'generation_calls': 4, 'training_called': False, 'token_audit': 'pending_Kaggle_tokenizer'})
    print('CPU check passed; no Secret, weights, model calls or GPU accessed.')


if __name__ == '__main__': main()
