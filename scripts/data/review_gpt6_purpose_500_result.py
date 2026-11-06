"""Verify the GPT-6 purpose-only 500 run without further inference or scans."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from fama_style.training import gpt6_purpose_500 as runtime
from fama_style.research.neutral_reconstruction_results import metrics
from fama_style.research.teacher_pair_expanded_pilot_review import checkpoint_header


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def write(p,v):runtime.write_json(p,v)
def write_rows(p,v):p.write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in v))


def verify(download,packet):
    receipt=read(download/'download_manifest.json');raw=download/'raw';folder=raw/runtime.PROTOCOL
    if receipt['slug']!='oliverlds/gpt6-purpose-500-4b' or receipt['version']!=1 or not receipt['version_explicitly_verified_via_SDK']:
        raise ValueError('exact_version_required')
    for name,h in receipt['files'].items():
        if sha(raw/name)!=h:raise ValueError('download_changed')
    splits,c=runtime.load_inputs(packet/'upload/gpt6_purpose_500_pairs.jsonl',packet/'upload/gpt6_purpose_500_config.json')
    report=read(folder/'evaluation.json');state=read(folder/'state.json')
    if report['config']!=c or not report['complete'] or not state['complete'] or not report['training_called'] or not state['training_called']:
        raise ValueError('incomplete_run_or_config_changed')
    if any(x['optimizer_steps']!=250 or x['training_examples_seen']!=500 for x in [report,state]):raise ValueError('training_budget_changed')
    if not report['adapter_weights_changed'] or not report['adapter_reload_weights_match']:raise ValueError('adapter_checks')
    if sha(raw/'gpt6_purpose_500.py')!=c['runtime_source_sha256'] or report['runtime']['runtime_source_sha256']!=c['runtime_source_sha256']:raise ValueError('runtime_changed')
    for k,v in [('model_id',runtime.MODEL_ID),('model_revision',runtime.REVISION),('gpu_count',1)]:
        if report['runtime'][k]!=v:raise ValueError('runtime_contract')
    if 'T4' not in report['runtime']['gpu'] or any(report['runtime']['packages'][k]!=v for k,v in runtime.PACKAGES.items()):raise ValueError('GPU_or_packages_changed')
    updates=read(folder/'training_log.json');grads=read(folder/'gradient_checks.json')
    if len(updates)!=250 or len(grads)!=250:raise ValueError('training_log_coverage')
    order=list(range(500));random.Random(runtime.HYPER['seed']).shuffle(order)
    expected=[c['training_ids'][i] for i in order];actual=[i for x in updates for i in x['example_ids']]
    if actual!=expected or actual!=report['training_order_ids'] or len(set(actual))!=500:raise ValueError('one_epoch_training_order')
    for i,(u,g) in enumerate(zip(updates,grads),1):
        if u['step']!=i or g['step']!=i or not g['finite'] or not math.isfinite(g['norm']) or g['norm']<=0 or not math.isfinite(u['loss']) or u['learning_rate']!=runtime.learning_rate(i):raise ValueError('update_or_gradient')
    labels=read(folder/'label_audit.json')
    if [r['example_id'] for r in labels]!=c['training_ids']+c['development_ids']:raise ValueError('label_coverage')
    if any(not a['target_roundtrip_matches'] or a['truncated'] or not 0<a['prompt_tokens']<a['full_tokens']<=1536 or a['supervised_tokens']!=a['full_tokens']-a['prompt_tokens'] for a in labels):raise ValueError('label_or_length')
    sanity=report['likelihood_sanity']
    if not sanity['passed'] or not math.isclose(sanity['manual_mean_nll'],sanity['builtin_mean_nll'],rel_tol=1e-5,abs_tol=1e-5):raise ValueError('likelihood_sanity')
    for name,h in report['adapter_files_sha256'].items():
        if sha(folder/'epoch_1/adapter'/name)!=h:raise ValueError('adapter_changed')
    header=checkpoint_header(folder/'epoch_1/adapter')
    if header['tensor_count']!=136 or header['parameters']!=2228224 or header['dtypes']!=['F32']:raise ValueError('adapter_layout')
    by={r['example_id']:r for group in splits.values() for r in group};outputs=[]
    if [s['name'] for s in report['snapshots']]!=['base','epoch_1']:raise ValueError('snapshot_names')
    for snap,step in zip(report['snapshots'],[0,250]):
        if not snap['complete'] or snap['optimizer_step']!=step:raise ValueError('snapshot_boundary')
        for field,ids in [('development_losses',c['development_ids']),('training_probe_losses',c['training_probe_ids'])]:
            values=snap[field]
            if len(values)!=len(ids) or {r['example_id'] for r in values}!=set(ids) or any(not math.isfinite(r['loss']) or r['supervised_tokens']<=0 for r in values):raise ValueError('loss_coverage')
            value=sum(r['loss']*r['supervised_tokens'] for r in values)/sum(r['supervised_tokens'] for r in values)
            if not math.isclose(value,snap['development_token_weighted_nll' if field=='development_losses' else 'training_token_weighted_nll'],abs_tol=1e-12):raise ValueError('loss_aggregation')
        if [r['example_id'] for r in snap['generations']]!=c['development_ids']:raise ValueError('generation_coverage')
        for g in snap['generations']:
            original=by[g['example_id']]
            if any(g[k]!=original[k] for k in ['source','paper_id','style_group','content_id','input_condition']) or g['reference']!=original['target'] or g['prompt_sha256']!=runtime.sha(original['prompt']) or g['output_sha256']!=runtime.sha(g['output']):raise ValueError('generation_binding')
            outputs.append(dict(g,snapshot=snap['name'],source_metrics=metrics(g['source'],g['output']),reference_metrics=metrics(g['reference'],g['output'])))
    logs=list(raw.glob('*.log'))
    if len(logs)!=1:raise ValueError('log_coverage')
    log=logs[0].read_text()
    if any(re.search(p,log) for p in [r'hf_[A-Za-z0-9]{20,}',r'KGAT_[A-Za-z0-9_\-]{15,}',r'AKIA[0-9A-Z]{16}']):raise ValueError('credential_pattern_requires_private_remediation')
    return report,state,outputs,header


def aggregate(items):
    result=dict(outputs=len(items),empty=sum(r['empty'] for r in items),output_limited=sum(r['output_limited'] for r in items),
        median_source_overlap=statistics.median(r['source_metrics']['word_sequence_overlap'] for r in items),
        median_reference_overlap=statistics.median(r['reference_metrics']['word_sequence_overlap'] for r in items),
        median_longest_source_run=statistics.median(r['source_metrics']['longest_shared_word_run'] for r in items),
        exact_word_copies=sum(r['source_metrics']['word_token_copy'] for r in items),
        source_overlap_at_least_point8=sum(r['source_metrics']['word_sequence_overlap']>=.8 for r in items))
    if all('assessment' in r for r in items):
        result['material_claim_issues_identified']=sum(r['assessment']['material_claim_change'] for r in items)
        if all('minor_notes' in r['assessment'] for r in items):
            result['outputs_with_minor_shift_notes']=sum(bool(r['assessment']['minor_notes']) for r in items)
        if all('readable' in r['assessment'] for r in items):
            result['readable']=sum(r['assessment']['readable'] for r in items)
    return result



def build(download,packet):
    output=download/'review_v1'
    if output.exists():raise ValueError('refuse_overwrite_review')
    report,state,items,header=verify(download,packet)
    output.mkdir()
    write_rows(output/'outputs_metrics.private.jsonl',items)
    conditions={condition:{name:aggregate([r for r in items if r['snapshot']==name and r['input_condition']==condition])
                for name in ['base','epoch_1']} for condition in ['existing_input','gemini_input']}
    summary=dict(protocol='gpt6_purpose_500_review_v1',slug='oliverlds/gpt6-purpose-500-4b',version=1,
        execution_complete=True,training_pairs=500,optimizer_updates=250,label_checks=524,
        adapter=header,elapsed_seconds=state['elapsed_seconds'],peak_GPU_bytes=state['peak_gpu_allocated_bytes'],
        target_nll={s['name']:s['development_token_weighted_nll'] for s in report['snapshots']},
        target_nll_by_input_condition={s['name']:s['development_nll_by_input_condition'] for s in report['snapshots']},
        conditions=conditions,new_GPU_jobs=0,new_model_calls=0,new_detector_calls=0,
        limits=['Reused development: 24 variants of 16 contents; eight held-out papers, no reserved test.',
        'Overlap and NLL are proxies, not author-specific style, meaning or detector success.',
        'Training data and training prompt changed relative to historical mixed-500.',
        'No automatic metric is an exhaustive claim-preservation check.'])
    prior_path=ROOT/'research_runs/data_scaling_v1/corpus_500_pilot_4b_v1/raw/corpus_500_pilot_v1/evaluation.json'
    if prior_path.exists():
        prior=read(prior_path);old={(s['name'],g['example_id']):g for s in prior['snapshots'] for g in s['generations']}
        matched=all((r['snapshot'],r['example_id']) in old and all(r[k]==old[r['snapshot'],r['example_id']][k]
                    for k in ['source','reference','prompt_sha256']) for r in items)
        summary['historical_mixed_500']=dict(matched_inference_inputs_references_and_prompts=matched,
            base_outputs_byte_identical=sum(r['output_sha256']==old['base',r['example_id']]['output_sha256'] for r in items if r['snapshot']=='base'),
            comparison_limit='Training prompts and data changed together; historical jobs, not a randomized isolated data experiment.')
    write(output/'summary.json',summary)
    write(output/'manifest.json',dict(builder_sha256=sha(Path(__file__)),
        download_manifest_sha256=sha(download/'download_manifest.json'),
        files={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()}))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download',type=Path,default=ROOT/'research_runs/data_scaling_v1/gpt6_purpose_500_4b_v1')
    p.add_argument('--packet',type=Path,default=ROOT/'data/gpt6_purpose_500_v1')
    a=p.parse_args();print(json.dumps(build(a.download,a.packet),indent=2))
