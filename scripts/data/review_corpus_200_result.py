"""Verify a completed 200-pair pilot and aggregate text-bound offline judgments.

Reads private downloaded artifacts; never loads weights or invokes a model.
"""
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
from fama_style.training import corpus_200_pilot as runtime
from fama_style.research.neutral_reconstruction_results import metrics
from fama_style.research.teacher_pair_expanded_pilot_review import checkpoint_header


def file_sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def rows(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
def write_rows(path,items):path.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in items))


def build(download,packet,output):
    if output.exists():raise ValueError('refuse_overwrite_result_review')
    raw=download/'raw';folder=raw/'corpus_200_pilot_v1'
    receipt=json.loads((download/'download_manifest.json').read_text())
    if receipt['slug']!='oliverlds/corpus-200-pilot-4b' or receipt['version']!=1:raise ValueError('wrong_run_version')
    for name,h in receipt['files'].items():
        if file_sha(raw/name)!=h:raise ValueError('download_changed')
    splits,config=runtime.load_inputs(packet/'upload/corpus_200_pairs.jsonl',packet/'upload/corpus_200_config.json')
    report=json.loads((folder/'evaluation.json').read_text());state=json.loads((folder/'state.json').read_text())
    if (report['config']!=config or not report['complete'] or not state['complete']
            or not report['training_called'] or not state['training_called']
            or report['optimizer_steps']!=100 or state['optimizer_steps']!=100
            or report['training_examples_seen']!=200 or state['training_examples_seen']!=200
            or not report['adapter_weights_changed'] or not report['adapter_reload_weights_match']):
        raise ValueError('incomplete_training_or_adapter_checks')
    if report['runtime']['runtime_source_sha256']!=file_sha(Path(runtime.__file__)) or file_sha(raw/'corpus_200_pilot.py')!=config['runtime_source_sha256']:
        raise ValueError('runtime_changed')
    if any(report['runtime'][k]!=v for k,v in [('model_id',runtime.MODEL_ID),('model_revision',runtime.REVISION),('gpu_count',1)]):
        raise ValueError('model_revision_or_device_changed')
    if any(report['runtime']['packages'][k]!=v for k,v in runtime.PACKAGES.items()):raise ValueError('packages_changed')
    updates=json.loads((folder/'training_log.json').read_text());gradients=json.loads((folder/'gradient_checks.json').read_text())
    if len(updates)!=100 or len(gradients)!=100:raise ValueError('hundred_updates_and_gradients_required')
    order=list(range(200));random.Random(runtime.HYPER['seed']).shuffle(order)
    expected=[config['training_ids'][i] for i in order]
    actual=[rid for update in updates for rid in update['example_ids']]
    if expected!=actual or actual!=report['training_order_ids'] or len(set(actual))!=200:raise ValueError('training_exposure_or_order_changed')
    for i,(update,gradient) in enumerate(zip(updates,gradients),1):
        if (update['step']!=i or gradient['step']!=i or not gradient['finite'] or not math.isfinite(gradient['norm'])
                or gradient['norm']<=0 or not math.isfinite(update['loss']) or update['learning_rate']!=runtime.learning_rate(i)):
            raise ValueError('update_or_gradient_contract_failed')
    labels=json.loads((folder/'label_audit.json').read_text());by_id={r['example_id']:r for group in splits.values() for r in group}
    if [r['example_id'] for r in labels]!=config['training_ids']+config['development_ids']:raise ValueError('label_coverage')
    if any(not a['target_roundtrip_matches'] or a['truncated'] or not 0<a['prompt_tokens']<a['full_tokens']<=1536
           or a['supervised_tokens']!=a['full_tokens']-a['prompt_tokens'] for a in labels):raise ValueError('label_or_length_check')
    if not report['likelihood_sanity']['passed'] or not math.isclose(report['likelihood_sanity']['manual_mean_nll'],report['likelihood_sanity']['builtin_mean_nll'],rel_tol=1e-5,abs_tol=1e-5):
        raise ValueError('manual_builtin_likelihood_failed')
    for name,h in report['adapter_files_sha256'].items():
        if file_sha(folder/'epoch_1/adapter'/name)!=h:raise ValueError('adapter_file_changed')
    adapter=checkpoint_header(folder/'epoch_1/adapter')
    if adapter['tensor_count']!=136 or adapter['parameters']!=2228224 or adapter['dtypes']!=['F32']:raise ValueError('wrong_adapter_layout')
    bundle={r['example_id']:r for r in rows(download/'masked_bundle.private.jsonl')}
    key=json.loads((download/'masked_key.private.json').read_text());mapping={(r['example_id'],r['snapshot']):r for r in key}
    judged=rows(download/'semantic_assessments.private.jsonl');judgments={(r['example_id'],r['label']):r for r in judged}
    if len(key)!=32 or len(mapping)!=32 or len(judgments)!=32 or len(judged)!=32:raise ValueError('assessment_coverage')
    outputs=[];summary={'protocol':'corpus_200_pilot_review_v1','slug':receipt['slug'],'version':1,
        'execution_complete':True,'training_pairs_seen':200,'optimizer_updates':100,'finite_nonzero_gradient_checks':100,
        'label_checks':216,'truncated_rows':0,'adapter':adapter,'snapshots':{},
        'elapsed_seconds':state['elapsed_seconds'],'peak_gpu_allocated_bytes':state['peak_gpu_allocated_bytes'],
        'runtime':report['runtime'],'detector_results':None,'new_GPU_jobs':0,'new_model_calls':0,
        'assessment_provenance':json.loads((download/'assessment_provenance.json').read_text())}
    for snap,name,step in zip(report['snapshots'],['base','epoch_1'],[0,100]):
        if not snap['complete'] or snap['name']!=name or snap['optimizer_step']!=step:raise ValueError('snapshot_boundary')
        for field,ids in [('development_losses',config['development_ids']),('training_probe_losses',config['training_probe_ids'])]:
            values=snap[field]
            if {r['example_id'] for r in values}!=set(ids) or len(values)!=len(ids) or any(not math.isfinite(r['loss']) or r['supervised_tokens']<=0 for r in values):raise ValueError('loss_coverage')
            score=sum(r['loss']*r['supervised_tokens'] for r in values)/sum(r['supervised_tokens'] for r in values)
            target=snap['development_token_weighted_nll' if field=='development_losses' else 'training_token_weighted_nll']
            if not math.isclose(score,target,abs_tol=1e-12):raise ValueError('weighted_loss_changed')
        if [g['example_id'] for g in snap['generations']]!=config['development_ids']:raise ValueError('generation_coverage')
        group=[]
        for generated in snap['generations']:
            rid=generated['example_id'];original=by_id[rid];entry=mapping[rid,name];assessment=judgments[rid,entry['label']]
            if (any(generated[k]!=original[k] for k in ['source','paper_id','style_group']) or generated['reference']!=original['target']
                    or generated['prompt_sha256']!=runtime.sha(original['prompt']) or runtime.sha(generated['output'])!=generated['output_sha256']
                    or generated['output_sha256']!=entry['text_sha256'] or generated['output_sha256']!=assessment['text_sha256']
                    or bundle[rid]['outputs'][entry['label']]!=generated['output']):raise ValueError('output_or_judgment_binding_failed')
            eligible=not assessment['material_claim_change'] and not assessment['unresolved'] and assessment['minor_semantic_shifts']<=1
            if eligible!=assessment['eligible_within_local_budget']:raise ValueError('eligibility_changed')
            item=dict(generated,snapshot=name,assessment=assessment,source_metrics=metrics(original['source'],generated['output']),
                      reference_metrics=metrics(original['target'],generated['output']))
            group.append(item);outputs.append(item)
        summary['snapshots'][name]={'outputs':len(group),'development_token_weighted_nll':snap['development_token_weighted_nll'],
            'training_probe_token_weighted_nll':snap['training_token_weighted_nll'],
            'material_claim_change_identified':sum(r['assessment']['material_claim_change'] for r in group),
            'eligible_within_local_budget':sum(r['assessment']['eligible_within_local_budget'] for r in group),
            'unresolved':sum(r['assessment']['unresolved'] for r in group),
            'empty':sum(r['empty'] for r in group),'output_limited':sum(r['output_limited'] for r in group),
            'exact_source_copies':sum(r['copied_source'] for r in group),
            'exact_word_token_copies':sum(r['source_metrics']['word_token_copy'] for r in group),
            'median_source_word_sequence_overlap':statistics.median(r['source_metrics']['word_sequence_overlap'] for r in group),
            'median_reference_word_sequence_overlap':statistics.median(r['reference_metrics']['word_sequence_overlap'] for r in group),
            'median_longest_shared_source_word_run':statistics.median(r['source_metrics']['longest_shared_word_run'] for r in group),
            'median_generated_words_matching_source_sequences_fraction':statistics.median(r['source_metrics']['generated_words_in_sequence_matches_fraction'] for r in group),
            'outputs_source_overlap_at_least_point8':sum(r['source_metrics']['word_sequence_overlap']>=.8 for r in group),
            'by_group':{g:{'outputs':4,'material_claim_change_identified':sum(r['assessment']['material_claim_change'] for r in group if r['style_group']==g),
                           'eligible_within_local_budget':sum(r['assessment']['eligible_within_local_budget'] for r in group if r['style_group']==g)} for g in runtime.GROUPS}}
    if len(report['snapshots'])!=2:raise ValueError('two_snapshots_required')
    paired={}
    for r in outputs:paired.setdefault(r['example_id'],{})[r['snapshot']]=r
    summary['paired_material_transitions']=dict(Counter('resolved' if p['base']['assessment']['material_claim_change'] and not p['epoch_1']['assessment']['material_claim_change'] else
        'introduced' if not p['base']['assessment']['material_claim_change'] and p['epoch_1']['assessment']['material_claim_change'] else
        'persistent' if p['epoch_1']['assessment']['material_claim_change'] else 'neither' for p in paired.values()))
    log=(raw/'corpus-200-pilot-4b.log').read_text()
    summary['credential_pattern_matches']=sum(len(re.findall(pattern,log)) for pattern in [r'hf_[A-Za-z0-9]{20,}',r'KGAT_[A-Za-z0-9_\-]{15,}',r'AKIA[0-9A-Z]{16}'])
    if summary['credential_pattern_matches']:raise ValueError('credential_pattern_requires_private_remediation')
    summary['limits']=['Automated partly masked same-agent semantic review; not independent calibrated ground truth.',
        'Sixteen examples from eight reused development papers; no sealed-test evaluation.',
        'Lower NLL and improved source retention do not establish distinctive author-style learning.',
        'Near copying can lower detector scores without learning a meaningful style transformation.',
        'The .8 overlap threshold is descriptive, not a predeclared success criterion.']
    output.mkdir(parents=True);write_rows(output/'outputs_and_assessments.private.jsonl',outputs)
    runtime.write_json(output/'summary.json',summary)
    runtime.write_json(output/'manifest.json',{'code_sha256':file_sha(Path(__file__)),'download_receipt_sha256':file_sha(download/'download_manifest.json'),
        'semantic_assessments_sha256':file_sha(download/'semantic_assessments.private.jsonl'),
        'files':{p.name:file_sha(p) for p in output.iterdir() if p.is_file()}})
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['download','packet','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();s=build(a.download,a.packet,a.output)
    print(json.dumps({k:s[k] for k in ['execution_complete','optimizer_updates','snapshots','paired_material_transitions']},indent=2))
