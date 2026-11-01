"""Offline execution and paired-generation audit for eleven seen training cases."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import statistics

from fama_style.evaluation import preference_training_inference as runtime
from .author_control_replication import verify_files
from .neutral_reconstruction_results import metrics


def read_rows(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_execution(jobs,config,report,state,audits):
    if (report.get('complete') is not True or state.get('complete') is not True or state.get('phase')!='complete'
            or report.get('mode')!='inference_only' or report['config']!=config
            or any(r.get('training_called') is not False or r.get('optimizer_steps')!=0 for r in (report,state))
            or state.get('completed_generations')!=22 or not 0<state['elapsed_seconds']<1800
            or not 0<state['peak_gpu_allocated_bytes']<16*1024**3):raise ValueError('complete_inference_only_execution_required')
    actual=report['runtime']
    if (any(actual[k]!=config[k] for k in ('model_id','model_revision','packages','runtime_source_sha256'))
            or actual['gpu_count']!=1 or 'T4' not in actual['gpu']):raise ValueError('frozen_model_runtime_changed')
    ids=[r['pair_id'] for r in jobs];by_id={r['pair_id']:r for r in jobs}
    if len(audits)!=11 or [r['pair_id'] for r in audits]!=ids:raise ValueError('eleven_prompt_audits_required')
    for audit in audits:
        row=by_id[audit['pair_id']]
        if (audit['prefix_ids_sha256']!=row['prefix_ids_sha256'] or audit['prompt_tokens']!=len(row['prefix_ids'])
                or audit['matches_training_prefix'] is not True):raise ValueError('training_token_prefix_not_reproduced')
    if [s['arm'] for s in report['snapshots']]!=list(runtime.ARMS):raise ValueError('two_fixed_adapters_required')
    for snap in report['snapshots']:
        if (snap['complete'] is not True or snap['all_parameters_frozen'] is not True or snap['checkpoint_epoch']!=2
                or [r['pair_id'] for r in snap['generations']]!=ids):raise ValueError('eleven_complete_generations_per_arm_required')
        for g in snap['generations']:
            row=by_id[g['pair_id']]
            if (any(g[k]!=row[k] for k in ('pair_id','example_id','anchor_group_id','paper_id','input_kind','source_sha256','prompt_sha256'))
                    or runtime.sha(g['output'])!=g['output_sha256'] or not 0<g['output_tokens']<=512
                    or g['empty']!=(not bool(g['output'].strip()))
                    or g['copied_source']!=(' '.join(g['output'].split())==' '.join(row['source'].split()))):
                raise ValueError('generation_text_or_binding_changed')


def masked_comparison(report,references):
    generated={(s['arm'],g['pair_id']):g for s in report['snapshots'] for g in s['generations']}
    bundle=[];key=[]
    for ref in references:
        pid=ref['pair_id'];order=['control','sft','dpo']
        random.Random(int(hashlib.sha256(('training-free-review-v1:'+pid).encode()).hexdigest(),16)).shuffle(order)
        row={k:ref[k] for k in ('pair_id','example_id','paper_id','family','input_kind','source','chosen')};row['outputs']={}
        for label,arm in zip('ABC',order):
            g=ref['control'] if arm=='control' else generated[(arm,pid)]
            row['outputs'][label]=g['output'];key.append({'pair_id':pid,'label':label,'arm':arm,'output_sha256':runtime.sha(g['output'])})
        bundle.append(row)
    return bundle,key


def summarize(bundle,key,judgments):
    mapping={(r['pair_id'],r['label']):r for r in key};cases={r['pair_id']:r for r in judgments['cases']}
    if (len(bundle)!=11 or len({r['pair_id'] for r in bundle})!=11 or len(key)!=33 or len(mapping)!=33
            or len(judgments['cases'])!=11 or set(cases)!={r['pair_id'] for r in bundle}):raise ValueError('masked_review_coverage')
    results=[];ranks={}
    for row in bundle:
        pid=row['pair_id'];judged=cases[pid];groups=judged['ranking_best_first']
        if any(not g for g in groups) or sorted(l for g in groups for l in g)!=list('ABC'):raise ValueError('invalid_generic_expression_ranks')
        rank={l:i for i,g in enumerate(groups) for l in g};local={}
        for label in 'ABC':
            k=mapping[(pid,label)];a=judged['outputs'][label];text=row['outputs'][label]
            if (k['arm'] in local or runtime.sha(text)!=k['output_sha256'] or a['output_sha256']!=k['output_sha256']
                    or type(a['material_claim_change']) is not bool or type(a['unresolved']) is not bool
                    or type(a['minor_semantic_shifts']) is not int or a['minor_semantic_shifts']<0 or not a['notes']):
                raise ValueError('masked_output_or_judgment_binding')
            local[k['arm']]=text;ranks[(k['arm'],pid)]=rank[label]
            results.append({'pair_id':pid,'example_id':row['example_id'],'paper_id':row['paper_id'],'family':row['family'],
                'arm':k['arm'],'output':text,'output_sha256':k['output_sha256'],'assessment':a,
                'eligible':not a['material_claim_change'] and not a['unresolved'] and a['minor_semantic_shifts']<=1,
                'source_metrics':metrics(row['source'],text),'reference_metrics':metrics(row['chosen'],text)})
        if set(local)!={'control','sft','dpo'}:raise ValueError('three_comparison_conditions_required')
        for a in local:
            for b in local:
                if local[a]==local[b] and ranks[(a,pid)]!=ranks[(b,pid)]:raise ValueError('identical_texts_must_tie')
    by={(r['arm'],r['pair_id']):r for r in results};ids=[r['pair_id'] for r in bundle];arms={}
    for arm in ('control','sft','dpo'):
        group=[r for r in results if r['arm']==arm]
        summary={'outputs':11,'material_claim_issues':sum(r['assessment']['material_claim_change'] for r in group),
            'eligible_within_local_budget':sum(r['eligible'] for r in group),
            'exact_control_outputs':sum(by[(arm,pid)]['output']==by[('control',pid)]['output'] for pid in ids),
            'median_source_word_sequence_overlap':statistics.median(r['source_metrics']['word_sequence_overlap'] for r in group),
            'median_reference_word_sequence_overlap':statistics.median(r['reference_metrics']['word_sequence_overlap'] for r in group),
            'median_longest_source_word_run':statistics.median(r['source_metrics']['longest_shared_word_run'] for r in group)}
        if arm!='control':
            preference=Counter('better' if ranks[(arm,pid)]<ranks[('control',pid)] else 'worse' if ranks[(arm,pid)]>ranks[('control',pid)] else 'tie' for pid in ids)
            summary['generic_expression_vs_control']={k:preference[k] for k in ('better','tie','worse')}
            summary['material_errors_resolved']=sum(by[('control',pid)]['assessment']['material_claim_change'] and not by[(arm,pid)]['assessment']['material_claim_change'] for pid in ids)
        summary['families']={family:{'outputs':len(g),'material_claim_issues':sum(r['assessment']['material_claim_change'] for r in g),
            'eligible':sum(r['eligible'] for r in g)} for family in sorted({r['family'] for r in group})
            for g in [[r for r in group if r['family']==family]]}
        arms[arm]=summary
    preference=Counter('better' if ranks[('dpo',pid)]<ranks[('sft',pid)] else 'worse' if ranks[('dpo',pid)]>ranks[('sft',pid)] else 'tie' for pid in ids)
    return {'arms':arms,'dpo_generic_expression_vs_sft':{k:preference[k] for k in ('better','tie','worse')}},results


def build(packet,download,review,output):
    if output.exists():raise ValueError('preserve_existing_results')
    verify_files(packet);jobs,config=runtime.load_inputs(packet/'upload/preference_training_jobs.jsonl',packet/'upload/preference_training_config.json')
    folder=download/'preference_training_inference_v1'
    report=json.loads((folder/'evaluation.json').read_text());state=json.loads((folder/'state.json').read_text())
    validate_execution(jobs,config,report,state,json.loads((folder/'token_audit.json').read_text()))
    if (runtime.file_sha(download/'preference_training_inference.py')!=config['runtime_source_sha256']
            or json.loads((folder/'generation_config.json').read_text())!=config['expected_generation_config']):raise ValueError('executed_source_or_decoding_changed')
    references=read_rows(packet/'private_key/reference_key.private.jsonl')
    if [r['pair_id'] for r in references]!=config['pair_ids']:raise ValueError('reference_key_order_changed')
    for ref,job in zip(references,jobs):
        if (ref['source']!=job['source'] or ref['prompt']!=job['prompt'] or runtime.sha(ref['chosen'])!=ref['chosen_sha256']
                or runtime.sha(ref['control']['output'])!=ref['rejected_sha256']):raise ValueError('reference_or_cached_control_changed')
    bundle,key=masked_comparison(report,references)
    if (bundle!=read_rows(review/'masked_bundle.private.jsonl') or key!=json.loads((review/'masked_key.private.json').read_text())):
        raise ValueError('masked_bundle_not_reproducible')
    judgments=json.loads((review/'masked_judgments.private.json').read_text());summary,results=summarize(bundle,key,judgments)
    summary.update(protocol='preference_training_inference_review_v1',slug='oliverlds/preference-training-inference-4b',
        user_reported_version=1,execution_complete=True,training_called=False,optimizer_steps=0,new_generations=22,
        reused_controls=11,prompt_audits_verified=11,elapsed_seconds=state['elapsed_seconds'],
        peak_gpu_allocated_bytes=state['peak_gpu_allocated_bytes'],primary_epoch=2,detector_calls=0,
        new_model_calls_during_review=0,new_gpu_jobs_submitted=0,additional_training_authorized=False,
        reserved_test_records_read=0,assessment_provisional=True,fama_style_established=False,
        limits=judgments['provenance']['limits'])
    flags={a:{k:sum(g[k] for s in report['snapshots'] if s['arm']==a for g in s['generations'])
        for k in ('empty','output_limited','copied_source')} for a in runtime.ARMS}
    for arm in flags:summary['arms'][arm].update(flags[arm])
    output.mkdir(parents=True);runtime.write_json(output/'summary.json',summary)
    (output/'per_output.private.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results))
    runtime.write_json(output/'receipt.json',{'packet_manifest_sha256':runtime.file_sha(packet/'manifest.json'),
        'runtime_review_sha256':runtime.file_sha(Path(__file__)),
        'download_sha256':{str(p.relative_to(download)):runtime.file_sha(p) for p in sorted(download.rglob('*')) if p.is_file()},
        'annotation_sha256':{p.name:runtime.file_sha(p) for p in sorted(review.glob('*.json*')) if p.is_file()}})
    runtime.write_json(output/'manifest.json',{'files':{p.name:runtime.file_sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','download','review','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.packet,a.download,a.review,a.output),indent=2))
