"""Offline, partly masked review of the completed bounded teacher-pair pilot."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import struct

from fama_style.training.teacher_pair_pilot import load_inputs,validate_baseline,sha,TARGET_PATTERN
from .author_control_replication import verify_files
from .neutral_reconstruction import write_json,write_rows
from .neutral_reconstruction_instruction import rows
from .neutral_reconstruction_results import file_sha,metrics
from .teacher_pair_baseline_review import validate_labels,bind_generations

PROTOCOL='teacher_pair_pilot_review_v1'
STAGES=('base','epoch_1','epoch_2')


def validate_execution(report,state,config,checks):
    if (report.get('complete') is not True or report.get('mode')!='pilot' or report['config']!=config
            or not report.get('training_called') or not report.get('adapter_weights_changed')
            or not report.get('adapter_reload_weights_match') or not state.get('complete')
            or state.get('optimizer_steps')!=16 or not state.get('training_called')
            or not state.get('adapter_reload_weights_match')):raise ValueError('pilot_completion_or_reload_failed')
    if len(checks)!=16 or [r['step'] for r in checks]!=list(range(1,17)):
        raise ValueError('sixteen_gradient_checks_required')
    if any(r['finite'] is not True or not math.isfinite(r['norm']) or r['norm']<=0 for r in checks):
        raise ValueError('gradient_check_failed')
    if [s['name'] for s in report['snapshots']]!=list(STAGES):raise ValueError('all_three_snapshots_required')
    for snap,step in zip(report['snapshots'],(0,8,16)):
        if not snap['complete'] or snap['optimizer_step']!=step:raise ValueError('snapshot_boundary_failed')
        if [r['example_id'] for r in snap['losses']]!=config['development_ids']:raise ValueError('development_loss_coverage')
        if any(not math.isfinite(r['loss']) or r['supervised_tokens']<=0 for r in snap['losses']):raise ValueError('invalid_loss')
        weighted=sum(r['loss']*r['supervised_tokens'] for r in snap['losses'])/sum(r['supervised_tokens'] for r in snap['losses'])
        if not math.isclose(weighted,snap['token_weighted_development_loss'],abs_tol=1e-12):raise ValueError('loss_aggregation_changed')


def join_masked(report,bundle,key,assessments,preferences,groups,config):
    raw={(s['name'],r['example_id']):r for s in report['snapshots'] for r in bind_generations(s['generations'],groups,config)}
    mapping={(r['example_id'],r['label']):r['snapshot'] for r in key}
    judged={(r['example_id'],r['label']):r for r in assessments}
    masked={(r['example_id'],label):text for r in bundle for label,text in r['outputs'].items()}
    if (len(mapping)!=30 or len(key)!=30 or len(judged)!=30 or len(assessments)!=30
            or len(masked)!=30 or set(mapping)!=set(judged) or set(mapping)!=set(masked)
            or {(stage,rid) for (rid,label),stage in mapping.items()}!=set(raw)):
        raise ValueError('masked_mapping_or_assessment_coverage')
    by_case={r['example_id']:r for r in bundle};pref_by={r['example_id']:r for r in preferences}
    if len(by_case)!=10 or len(bundle)!=10 or len(pref_by)!=10 or len(preferences)!=10 or set(pref_by)!=set(by_case):
        raise ValueError('case_or_preference_coverage')
    for rid,pref in pref_by.items():
        labels=[label for group in pref['ranking_best_first'] for label in group]
        if sorted(labels)!=list('ABC') or any(not group for group in pref['ranking_best_first']):raise ValueError('invalid_style_ranking')
    result=[]
    for (rid,label),stage in sorted(mapping.items()):
        row=raw[(stage,rid)];text=masked[(rid,label)];case=by_case[rid];assessment=judged[(rid,label)]
        if text!=row['output'] or sha(text)!=assessment['text_sha256']:raise ValueError('masked_text_changed')
        if case['source']!=row['source'] or case['reference']!=row['reference'] or case['split']!=row['split']:
            raise ValueError('masked_reference_or_split_changed')
        result.append(row|{'snapshot':stage,'masked_label':label,'assessment':assessment,
            'source_metrics':metrics(row['source'],row['output']),
            'reference_metrics':metrics(row['reference'],row['output']),'training_ready':False})
    return result,mapping


def checkpoint_header(folder):
    config=json.loads((folder/'adapter_config.json').read_text())
    if (config['r']!=8 or config['lora_alpha']!=16 or config['lora_dropout']!=0.05
            or config['target_modules']!=TARGET_PATTERN or config['base_model_name_or_path']!='google/gemma-3-4b-it'):
        raise ValueError('checkpoint_config_changed')
    path=folder/'adapter_model.safetensors'
    with path.open('rb') as stream:
        length=struct.unpack('<Q',stream.read(8))[0]
        if length>path.stat().st_size-8:raise ValueError('checkpoint_header_size')
        tensors={k:v for k,v in json.loads(stream.read(length)).items() if k!='__metadata__'}
    if not tensors or any('language_model' not in k or '.lora_' not in k or len(v['shape'])!=2 or 8 not in v['shape'] for k,v in tensors.items()):
        raise ValueError('non_language_or_wrong_rank_checkpoint')
    return {'tensor_count':len(tensors),'parameters':sum(math.prod(v['shape']) for v in tensors.values()),
        'dtypes':sorted({v['dtype'] for v in tensors.values()}),'sha256':file_sha(path),
        'base_revision_policy':'Use the pinned revision from the frozen runtime/config; adapter_config revision is not independently pinned.'}


def build(packet,download,baseline,review,output):
    if output.exists():raise ValueError('refuse_overwrite_pilot_review')
    verify_files(packet)
    groups,config=load_inputs(packet/'upload/teacher_pair_pilot.jsonl',packet/'upload/teacher_pair_config.json')
    previous=validate_baseline(baseline,config);folder=download/'teacher_pair_pilot_v1'
    report=json.loads((folder/'evaluation.json').read_text());state=json.loads((folder/'state.json').read_text())
    checks=json.loads((folder/'gradient_checks.json').read_text());validate_execution(report,state,config,checks)
    source_sha=json.loads((packet/'lineage.json').read_text())['runtime_source_sha256']
    if report['runtime']['runtime_source_sha256']!=source_sha or file_sha(download/'teacher_pair_pilot.py')!=source_sha:
        raise ValueError('runtime_source_changed')
    validate_labels(json.loads((folder/'label_audit.json').read_text()),groups)
    if report['previous_baseline_sha256']!=sha(json.dumps(previous,sort_keys=True)):raise ValueError('wrong_previous_baseline')
    exact=sum(a['output']==b['output'] for a,b in zip(report['snapshots'][0]['generations'],previous['snapshots'][0]['generations']))
    if exact!=10 or report['previous_baseline_exact_output_matches']!=exact:raise ValueError('base_outputs_differ')
    results,mapping=join_masked(report,rows(review/'masked_bundle.private.jsonl'),json.loads((review/'masked_key.private.json').read_text()),
        rows(review/'masked_assessments.private.jsonl'),rows(review/'masked_style_preferences.private.jsonl'),groups,config)
    prefs=rows(review/'masked_style_preferences.private.jsonl')
    by_stage={stage:{r['example_id']:r for r in results if r['snapshot']==stage} for stage in STAGES}
    summary={'protocol':PROTOCOL,'actual_slug':'oliverlds/teacher-pair-pilot-4b','user_reported_version':1,
        'execution_complete':True,'optimizer_steps':16,'gradient_checks':16,'adapter_weights_changed':True,
        'adapter_reload_weights_match':True,'baseline_exact_outputs_reproduced':10,'snapshots':{},
        'elapsed_seconds':state['elapsed_seconds'],'peak_gpu_allocated_bytes':state['peak_gpu_allocated_bytes'],
        'checkpoints':{str(i):checkpoint_header(folder/f'epoch_{i}/adapter') for i in (1,2)},
        'new_model_calls':0,'additional_training_authorized':False,
        'limits':['Partly masked automated review; baseline was previously seen, conditions may be inferred, and judgments are not independent or calibrated.',
                  'Eight reused development cases, four author/paper-held-out sources; no reserved test evaluation.',
                  'Generic concision/organization preferences are not evidence of Fama-specific style or human authorship.']}
    paper_by={r['example_id']:r['paper_id'] for group in groups.values() for r in group}
    for snap in report['snapshots']:
        stage=snap['name'];item={'optimizer_step':snap['optimizer_step'],'token_weighted_development_loss':snap['token_weighted_development_loss'],
            'mean_example_development_loss':snap['mean_example_development_loss'],'splits':{}}
        for split in ('development','training'):
            group=[r for r in by_stage[stage].values() if r['split']==split]
            item['splits'][split]={'outputs':len(group),'material_claim_issues':sum(r['assessment']['material_claim_issue_identified'] for r in group),
                'exact_source_copies':sum(r['copied_source'] for r in group),'empty':sum(r['empty'] for r in group),
                'output_limited':sum(r['output_limited'] for r in group),
                'median_source_overlap':statistics.median(r['source_metrics']['word_sequence_overlap'] for r in group),
                'median_reference_overlap':statistics.median(r['reference_metrics']['word_sequence_overlap'] for r in group)}
        if stage!='base':
            outcomes={'better':0,'tie':0,'worse':0};papers=set();meaning={'improved':0,'same':0,'worse':0}
            for pref in prefs:
                if pref['split']!='development':continue
                rid=pref['example_id'];ranks={mapping[(rid,label)]:i for i,g in enumerate(pref['ranking_best_first']) for label in g}
                result='better' if ranks[stage]<ranks['base'] else 'worse' if ranks[stage]>ranks['base'] else 'tie'
                outcomes[result]+=1
                if result=='better':papers.add(paper_by[rid])
                old=by_stage['base'][rid]['assessment']['material_claim_issue_identified'];new=by_stage[stage][rid]['assessment']['material_claim_issue_identified']
                meaning['improved' if old and not new else 'worse' if new and not old else 'same']+=1
            item['style_preference_vs_base_development']=outcomes;item['style_win_papers']=len(papers)
            item['material_issue_transitions_vs_base']=meaning
            item['development_outputs_identical_to_base']=sum(by_stage[stage][rid]['output']==r['output'] for rid,r in by_stage['base'].items() if r['split']=='development')
        summary['snapshots'][stage]=item
    summary['final_loss_relative_reduction']=1-summary['snapshots']['epoch_2']['token_weighted_development_loss']/summary['snapshots']['base']['token_weighted_development_loss']
    summary['development_outputs_identical_between_epochs']=sum(by_stage['epoch_1'][rid]['output']==r['output'] for rid,r in by_stage['epoch_2'].items() if r['split']=='development')
    reload=report['reload_probe'];expected=by_stage['epoch_2'][config['development_ids'][0]]
    if reload['example_id']!=expected['example_id'] or reload['source']!=expected['source'] or reload['reference']!=expected['reference']:
        raise ValueError('reload_probe_binding_changed')
    summary['reload_probe_matches_epoch_2']=reload['output']==expected['output']
    output.mkdir(parents=True);write_rows(output/'per_output.private.jsonl',results);write_json(output/'summary.json',summary)
    write_json(output/'receipt.json',{'download_files_sha256':{str(p.relative_to(download)):file_sha(p) for p in sorted(download.rglob('*')) if p.is_file()},
        'packet_manifest_sha256':file_sha(packet/'manifest.json'),'external_baseline_sha256':file_sha(baseline),
        'review_inputs_sha256':{p.name:file_sha(p) for p in sorted(review.iterdir()) if p.is_file()},
        'metrics_code_sha256':file_sha(Path(__file__).with_name('neutral_reconstruction_results.py'))})
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'code_sha256':file_sha(Path(__file__)),
        'files':{p.name:file_sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','download','baseline','review','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();print(json.dumps(build(args.packet,args.download,args.baseline,args.review,args.output)))


if __name__=='__main__':main()
