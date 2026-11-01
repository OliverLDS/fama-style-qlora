"""Offline matched-budget review of the fresh input-replacement SFT experiment."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from fama_style.training import teacher_pair_input_replacement as runtime
from .author_control_replication import verify_files
from .neutral_reconstruction import write_json,write_rows
from .neutral_reconstruction_results import file_sha,metrics
from .teacher_pair_expanded_pilot_review import validate_execution,checkpoint_header
from .teacher_pair_expanded_baseline_review import validate_labels
from .teacher_pair_baseline_review import bind_generations

PROTOCOL='input_replacement_review_v1'
CONDITIONS=('base','control_epoch_1','control_epoch_2','replacement_epoch_1','replacement_epoch_2')


def rows(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def join_masked(raw,bundle,key,assessments,preferences,historical):
    mapping={(r['example_id'],r['label']):r['condition'] for r in key}
    judged={(r['example_id'],r['label']):r for r in assessments}
    masked={(r['example_id'],label):text for r in bundle for label,text in r['outputs'].items()}
    cases={r['example_id']:r for r in bundle};prefs={r['example_id']:r for r in preferences}
    if (len(raw)!=50 or len(key)!=50 or len(mapping)!=50 or len(assessments)!=50 or len(judged)!=50
            or len(masked)!=50 or set(mapping)!=set(judged) or set(mapping)!=set(masked)
            or {(condition,rid) for (rid,label),condition in mapping.items()}!=set(raw)
            or len(bundle)!=10 or len(cases)!=10 or len(preferences)!=10 or len(prefs)!=10 or set(cases)!=set(prefs)):
        raise ValueError('complete_five_way_case_and_assessment_coverage_required')
    old={(r['example_id'],r['output']):r['assessment'] for r in historical}
    ranks={}
    for rid,pref in prefs.items():
        groups=pref['ranking_best_first']
        if sorted(label for group in groups for label in group)!=list('ABCDE') or any(not group for group in groups):
            raise ValueError('five_unique_style_labels_required')
        ranks[rid]={label:i for i,group in enumerate(groups) for label in group}
        for label,text in cases[rid]['outputs'].items():
            for other,other_text in cases[rid]['outputs'].items():
                if text==other_text and ranks[rid][label]!=ranks[rid][other]:raise ValueError('identical_texts_must_tie')
    results=[]
    for (rid,label),condition in sorted(mapping.items()):
        row=raw[(condition,rid)];case=cases[rid];assessment=judged[(rid,label)]
        if (masked[(rid,label)]!=row['output'] or assessment['text_sha256']!=runtime.sha(row['output'])
                or any(case[k]!=row[k] for k in ('source','reference','split'))
                or type(assessment['material_claim_issue_identified']) is not bool or not assessment['notes']):
            raise ValueError('masked_evidence_or_assessment_binding_changed')
        if assessment['reused_text_bound_assessment']:
            prior=old.get((rid,row['output']))
            if prior is None or any(prior[k]!=assessment[k] for k in ('text_sha256','material_claim_issue_identified','notes')):
                raise ValueError('reused_judgment_not_bound_to_historical_evidence')
        results.append(row|{'condition':condition,'masked_label':label,'style_rank':ranks[rid][label],
            'assessment':assessment,'source_metrics':metrics(row['source'],row['output']),
            'reference_metrics':metrics(row['reference'],row['output'])})
    return results


def paired(control,treatment):
    by={r['example_id']:r for r in control}
    if len(by)!=len(control) or len(treatment)!=len(control) or {r['example_id'] for r in treatment}!=set(by):
        raise ValueError('same_unique_cases_for_matched_comparison')
    meaning=Counter();style=Counter();matches=0
    for new in treatment:
        old=by[new['example_id']]
        a=old['assessment']['material_claim_issue_identified'];b=new['assessment']['material_claim_issue_identified']
        meaning['improved' if a and not b else 'worse' if b and not a else 'same']+=1
        style['better' if new['style_rank']<old['style_rank'] else 'worse' if new['style_rank']>old['style_rank'] else 'tie']+=1
        matches+=old['output']==new['output']
    return {'material_issue_transitions':{k:meaning[k] for k in ('improved','same','worse')},
        'generic_style_preferences':{k:style[k] for k in ('better','tie','worse')},'identical_outputs':matches}


def build(packet,download,baseline,control_report_path,control_review,review,authorization,output):
    if output.exists():raise ValueError('refuse_overwrite_input_replacement_review')
    verify_files(packet);verify_files(control_review)
    plan=json.loads((packet/'plan.json').read_text());approved=json.loads(authorization.read_text())
    if (approved.get('user_authorized') is not True or approved.get('optimizer_steps')!=46
            or approved.get('fresh_adapter_from_base') is not True or approved.get('full_training_authorized') is not False
            or approved['packet_manifest_sha256']!=file_sha(packet/'manifest.json')
            or approved['baseline_evaluation_sha256']!=file_sha(baseline)):
        raise ValueError('approved_scope_or_packet_changed')
    groups,config=runtime.load_inputs(packet/'upload/teacher_pair_replacement.jsonl',packet/'upload/teacher_pair_replacement_config.json')
    previous=runtime.validate_baseline(baseline,config);folder=download/'input_replacement_pilot_v1'
    report=json.loads((folder/'evaluation.json').read_text());state=json.loads((folder/'state.json').read_text())
    checks=json.loads((folder/'gradient_checks.json').read_text());validate_execution(report,state,config,checks)
    if (report.get('fresh_adapter_from_base') is not True or report.get('fresh_adapter_zero_B_verified') is not True
            or report['control_base_check']!=runtime.compare_base(report['snapshots'][0],previous)):
        raise ValueError('fresh_initialization_or_common_base_failed')
    if (report['runtime']['runtime_source_sha256']!=plan['runtime_source_sha256']
            or file_sha(download/'teacher_pair_input_replacement.py')!=plan['runtime_source_sha256']
            or approved['runtime_source_sha256']!=plan['runtime_source_sha256']):raise ValueError('runtime_changed')
    validate_labels(json.loads((folder/'label_audit.json').read_text()),groups)
    if report['previous_baseline_sha256']!=runtime.sha(json.dumps(previous,sort_keys=True)):raise ValueError('wrong_baseline_report')
    receipt=json.loads((control_review/'receipt.json').read_text())
    if (receipt['download_files_sha256']['teacher_pair_pilot_v2/evaluation.json']!=file_sha(control_report_path)
            or file_sha(control_review/'manifest.json')!=plan['control_pilot_review_manifest_sha256']):
        raise ValueError('matched_control_report_changed')
    control=json.loads(control_report_path.read_text())
    if control['config']!=config['control_config']:raise ValueError('control_settings_changed')
    raw={('base',g['example_id']):g for g in bind_generations(report['snapshots'][0]['generations'],groups,config)}
    snapshots={'base':report['snapshots'][0]}
    for origin,run in [('control',control),('replacement',report)]:
        for snap in run['snapshots'][1:]:
            condition=origin+'_'+snap['name'];snapshots[condition]=snap
            if snap['optimizer_step']!=({'epoch_1':23,'epoch_2':46}[snap['name']]):raise ValueError('matched_step_boundary_changed')
            for g in bind_generations(snap['generations'],groups,config):raw[(condition,g['example_id'])]=g
    result=join_masked(raw,rows(review/'masked_bundle.private.jsonl'),json.loads((review/'masked_key.private.json').read_text()),
        rows(review/'masked_assessments.private.jsonl'),rows(review/'masked_style_preferences.private.jsonl'),rows(control_review/'per_output.private.jsonl'))
    by={condition:[r for r in result if r['condition']==condition] for condition in CONDITIONS}
    summary={'protocol':PROTOCOL,'actual_slug':'oliverlds/input-replacement-4b','requested_version':1,
        'execution_complete':True,'optimizer_steps':46,'gradient_checks':46,'fresh_adapter_from_base':True,
        'fresh_zero_B_verified':True,'adapter_weights_changed':True,'adapter_reload_weights_match':True,
        'common_base_check':report['control_base_check'],'label_audits':54,'max_full_tokens':max(a['full_tokens'] for a in json.loads((folder/'label_audit.json').read_text())),
        'elapsed_seconds':state['elapsed_seconds'],'peak_gpu_allocated_bytes':state['peak_gpu_allocated_bytes'],
        'reused_semantic_assessments':sum(r['assessment']['reused_text_bound_assessment'] for r in result),
        'new_semantic_assessments':sum(not r['assessment']['reused_text_bound_assessment'] for r in result),
        'checkpoints':{str(i):checkpoint_header(folder/f'epoch_{i}/adapter') for i in (1,2)},'conditions':{},'matched_comparisons':{},
        'new_model_calls_during_review':0,'additional_training_authorized':False,'full_scale_training_ready':False,
        'limits':['Single-seed matched input-replacement comparison on eight reused development examples; no reserved test evaluation.',
            'Same reviewer, five-way shuffled labels; historical outputs known and exact-evidence judgments reused. Not independent or fully blind.',
            'Generic expression preferences, source overlap and reference loss do not establish a Fama signature, human authorship or detector improvement.',
            'Only input regime is intentionally changed, including wording/organization/length; this does not isolate lexical overlap as a causal factor.']}
    for condition,group in by.items():
        snap=snapshots[condition];item={'optimizer_step':snap['optimizer_step'],'token_weighted_development_loss':snap['token_weighted_development_loss'],'splits':{}}
        for split in ('development','training'):
            selected=[r for r in group if r['split']==split]
            item['splits'][split]={'outputs':len(selected),'material_claim_issues':sum(r['assessment']['material_claim_issue_identified'] for r in selected),
                'exact_source_copies':sum(r['copied_source'] for r in selected),'empty':sum(r['empty'] for r in selected),
                'output_limited':sum(r['output_limited'] for r in selected),
                'median_source_overlap':statistics.median(r['source_metrics']['word_sequence_overlap'] for r in selected),
                'median_reference_overlap':statistics.median(r['reference_metrics']['word_sequence_overlap'] for r in selected)}
        if condition!='base':item['vs_base_development']=paired([r for r in by['base'] if r['split']=='development'],[r for r in group if r['split']=='development'])
        summary['conditions'][condition]=item
    for epoch in (1,2):
        a=[r for r in by[f'control_epoch_{epoch}'] if r['split']=='development'];b=[r for r in by[f'replacement_epoch_{epoch}'] if r['split']=='development']
        summary['matched_comparisons'][str(epoch)]=paired(a,b)
    final=summary['conditions']['replacement_epoch_2'];ctrl=summary['conditions']['control_epoch_2'];style=summary['matched_comparisons']['2']['generic_style_preferences']
    summary['predeclared_final_gate']={'fewer_than_six_material_issue_outputs':final['splits']['development']['material_claim_issues']<6,
        'at_least_six_expression_wins_vs_base':final['vs_base_development']['generic_style_preferences']['better']>=6,
        'no_net_expression_loss_vs_matched_control':style['better']>=style['worse'],
        'no_increase_in_copying_empty_or_limited':all(final['splits']['development'][k]<=ctrl['splits']['development'][k] for k in ('exact_source_copies','empty','output_limited'))}
    summary['predeclared_final_gate']['all_pass']=all(summary['predeclared_final_gate'].values())
    summary['final_loss_relative_difference_vs_control']=final['token_weighted_development_loss']/ctrl['token_weighted_development_loss']-1
    reload=report['reload_probe'];expected=next(g for g in report['snapshots'][2]['generations'] if g['example_id']==config['development_ids'][0])
    if any(reload[k]!=expected[k] for k in ('example_id','source','reference','output')):raise ValueError('reload_probe_changed')
    summary['reload_probe_matches_final']=True
    output.mkdir(parents=True);write_rows(output/'per_output.private.jsonl',result);write_json(output/'summary.json',summary)
    write_json(output/'receipt.json',{'packet_manifest_sha256':file_sha(packet/'manifest.json'),'authorization_sha256':file_sha(authorization),
        'control_review_manifest_sha256':file_sha(control_review/'manifest.json'),'baseline_sha256':file_sha(baseline),
        'download_sha256':{str(p.relative_to(download)):file_sha(p) for p in sorted(download.rglob('*')) if p.is_file()},
        'review_sha256':{p.name:file_sha(p) for p in sorted(review.iterdir()) if p.is_file()}})
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'code_sha256':file_sha(Path(__file__)),
        'files':{p.name:file_sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','download','baseline','control-report-path','control-review','review','authorization','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();print(json.dumps(build(args.packet,args.download,args.baseline,args.control_report_path,args.control_review,args.review,args.authorization,args.output)))


if __name__=='__main__':main()
