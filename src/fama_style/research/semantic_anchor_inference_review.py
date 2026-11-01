"""Offline audit of all anchor rewrites and conditional-likelihood contrasts."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import statistics

from fama_style.evaluation import semantic_anchor_4b as runtime
from .author_control_replication import verify_files
from .neutral_reconstruction import write_json,write_rows
from .neutral_reconstruction_results import file_sha,metrics

PROTOCOL='semantic_anchor_inference_review_v1'


def rows(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def validate_execution(report,state,audit,jobs,config):
    if (report.get('complete') is not True or report['config']!=config or report.get('training_called') is not False
            or state.get('complete') is not True or state.get('training_called') is not False
            or state.get('optimizer_steps')!=0 or state.get('completed_generations')!=48
            or state.get('completed_completion_scores')!=96):raise ValueError('complete_zero_update_run_required')
    expected={(j['job_id'],c) for j in jobs for c in ('chosen','rejected')}
    by={(a['job_id'],a['candidate']):a for a in audit}
    if len(audit)!=48 or len(by)!=48 or set(by)!=expected:raise ValueError('all_48_label_audits_required')
    for a in audit:
        if (a['decoded_target_matches'] is not True or not 0<a['prompt_tokens']<a['full_tokens']<=runtime.MAX_LENGTH
                or a['full_tokens']-a['prompt_tokens']!=a['supervised_tokens']
                or a['prompt_tokens']+runtime.DECODING['max_new_tokens']>runtime.MAX_LENGTH
                or re.fullmatch('[0-9a-f]{64}',a['input_ids_sha256']) is None):
            raise ValueError('label_boundary_or_context_failed')
    if [s['name'] for s in report['snapshots']]!=list(runtime.STAGES):raise ValueError('both_frozen_stages_required')
    for snap in report['snapshots']:
        if not snap['complete'] or [r['job_id'] for r in snap['generations']]!=[j['job_id'] for j in jobs]:
            raise ValueError('all_24_first_generations_required')
        for g,j in zip(snap['generations'],jobs):
            if any(g[k]!=j[k] for k in ('example_id','paper_id','family','input_kind','source','prompt','target_sha256')):
                raise ValueError('generation_evidence_changed')
            if (g['empty']!=(not bool(g['output'].strip())) or g['copied_source']!=(' '.join(g['output'].split())==' '.join(j['source'].split()))
                    or not 0<=g['output_tokens']<=runtime.DECODING['max_new_tokens']):raise ValueError('generation_flags_changed')
        scores={(r['job_id'],r['candidate']):r for r in snap['scores']}
        if len(scores)!=48 or len(snap['scores'])!=48 or set(scores)!=expected:raise ValueError('all_48_completion_scores_required')
        contrasts={r['job_id']:r for r in snap['contrasts']}
        if len(contrasts)!=24 or len(snap['contrasts'])!=24:raise ValueError('all_24_contrasts_required')
        for job in jobs:
            pair=[]
            for candidate in ('chosen','rejected'):
                score=scores[(job['job_id'],candidate)]
                if (score['completion_sha256']!=runtime.sha(job[candidate])
                        or score['supervised_tokens']!=by[(job['job_id'],candidate)]['supervised_tokens']
                        or not math.isfinite(score['mean_nll']) or score['mean_nll']<0
                        or not math.isclose(score['total_nll'],score['mean_nll']*score['supervised_tokens'],abs_tol=1e-12,rel_tol=0)):
                    raise ValueError('completion_score_binding_changed')
                pair.append(score)
            contrast=contrasts[job['job_id']]
            if any(contrast[k]!=job[k] for k in ('example_id','family','input_kind')):raise ValueError('contrast_condition_changed')
            if any(contrast[k]!=v for k,v in runtime.margin(*pair).items()):raise ValueError('contrast_margin_or_sign_changed')


def join_assessments(report,jobs,anchors,bundle,key,assessments,preferences):
    generated={(s['name'],g['job_id']):g for s in report['snapshots'] for g in s['generations']}
    mapping={(r['example_id'],r['label']):r for r in key}
    judged={(r['example_id'],r['label']):r for r in assessments}
    masked={(r['example_id'],label):text for r in bundle for label,text in r['outputs'].items()}
    if (len(key)!=48 or len(mapping)!=48 or len(assessments)!=48 or len(judged)!=48
            or len(masked)!=48 or set(mapping)!=set(judged) or set(mapping)!=set(masked)
            or {(r['stage'],r['job_id']) for r in key}!=set(generated)):
        raise ValueError('complete_unique_masked_mapping_required')
    anchor_by={r['example_id']:r for r in anchors};cases={r['example_id']:r for r in bundle}
    prefs={r['example_id']:r for r in preferences};job_by={r['job_id']:r for r in jobs}
    if len(bundle)!=12 or len(cases)!=12 or len(preferences)!=12 or len(prefs)!=12 or set(prefs)!=set(cases) or set(cases)!=set(anchor_by):
        raise ValueError('all_12_cases_and_preferences_required')
    ranks={}
    for rid,case in cases.items():
        a=anchor_by[rid]
        if case['original_input']!=a['source'] or case['variant_input']!=a['faithful_variant'] or case['reference']!=a['chosen']:
            raise ValueError('masked_anchor_evidence_changed')
        if case['claims']!=[{k:c[k] for k in ('claim_id','statement','constraints')} for c in a['claims']]:
            raise ValueError('masked_claim_inventory_changed')
        ranking=prefs[rid]['ranking_best_first']
        if sorted(label for group in ranking for label in group)!=list('ABCD') or any(not group for group in ranking):
            raise ValueError('complete_four_way_style_ranking_required')
        ranks[rid]={label:i for i,group in enumerate(ranking) for label in group}
        for label,text in case['outputs'].items():
            for other,other_text in case['outputs'].items():
                if text==other_text and ranks[rid][label]!=ranks[rid][other]:raise ValueError('identical_outputs_must_tie')
    result=[]
    for marker,row in sorted(mapping.items()):
        g=generated[(row['stage'],row['job_id'])];job=job_by[row['job_id']];assessment=judged[marker]
        if (g['example_id']!=row['example_id'] or g['input_kind']!=row['input_kind'] or masked[marker]!=g['output']
                or assessment['text_sha256']!=runtime.sha(g['output']) or not assessment['notes']
                or type(assessment['material_claim_issue_identified']) is not bool):raise ValueError('masked_text_or_condition_changed')
        result.append(g|{'stage':row['stage'],'masked_label':row['label'],'assessment':assessment,
            'style_rank':ranks[row['example_id']][row['label']],
            'source_metrics':metrics(job['source'],g['output']),'reference_metrics':metrics(job['chosen'],g['output'])})
    return result


def compare(group_a,group_b):
    by={r['example_id']:r for r in group_a};transitions=Counter();style=Counter()
    if len(by)!=len(group_a) or {r['example_id'] for r in group_b}!=set(by) or len(group_b)!=len(group_a):
        raise ValueError('same_unique_cases_required_for_paired_comparison')
    for b in group_b:
        a=by[b['example_id']];old=a['assessment']['material_claim_issue_identified'];new=b['assessment']['material_claim_issue_identified']
        transitions['improved' if old and not new else 'worse' if new and not old else 'same']+=1
        style['better' if b['style_rank']<a['style_rank'] else 'worse' if b['style_rank']>a['style_rank'] else 'tie']+=1
    return {'material_issue_transitions':dict(transitions),'generic_style_preferences':dict(style)}


def build(packet,download,review,output):
    if output.exists():raise ValueError('refuse_overwrite_anchor_review')
    verify_files(packet);jobs,config=runtime.load_inputs(packet/'upload/semantic_anchor_jobs.jsonl',packet/'upload/semantic_anchor_config.json')
    folder=download/'semantic_anchor_4b_v1';report=json.loads((folder/'evaluation.json').read_text());state=json.loads((folder/'state.json').read_text())
    audit=json.loads((folder/'label_audit.json').read_text());validate_execution(report,state,audit,jobs,config)
    plan=json.loads((packet/'plan.json').read_text());actual=report['runtime']
    if (actual['runtime_source_sha256']!=plan['runtime_source_sha256'] or file_sha(download/'semantic_anchor_4b.py')!=plan['runtime_source_sha256']
            or actual['model_id']!=runtime.MODEL_ID or actual['model_revision']!=runtime.REVISION
            or actual['gpu_count']!=1 or 'T4' not in actual['gpu']):raise ValueError('runtime_or_hardware_changed')
    results=join_assessments(report,jobs,rows(packet/'private_key/anchors.private.jsonl'),rows(review/'masked_bundle.private.jsonl'),
        json.loads((review/'masked_key.private.json').read_text()),rows(review/'masked_assessments.private.jsonl'),rows(review/'masked_style_preferences.private.jsonl'))
    summary={'protocol':PROTOCOL,'actual_slug':'oliverlds/semantic-anchor-4b','requested_version':1,
        'execution_complete':True,'training_called':False,'optimizer_steps':0,'generations':48,'completion_scores':96,
        'label_audits':48,'max_full_tokens':max(r['full_tokens'] for r in audit),
        'max_prompt_tokens':max(r['prompt_tokens'] for r in audit),'elapsed_seconds':state['elapsed_seconds'],
        'peak_gpu_allocated_bytes':state['peak_gpu_allocated_bytes'],'groups':{},'variant_vs_original':{},'adapter_vs_base':{},
        'new_model_calls_during_review':0,'additional_training_authorized':False,
        'limits':['Twelve training-side cases; adapter saw original inputs and all targets. Input wording robustness is not held-out generalization.',
            'Same reviewer created variants and judged outputs; shuffled labels reduce visible condition cues but are not independent or fully blind.',
            'Material-issue counts and generic style rankings are provisional; four severity-boundary judgments are explicitly retained.',
            'Likelihood contrasts target specific synthetic edits, whereas generation assessment covers all substantive claims. Neither is a Fama-style or detector benchmark.']}
    lookup={}
    for stage in runtime.STAGES:
        snap=next(s for s in report['snapshots'] if s['name']==stage)
        for kind in ('original','variant'):
            group=[r for r in results if r['stage']==stage and r['input_kind']==kind];lookup[(stage,kind)]=group
            contrasts=[r for r in snap['contrasts'] if r['input_kind']==kind]
            material=sum(r['assessment']['material_claim_issue_identified'] for r in group)
            uncertain_bad=sum(r['assessment']['material_claim_issue_identified'] and r['assessment']['severity_boundary_uncertain'] for r in group)
            uncertain_good=sum(not r['assessment']['material_claim_issue_identified'] and r['assessment']['severity_boundary_uncertain'] for r in group)
            name=stage+'__'+kind
            summary['groups'][name]={'outputs':len(group),'material_claim_issues':material,
                'severity_sensitivity_range':[material-uncertain_bad,material+uncertain_good],
                'exact_source_copies':sum(r['copied_source'] for r in group),'empty':sum(r['empty'] for r in group),
                'output_limited':sum(r['output_limited'] for r in group),
                'median_source_overlap':statistics.median(r['source_metrics']['word_sequence_overlap'] for r in group),
                'median_reference_overlap':statistics.median(r['reference_metrics']['word_sequence_overlap'] for r in group),
                'mean_nll_preferences':dict(Counter(r['mean_nll_preference'] for r in contrasts)),
                'total_nll_preferences':dict(Counter(r['total_nll_preference'] for r in contrasts)),
                'mean_total_disagreements':sum(r['mean_total_disagree'] for r in contrasts),
                'median_mean_nll_margin':statistics.median(r['mean_nll_margin_rejected_minus_chosen'] for r in contrasts),
                'chosen_preferred_but_generation_material_issue':sum(c['mean_nll_preference']=='chosen' and g['assessment']['material_claim_issue_identified'] for c in contrasts for g in group if c['job_id']==g['job_id']),
                'families':{family:{'outputs':3,'material_claim_issues':sum(r['assessment']['material_claim_issue_identified'] for r in group if r['family']==family),
                    'mean_nll_preferences':dict(Counter(r['mean_nll_preference'] for r in contrasts if r['family']==family))} for family in ('scope','time','numbers','reasoning')}}
        summary['variant_vs_original'][stage]=compare(lookup[(stage,'original')],lookup[(stage,'variant')])
    for kind in ('original','variant'):summary['adapter_vs_base'][kind]=compare(lookup[('base',kind)],lookup[('expanded_epoch_2',kind)])
    output.mkdir(parents=True);write_rows(output/'per_output.private.jsonl',results);write_json(output/'summary.json',summary)
    write_json(output/'receipt.json',{'packet_manifest_sha256':file_sha(packet/'manifest.json'),
        'download_sha256':{str(p.relative_to(download)):file_sha(p) for p in sorted(download.rglob('*')) if p.is_file()},
        'review_sha256':{p.name:file_sha(p) for p in sorted(review.iterdir()) if p.is_file()},
        'metrics_sha256':file_sha(Path(__file__).with_name('neutral_reconstruction_results.py'))})
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'code_sha256':file_sha(Path(__file__)),
        'files':{p.name:file_sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','download','review','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();print(json.dumps(build(args.packet,args.download,args.review,args.output)))


if __name__=='__main__':main()
