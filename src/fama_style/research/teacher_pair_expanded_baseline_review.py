"""CPU review of the expanded baseline; reuse judgments only for identical evidence."""
import argparse
import json
import math
from pathlib import Path
import re
import statistics

from fama_style.training import teacher_pair_expanded_pilot as runtime
from .author_control_replication import verify_files
from .local_reconstruction_review import join_assessments
from .neutral_reconstruction import write_json,write_rows
from .neutral_reconstruction_instruction import rows
from .neutral_reconstruction_results import file_sha,metrics
from .teacher_pair_baseline_review import bind_generations

PROTOCOL='teacher_pair_expanded_baseline_review_v1'


def validate_labels(audit,groups):
    expected={r['example_id']:r['split'] for group in groups.values() for r in group}
    if (len(expected)!=54 or len(audit)!=54 or len({r['example_id'] for r in audit})!=54
            or {r['example_id'] for r in audit}!=set(expected)):
        raise ValueError('all_54_label_checks_required')
    for row in audit:
        if (row['split']!=expected[row['example_id']] or row['labels_match_target'] is not True
                or not 0<row['masked_tokens']<row['full_tokens']<=runtime.HYPER['max_length']
                or row['supervised_tokens']!=row['full_tokens']-row['masked_tokens']
                or row['masked_tokens']+runtime.DECODING['max_new_tokens']>runtime.HYPER['max_length']
                or re.fullmatch('[0-9a-f]{64}',row['input_ids_sha256']) is None):
            raise ValueError('label_or_generation_budget_check_failed')


def reuse_assessments(current,previous,assessments,groups,config):
    bound=bind_generations(current,groups,config)
    bind_generations(previous,groups,config)
    fields=('example_id','source','reference','split','prompt_sha256','output')
    if any(any(a[k]!=b[k] for k in fields) for a,b in zip(current,previous)):
        raise ValueError('cannot_reuse_review_for_changed_evidence')
    for row in current:
        if (row['empty']!= (not bool(row['output'].strip()))
                or row['copied_source']!=(' '.join(row['output'].split())==' '.join(row['source'].split()))):
            raise ValueError('generation_flag_mismatch')
    return join_assessments(bound,assessments)


def build(packet,download,previous_download,previous_review,output):
    if output.exists():raise ValueError('refuse_overwrite_expanded_review')
    verify_files(packet);verify_files(previous_review/'results')
    groups,config=runtime.load_inputs(packet/'upload/teacher_pair_expanded.jsonl',packet/'upload/teacher_pair_expanded_config.json')
    plan=json.loads((packet/'plan.json').read_text());folder=download/'teacher_pair_base_v2'
    report=runtime.validate_baseline(folder/'evaluation.json',config)
    state=json.loads((folder/'state.json').read_text())
    if (state.get('complete') is not True or state.get('training_called') is not False
            or state.get('optimizer_steps')!=0 or state.get('mode')!='baseline'
            or report.get('training_called') is not False or len(report['snapshots'])!=1):
        raise ValueError('complete_zero_update_baseline_required')
    if (file_sha(download/'teacher_pair_expanded_pilot.py')!=plan['runtime_source_sha256']
            or report['runtime']['model_id']!=runtime.MODEL_ID or report['runtime']['gpu_count']!=1
            or 'T4' not in report['runtime']['gpu']):raise ValueError('runtime_or_device_changed')
    audit=json.loads((folder/'label_audit.json').read_text());validate_labels(audit,groups)
    snap=report['snapshots'][0]
    if any(not math.isfinite(r['loss']) or r['supervised_tokens']<=0 for r in snap['losses']):
        raise ValueError('invalid_baseline_loss')
    weighted=sum(r['loss']*r['supervised_tokens'] for r in snap['losses'])/sum(r['supervised_tokens'] for r in snap['losses'])
    mean=sum(r['loss'] for r in snap['losses'])/8
    if (not math.isclose(weighted,snap['token_weighted_development_loss'],abs_tol=1e-12,rel_tol=0)
            or not math.isclose(mean,snap['mean_example_development_loss'],abs_tol=1e-12,rel_tol=0)):
        raise ValueError('loss_aggregation_changed')
    prior_path=previous_download/'teacher_pair_base_v1/evaluation.json'
    old=json.loads(prior_path.read_text());prior=old['snapshots'][0]
    old_summary=json.loads((previous_review/'results/summary.json').read_text())
    old_receipt=json.loads((previous_review/'results/receipt.json').read_text())
    if old_receipt['download_sha256']['teacher_pair_base_v1/evaluation.json']!=file_sha(prior_path):
        raise ValueError('previous_review_baseline_binding_changed')
    assessment_path=previous_review/'assessments.private.jsonl'
    if old_receipt['assessment_sha256']!=file_sha(assessment_path):raise ValueError('prior_assessments_changed')
    joined=reuse_assessments(snap['generations'],prior['generations'],rows(assessment_path),groups,config)
    per_output=[r|{'source_metrics':metrics(r['source'],r['text']),
        'reference_metrics':metrics(r['reference'],r['text']),'assessment_reused_from_v1':True} for r in joined]
    summary={'protocol':PROTOCOL,'actual_slug':'oliverlds/teacher-pair-expanded-base-4b',
        'requested_version':1,'execution_complete':True,'training_called':False,'optimizer_steps':0,
        'elapsed_seconds':state['elapsed_seconds'],'model_id':runtime.MODEL_ID,'model_revision':runtime.REVISION,
        'gpu':report['runtime']['gpu'],'gpu_count':1,'label_audit_rows':54,
        'max_full_tokens':max(r['full_tokens'] for r in audit),'max_prompt_tokens':max(r['masked_tokens'] for r in audit),
        'max_generation_context_tokens':max(r['masked_tokens'] for r in audit)+runtime.DECODING['max_new_tokens'],
        'token_weighted_development_loss':weighted,'mean_example_development_loss':mean,
        'exact_prior_generation_matches':10,'prior_loss_rows_identical':snap['losses']==prior['losses'],
        'prior_runtime_packages_identical':report['runtime']['packages']==old['runtime']['packages'],
        'reused_text_bound_assessments':10,'new_semantic_assessments':0,'splits':{},
        'new_model_calls_during_review':0,'training_authorized':False,'gpu_jobs_submitted_during_review':0,
        'limits':['Identical evidence reuses the v1 automated unblinded judgments; this is not independent replication of semantic assessment.',
            'Eight previously exposed development cases from four papers/authors; training probes reported separately.',
            'Baseline reproducibility and successful token checks do not establish a style gain, Fama signature, human authorship or detector improvement.']}
    for split in ('development','training'):
        group=[r for r in per_output if r['split']==split]
        summary['splits'][split]={'outputs':len(group),'empty':sum(r['empty'] for r in group),
            'output_limited':sum(r['output_limited'] for r in group),'exact_source_copies':sum(r['copied_source'] for r in group),
            'material_claim_issue_identified':sum(r['assessment']['material_claim_issue_identified'] for r in group),
            'median_source_word_sequence_overlap':statistics.median(r['source_metrics']['word_sequence_overlap'] for r in group),
            'median_reference_word_sequence_overlap':statistics.median(r['reference_metrics']['word_sequence_overlap'] for r in group)}
        if summary['splits'][split]['material_claim_issue_identified']!=old_summary['splits'][split]['material_claim_issue_identified']:
            raise ValueError('reused_assessment_aggregation_changed')
    output.mkdir(parents=True);write_rows(output/'per_output.private.jsonl',per_output);write_json(output/'summary.json',summary)
    write_json(output/'receipt.json',{'download_sha256':{str(p.relative_to(download)):file_sha(p) for p in sorted(download.rglob('*')) if p.is_file()},
        'packet_manifest_sha256':file_sha(packet/'manifest.json'),'prior_baseline_sha256':file_sha(prior_path),
        'prior_review_manifest_sha256':file_sha(previous_review/'results/manifest.json'),
        'reused_assessment_sha256':file_sha(assessment_path)})
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'code_sha256':file_sha(Path(__file__)),
        'files':{p.name:file_sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','download','previous-download','previous-review','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.packet,args.download,args.previous_download,args.previous_review,args.output)))


if __name__=='__main__':main()
