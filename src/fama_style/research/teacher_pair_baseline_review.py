"""Review frozen reverse-direction baseline outputs without model calls or training."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics

from fama_style.training.teacher_pair_pilot import load_inputs,validate_baseline,HYPER,sha
from .author_control_replication import verify_files
from .neutral_reconstruction import write_json,write_rows
from .neutral_reconstruction_instruction import rows
from .neutral_reconstruction_results import file_sha,metrics
from .local_reconstruction_review import join_assessments

PROTOCOL='teacher_pair_baseline_review_v1'


def validate_labels(audit,groups):
    expected={r['example_id']:r['split'] for group in groups.values() for r in group}
    if len(audit)!=24 or len({r['example_id'] for r in audit})!=24 or {r['example_id'] for r in audit}!=set(expected):
        raise ValueError('24_label_audit_rows_required')
    for row in audit:
        if (row['split']!=expected[row['example_id']] or row['labels_match_target'] is not True
                or not 0<row['masked_tokens']<row['full_tokens']<=HYPER['max_length']
                or row['full_tokens']!=row['masked_tokens']+row['supervised_tokens']):
            raise ValueError('label_boundary_or_target_audit_failed')


def bind_generations(generations,groups,config):
    expected=config['development_ids']+config['training_probe_ids']
    if [r['example_id'] for r in generations]!=expected:raise ValueError('expected_eight_development_two_probes')
    by={r['example_id']:r for group in groups.values() for r in group}
    for row in generations:
        original=by[row['example_id']]
        if (row['source']!=original['source'] or row['reference']!=original['target']
                or row['split']!=original['split'] or row['prompt_sha256']!=sha(original['prompt'])):
            raise ValueError('generation_source_or_reference_binding_changed')
    return [r|{'job_id':r['example_id'],'text':r['output']} for r in generations]


def build(packet,download,review,output):
    if output.exists():raise ValueError('refuse_overwrite_review')
    verify_files(packet)
    groups,config=load_inputs(packet/'upload/teacher_pair_pilot.jsonl',packet/'upload/teacher_pair_config.json')
    folder=download/'teacher_pair_base_v1'
    report=validate_baseline(folder/'evaluation.json',config)
    state=json.loads((folder/'state.json').read_text())
    if not state['complete'] or state['training_called'] or state['optimizer_steps']!=0:raise ValueError('not_zero_update_baseline')
    runtime_sha=json.loads((packet/'lineage.json').read_text())['runtime_source_sha256']
    if report['runtime']['runtime_source_sha256']!=runtime_sha or file_sha(download/'teacher_pair_pilot.py')!=runtime_sha:
        raise ValueError('frozen_runtime_mismatch')
    audit=json.loads((folder/'label_audit.json').read_text());validate_labels(audit,groups)
    snap=report['snapshots'][0]
    if any(not math.isfinite(r['loss']) for r in snap['losses']):raise ValueError('nonfinite_baseline_loss')
    recomputed=sum(r['loss']*r['supervised_tokens'] for r in snap['losses'])/sum(r['supervised_tokens'] for r in snap['losses'])
    if not math.isclose(recomputed,snap['token_weighted_development_loss'],abs_tol=1e-12):raise ValueError('loss_aggregation_mismatch')
    generations=bind_generations(snap['generations'],groups,config)
    joined=join_assessments(generations,rows(review/'assessments.private.jsonl'))
    results=[]
    for row in joined:
        source_metrics=metrics(row['source'],row['text']);reference_metrics=metrics(row['reference'],row['text'])
        results.append(row|{'source_metrics':source_metrics,'reference_metrics':reference_metrics,'training_ready':False})
    summary={'protocol':PROTOCOL,'actual_slug':'oliverlds/teacher-pair-base-4b','user_reported_version':1,
        'model_id':report['runtime']['model_id'],'model_revision':report['runtime']['model_revision'],
        'execution_complete':True,'training_called':False,'optimizer_steps':0,'elapsed_seconds':state['elapsed_seconds'],
        'token_weighted_development_loss':snap['token_weighted_development_loss'],
        'mean_example_development_loss':snap['mean_example_development_loss'],
        'label_audit_rows':24,'max_full_tokens':max(r['full_tokens'] for r in audit),'splits':{},
        'new_model_calls':0,'training_authorized':False,'limits':[
            'Automated unblinded qualitative judgments; severity is not independently calibrated.',
            'Development papers/authors are held out from updates but already project-exposed.',
            'No adapter, style benefit, Fama-specific learning or detector outcome is established.']}
    for split in ('development','training'):
        group=[r for r in results if r['split']==split]
        summary['splits'][split]={'outputs':len(group),'empty':sum(r['empty'] for r in group),
            'output_limited':sum(r['output_limited'] for r in group),'exact_source_copies':sum(r['copied_source'] for r in group),
            'material_claim_issue_identified':sum(r['assessment']['material_claim_issue_identified'] for r in group),
            'outputs_with_any_review_flag':sum(bool(r['assessment']['flags']) for r in group),
            'median_source_word_sequence_overlap':statistics.median(r['source_metrics']['word_sequence_overlap'] for r in group),
            'median_reference_word_sequence_overlap':statistics.median(r['reference_metrics']['word_sequence_overlap'] for r in group),
            'median_output_to_source_length_ratio':statistics.median(r['source_metrics']['length_ratio'] for r in group)}
    output.mkdir(parents=True);write_rows(output/'per_output.private.jsonl',results);write_json(output/'summary.json',summary)
    write_json(output/'receipt.json',{'download_sha256':{str(p.relative_to(download)):file_sha(p) for p in sorted(download.rglob('*')) if p.is_file()},
        'packet_manifest_sha256':file_sha(packet/'manifest.json'),'assessment_sha256':file_sha(review/'assessments.private.jsonl'),
        'metrics_code_sha256':file_sha(Path(__file__).with_name('neutral_reconstruction_results.py'))})
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'code_sha256':file_sha(Path(__file__)),
        'files':{p.name:file_sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','download','review','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();print(json.dumps(build(args.packet,args.download,args.review,args.output)))


if __name__=='__main__':main()
