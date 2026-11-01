"""CPU-only review of saved local generations with explicit qualitative judgments."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from .author_control_replication import verify_files
from .neutral_reconstruction import write_json,write_rows
from .neutral_reconstruction_instruction import rows
from .neutral_reconstruction_results import file_sha,metrics
from .local_reconstruction import METHODS

PROTOCOL='local_reconstruction_1b_review_v1'


def join_assessments(outputs,assessments):
    by={r['job_id']:r for r in assessments}
    if len(by)!=len(assessments) or len({r['job_id'] for r in outputs})!=len(outputs) or set(by)!={r['job_id'] for r in outputs}:
        raise ValueError('assessment_or_output_coverage')
    result=[]
    for row in outputs:
        if by[row['job_id']]['text_sha256']!=hashlib.sha256(row['text'].encode()).hexdigest():
            raise ValueError('assessment_text_changed')
        result.append(row|{'assessment':by[row['job_id']]})
    return result


def build(packet,run,review,output):
    if output.exists():raise ValueError('refuse_overwrite_results')
    verify_files(packet);verify_files(run)
    state=json.loads((run/'state.json').read_text())
    if not state['complete'] or state['completed_jobs']!=9:raise ValueError('run_incomplete')
    metadata=json.loads((run/'runtime_metadata.json').read_text())
    if metadata['packet_manifest_sha256']!=file_sha(packet/'manifest.json'):raise ValueError('wrong_packet')
    jobs={r['job_id']:r for r in rows(packet/'jobs.private.jsonl')}
    outputs=rows(run/'generations.private.jsonl')
    if len(outputs)!=9 or set(jobs)!={r['job_id'] for r in outputs}:raise ValueError('nine_frozen_jobs_required')
    joined=join_assessments(outputs,rows(review/'assessments.private.jsonl'))
    refs={r['record_id']:r['original_paragraph'] for r in rows(packet/'references.private.jsonl')}
    token_audit={r['job_id']:r for r in json.loads((run/'token_audit.json').read_text())}
    grouped=defaultdict(list);result=[]
    for row in joined:
        job=jobs[row['job_id']]
        if any(job[k]!=row[k] for k in ('record_id','method','prompt_sha256')):raise ValueError('job_binding_changed')
        record=row|{'original_paragraph':refs[row['record_id']],
                    'metrics':metrics(refs[row['record_id']],row['text']),'training_ready':False}
        result.append(record);grouped[row['method']].append(record)
    summary={'protocol':PROTOCOL,'generated_outputs':9,'model':metadata['model'],
        'model_file_sha256':metadata['model_file_sha256'],'server_build':metadata['build_info'],
        'context_per_slot':metadata['context_per_slot'],'elapsed_seconds':state['elapsed_seconds'],
        'actual_prompt_vs_preflight_differences':sorted({r['raw_response']['usage']['prompt_tokens']-token_audit[r['job_id']]['input_tokens_preflight'] for r in joined}),
        'methods':{},'training_called':False,'additional_generation_calls':0,
        'limits':['Three training-side failure-selected cases, unblinded automated qualitative judgments.',
                  'Not a Gemma 4B eligibility gate or estimate of generalization.',
                  'No Fama-style, human-authorship or detector performance is measured.']}
    for method in METHODS:
        group=grouped[method]
        summary['methods'][method]={'outputs':len(group),'blocked':sum(r['status']=='blocked' for r in group),
            'empty':sum(not r['text'].strip() for r in group),'length_limited':sum(r['finish_reason']=='length' for r in group),
            'material_claim_issue_identified':sum(r['assessment']['material_claim_issue_identified'] for r in group),
            'median_word_sequence_overlap':statistics.median(r['metrics']['word_sequence_overlap'] for r in group),
            'median_prompt_tokens':statistics.median(r['raw_response']['usage']['prompt_tokens'] for r in group),
            'outputs_with_missing_numeric_values':sum(bool(r['metrics']['missing_numeric_values']) for r in group),
            'outputs_with_added_numeric_values':sum(bool(r['metrics']['added_numeric_values']) for r in group)}
    output.mkdir(parents=True);write_rows(output/'per_output.private.jsonl',result);write_json(output/'summary.json',summary)
    write_json(output/'receipt.json',{'packet_manifest_sha256':file_sha(packet/'manifest.json'),
        'run_manifest_sha256':file_sha(run/'manifest.json'),'assessment_sha256':file_sha(review/'assessments.private.jsonl'),
        'metrics_code_sha256':file_sha(Path(__file__).with_name('neutral_reconstruction_results.py'))})
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'code_sha256':file_sha(Path(__file__)),
        'files':{p.name:file_sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','run','review','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();print(json.dumps(build(args.packet,args.run,args.review,args.output)))


if __name__=='__main__':main()
