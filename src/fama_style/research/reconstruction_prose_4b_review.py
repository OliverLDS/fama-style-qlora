"""Review saved matched 4B prose; no generation, training or text repair."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from fama_style.evaluation.reconstruction_prose_4b import load_inputs,classify_output
from .author_control_replication import verify_files
from .neutral_reconstruction import write_json,write_rows
from .neutral_reconstruction_instruction import rows
from .neutral_reconstruction_results import file_sha,metrics
from .local_reconstruction_review import join_assessments

PROTOCOL='neutral_reconstruction_prose_gemma4b_review_v1'


def validate_outputs(outputs,jobs):
    by={r['job_id']:r for r in jobs}
    ids=[r['job_id'] for r in outputs]
    if len(by)!=9 or len(ids)!=9 or len(set(ids))!=9 or set(ids)!=set(by):
        raise ValueError('nine_job_coverage_required')
    result=[]
    for row in outputs:
        job=by[row['job_id']]
        if any(row[k]!=job[k] for k in ('record_id','method','prompt_sha256')):raise ValueError('job_binding_changed')
        if row['parsed']!=classify_output(row['raw_output']):raise ValueError('recorded_contract_flags_changed')
        result.append(row|{'text':row['raw_output']})
    return result


def build(packet,download,review,output):
    if output.exists():raise ValueError('refuse_overwrite_results')
    verify_files(packet)
    jobs,config=load_inputs(packet/'upload/reconstruction_prose_4b_jobs.jsonl',packet/'upload/reconstruction_prose_4b_config.json')
    folder=download/'reconstruction_prose_4b_v1'
    metadata=json.loads((folder/'runtime_metadata.json').read_text());state=json.loads((folder/'state.json').read_text())
    if not state['execution_complete'] or state['completed_jobs']!=9 or state['training_called']:
        raise ValueError('not_completed_nine_job_inference')
    if metadata['config']!=config or metadata['model_revision']!=config['model_revision'] or metadata['training_called']:
        raise ValueError('runtime_config_or_revision_changed')
    expected_code=json.loads((packet/'lineage.json').read_text())['runtime_sha256']
    if metadata['runtime_source_sha256']!=expected_code or file_sha(download/'reconstruction_prose_4b.py')!=expected_code:
        raise ValueError('runtime_source_changed')
    outputs=validate_outputs(rows(folder/'generations.private.jsonl'),jobs)
    joined=join_assessments(outputs,rows(review/'assessments.private.jsonl'))
    references={r['record_id']:r['original_paragraph'] for r in rows(packet/'private_key/references.private.jsonl')}
    result=[];grouped=defaultdict(list)
    for row in joined:
        rid=row['record_id'];m=metrics(references[rid],row['text'])
        record=row|{'original_paragraph':references[rid],'metrics':m,'training_ready':False}
        result.append(record);grouped[row['method']].append(record)
    summary={'protocol':PROTOCOL,'status':'executed_reviewed','actual_slug':'oliverlds/reconstruction-prose-4b','user_reported_version':1,
        'model_id':metadata['model_id'],'model_revision':metadata['model_revision'],'completed_jobs':9,
        'execution_complete':True,'response_contract_complete':state['response_contract_complete'],
        'elapsed_seconds':state['elapsed_seconds'],'peak_gpu_allocated_bytes':state['peak_gpu_allocated_bytes'],
        'methods':{},'new_model_calls':0,'training_called':False,'reserved_content_used':False,
        'limits':['Three failure-selected training-side cases; unblinded automated qualitative review, not independent semantic gold.',
                  'No Fama-style, human-authorship, detector or post-training benefit is established.',
                  'Teacher-side data-construction ability is not a prerequisite for reverse-direction text-pair SFT.']}
    for method,group in sorted(grouped.items()):
        summary['methods'][method]={'outputs':len(group),'claim_or_coverage_flags':sum(bool(r['assessment']['flags']) for r in group),
            'median_word_sequence_overlap':statistics.median(r['metrics']['word_sequence_overlap'] for r in group),
            'median_prompt_tokens':statistics.median(r['input_tokens'] for r in group),
            'median_longest_shared_word_run':statistics.median(r['metrics']['longest_shared_word_run'] for r in group),
            'blocked':sum(r['parsed']['status']=='blocked' for r in group),'empty':sum(r['parsed']['status']=='empty' for r in group),
            'length_limited':sum(r['hit_output_limit'] for r in group),'format_flags':sum(bool(r['parsed']['format_flags']) for r in group),
            'outputs_with_missing_numeric_values':sum(bool(r['metrics']['missing_numeric_values']) for r in group),
            'outputs_with_added_numeric_values':sum(bool(r['metrics']['added_numeric_values']) for r in group)}
    summary['compact_lower_overlap_than_direct']=sum(
        next(r for r in grouped['compact_claims'] if r['record_id']==rid)['metrics']['word_sequence_overlap']<
        next(r for r in grouped['direct_rewrite'] if r['record_id']==rid)['metrics']['word_sequence_overlap'] for rid in references)
    output.mkdir(parents=True);write_rows(output/'per_output.private.jsonl',result);write_json(output/'summary.json',summary)
    write_json(output/'receipt.json',{'downloaded_files_sha256':{str(p.relative_to(download)):file_sha(p) for p in sorted(download.rglob('*')) if p.is_file()},
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
