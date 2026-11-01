"""Recover and assess saved Gemma outputs offline; never regenerate or fill fields."""
import argparse
import json
from pathlib import Path
from collections import Counter

from fama_style.evaluation.reconstruction_4b import load_inputs,response_object,parse_output,sha
from .author_control_replication import verify_files
from .neutral_reconstruction import write_json,write_rows
from .neutral_reconstruction_instruction import rows
from .neutral_reconstruction_results import file_sha,metrics

PROTOCOL='neutral_reconstruction_gemma4b_review_v1'


def recover(row,job):
    if any(row[k]!=job[k] for k in ('job_id','record_id','method','prompt_sha256')):
        raise ValueError('response_job_binding_changed')
    obj,transport=response_object(row['raw_output'])
    if not isinstance(obj,dict) or obj.get('record_id')!=job['record_id'] or obj.get('method')!=job['method']:
        raise ValueError('unrecoverable_response_identity')
    parsed=parse_output(row['raw_output'],job)
    text=obj.get('text')
    if not isinstance(text,str) or not text.strip():raise ValueError('no_assessable_saved_text')
    return {'job_id':job['job_id'],'record_id':job['record_id'],'method':job['method'],
        'raw_output':row['raw_output'],'original_parser_result':row['parsed'],
        'transport':transport,'normalized_parser_result':parsed,
        'schema_conforming':parsed['status']!='format_error',
        'missing_fields':sorted({'record_id','method','status','text','reason'}-set(obj)),
        'text':text,'text_sha256':sha(text),'hit_output_limit':row['hit_output_limit'],
        'output_tokens':row['output_tokens'],'training_ready':False}


def build(packet,download,review,output):
    if output.exists():raise ValueError('refuse_overwrite_review')
    verify_files(packet)
    jobs,config=load_inputs(packet/'upload/reconstruction_4b_jobs.jsonl',packet/'upload/reconstruction_4b_config.json')
    job_by={r['job_id']:r for r in jobs};folder=download/'reconstruction_4b_v1'
    metadata=json.loads((folder/'runtime_metadata.json').read_text());state=json.loads((folder/'state.json').read_text())
    if metadata['config']!=config or metadata['model_id']!=config['model_id'] or metadata['training_called']:
        raise ValueError('runtime_not_expected_inference')
    source_sha=json.loads((packet/'lineage.json').read_text())['runtime_sha256']
    if source_sha!=file_sha(download/'reconstruction_4b.py'):raise ValueError('downloaded_runtime_differs_from_frozen_v1')
    received=rows(folder/'generations.private.jsonl');ids=[r['job_id'] for r in received]
    if len(ids)!=len(set(ids)) or set(ids)!=set(job_by) or state['completed_jobs']!=6:raise ValueError('output_coverage')
    assessments=rows(review/'assessments.private.jsonl');assessed={a['job_id']:a for a in assessments}
    if len(assessed)!=len(assessments) or set(assessed)!=set(job_by):raise ValueError('assessment_coverage')
    references={r['record_id']:r['original_paragraph'] for r in rows(packet/'private_key/references.private.jsonl')}
    results=[]
    for raw in received:
        r=recover(raw,job_by[raw['job_id']]);a=assessed[r['job_id']]
        if a['text_sha256']!=r['text_sha256']:raise ValueError('assessment_text_binding_changed')
        r.update(original_paragraph=references[r['record_id']],assessment=a,
                 metrics=metrics(references[r['record_id']],r['text']))
        results.append(r)
    summary={'protocol':PROTOCOL,'requested_version':1,'actual_slug':'oliverlds/reconstruction-4b',
        'model_id':metadata['model_id'],'model_revision':metadata['model_revision'],
        'execution_completed_jobs':6,'original_kaggle_status':'failed_after_all_generations',
        'initial_strict_parser_accepted':0,'single_fence_json_objects_recovered':len(results),
        'schema_conforming_after_fence_only_normalization':sum(r['schema_conforming'] for r in results),
        'schema_nonconforming_missing_fields':sum(bool(r['missing_fields']) for r in results),
        'truncated_outputs':sum(r['hit_output_limit'] for r in results),
        'elapsed_seconds':state['elapsed_seconds'],'peak_gpu_allocated_bytes':state['peak_gpu_allocated_bytes'],
        'material_issue_counts_by_method':dict(Counter(r['method'] for r in results if r['assessment']['material_claim_issue_identified'])),
        'raw_outputs_modified':False,'new_model_calls':0,'new_gpu_jobs':0,'training_called':False,
        'limits':['Six texts are assessable, but two remain schema-nonconforming and are not relabelled as valid generations.',
                  'Automated, unblinded qualitative review of three failure-selected development cases.',
                  'No Fama-style, human-authorship or detector outcome is established.']}
    output.mkdir(parents=True)
    write_rows(output/'per_output.private.jsonl',results);write_json(output/'summary.json',summary)
    write_json(output/'receipt.json',{'downloaded_files_sha256':{str(p.relative_to(download)):file_sha(p) for p in sorted(download.rglob('*')) if p.is_file()},
        'packet_manifest_sha256':file_sha(packet/'manifest.json'),
        'assessment_sha256':file_sha(review/'assessments.private.jsonl'),
        'parser_source_sha256':file_sha(Path(__file__).parents[1]/'evaluation/reconstruction_4b.py'),
        'normalization':'Remove exactly one complete JSON Markdown fence; no field filling, text editing or model call.'})
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'code_sha256':file_sha(Path(__file__)),
        'files':{p.name:file_sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','download','review','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();print(json.dumps(build(args.packet,args.download,args.review,args.output)))


if __name__=='__main__':main()
