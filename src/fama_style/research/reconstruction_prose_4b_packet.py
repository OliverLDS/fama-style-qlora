"""Freeze the existing nine local prose prompts for matched 4B inference."""
import argparse
import json
from pathlib import Path

from .author_control_replication import verify_files
from .neutral_reconstruction import write_json
from .neutral_reconstruction_instruction import EXPECTED_IDS,rows
from .neutral_reconstruction_results import file_sha
from fama_style.evaluation.reconstruction_prose_4b import MODEL_ID,MODEL_REVISION,PROTOCOL,DECODING,load_inputs,sha


def build(local_packet,output):
    if output.exists():raise ValueError('refuse_overwrite_packet')
    verify_files(local_packet)
    jobs=rows(local_packet/'jobs.private.jsonl')
    if {r['record_id'] for r in jobs}!=EXPECTED_IDS:raise ValueError('expected_three_exposed_cases')
    output.mkdir(parents=True);(output/'upload').mkdir();(output/'private_key').mkdir()
    target=output/'upload/reconstruction_prose_4b_jobs.jsonl'
    target.write_bytes((local_packet/'jobs.private.jsonl').read_bytes())
    config={'protocol':PROTOCOL,'model_id':MODEL_ID,'model_revision':MODEL_REVISION,'seed':1729,
        'decoding':DECODING,'quantization':'NF4 double quantization; FP32 compute',
        'record_ids':sorted(EXPECTED_IDS),'jobs_sha256':sha(target.read_text()),'jobs':9,'cases':3,
        'training_called':False}
    write_json(output/'upload/reconstruction_prose_4b_config.json',config)
    load_inputs(target,output/'upload/reconstruction_prose_4b_config.json')
    (output/'private_key/references.private.jsonl').write_bytes((local_packet/'references.private.jsonl').read_bytes())
    summary={'protocol':PROTOCOL,'status':'prepared_not_executed','model_id':MODEL_ID,'model_revision':MODEL_REVISION,
        'jobs':9,'cases':3,'methods':['direct_rewrite','detailed_brief','compact_claims'],
        'prompts_byte_identical_to_local':True,'training_side_only':True,'reserved_content_used':False,
        'training_jobs':0,'gpu_jobs_submitted':0,
        'comparison':'Within-model comparison of three input representations under identical prose instruction and decoding.',
        'format_policy':'Save raw prose; attach job metadata in Python, not through model-generated JSON.',
        'execution_policy':'All outputs retained. Execution completion does not imply contract compliance or semantic quality.',
        'limits':'Same three failure-selected cases; cross-model differences include engine, quantization and native chat template. No post-training or Fama-style claim.'}
    write_json(output/'summary.json',summary)
    write_json(output/'lineage.json',{'local_packet_manifest_sha256':file_sha(local_packet/'manifest.json'),
        'local_jobs_sha256':file_sha(local_packet/'jobs.private.jsonl'),
        'runtime_sha256':file_sha(Path(__file__).parents[1]/'evaluation/reconstruction_prose_4b.py')})
    (output/'START_HERE.md').write_text('''# Matched Gemma 4B prose comparison

Prepared only. Import kaggle/reconstruction_prose_4b.ipynb as a NEW PRIVATE notebook.
Create a PRIVATE Dataset titled "Reconstruction Prose 4B Comparison" containing
exactly the two JSON files in upload/. Attach that Dataset to the notebook.
Keep private_key/, raw responses and all other packages out of the Dataset.

Enable HF_TOKEN in Kaggle Secrets, Internet and GPU T4 x2. Only GPU 0 is used.
Start a fresh session, then Save & Run All. No adapter or training is involved.
Download the full /kaggle/working/reconstruction_prose_4b_v1 output directory,
including generations.private.jsonl, state.json, runtime_metadata.json,
token_audit.json and generation_config.json. Keep these outputs private.
Share the actual notebook URL slug and version afterward, without credentials.

All nine first outputs are retained. A successful notebook status means the
requests executed, not that claims or style passed. Format, empty, blocked and
length-limit counts are reported separately; no retries or text repair occur.
''')
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'code_sha256':file_sha(Path(__file__)),
        'files':{str(p.relative_to(output)):file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-packet',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(build(args.local_packet,args.output)))


if __name__=='__main__':main()
