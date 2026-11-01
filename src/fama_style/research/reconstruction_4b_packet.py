"""Freeze three existing training-side cases for six Gemma 4B inference jobs."""
import argparse
import json
from pathlib import Path

from .author_control_replication import verify_files
from .neutral_reconstruction import write_json,write_rows,make_handoff
from .neutral_reconstruction_instruction import EXPECTED_IDS,rows
from .neutral_reconstruction_results import file_sha
from fama_style.evaluation.reconstruction_4b import MODEL_ID,PROTOCOL,DECODING,prompt_for,sha,load_inputs


def make_jobs(references, briefs):
    if len(references)!=3 or {r['record_id'] for r in references}!=set(briefs): raise ValueError('three_matched_cases_required')
    jobs=[]
    for i,ref in enumerate(sorted(references,key=lambda r:r['record_id'])):
        rid=ref['record_id']
        # Alternate condition order without carrying conversation state across jobs.
        methods=['direct_rewrite','neutral_reconstruction']
        if i%2: methods.reverse()
        for method in methods:
            source=ref['original_paragraph'] if method=='direct_rewrite' else json.dumps(
                {'record_id':rid,'content_brief':briefs[rid]},ensure_ascii=False,sort_keys=True)
            prompt=prompt_for(rid,method,source)
            jobs.append({'job_id':rid+'_'+method,'record_id':rid,'method':method,
                         'source':source,'prompt':prompt,'prompt_sha256':sha(prompt)})
    return jobs


def build(packet, instruction, repair, output):
    if output.exists(): raise ValueError('refuse_overwrite_packet')
    for path in (packet,instruction,repair): verify_files(path)
    refs=[r for r in rows(packet/'private_key/reviewed_references.private.jsonl') if r['record_id'] in EXPECTED_IDS]
    if len(refs)!=3: raise ValueError('three_expected_cases_missing')
    briefs={r['record_id']:r['content_brief'] for r in rows(instruction/'inputs_01.jsonl')}
    if set(briefs)!=EXPECTED_IDS: raise ValueError('frozen_selection_changed')
    for ref in refs:
        if briefs[ref['record_id']] != make_handoff(ref['brief'],ref['original_paragraph']): raise ValueError('base_handoff_changed')
    changes=rows(repair/'derived_briefs.private.jsonl')
    if len(changes)!=1 or changes[0]['record_id']!='NR_0c1c81ebfc38e817': raise ValueError('expected_single_repair')
    ref_by={r['record_id']:r for r in refs}
    for changed in changes:
        ref=ref_by[changed['record_id']]
        if changed['reference_sha256']!=sha(ref['original_paragraph']): raise ValueError('repaired_reference_changed')
        if changed['handoff']!=make_handoff(changed['brief'],ref['original_paragraph']): raise ValueError('repaired_handoff_changed')
        briefs[changed['record_id']]=changed['handoff']
    jobs=make_jobs(refs,briefs)
    output.mkdir(parents=True);(output/'upload').mkdir();(output/'private_key').mkdir()
    jobs_path=output/'upload/reconstruction_4b_jobs.jsonl';write_rows(jobs_path,jobs)
    config={'protocol':PROTOCOL,'model_id':MODEL_ID,'seed':1729,'decoding':DECODING,
        'quantization':'NF4 double quantization; FP32 compute','record_ids':sorted(EXPECTED_IDS),
        'jobs_sha256':sha(jobs_path.read_text()),'jobs':6,'cases':3,'training_called':False}
    write_json(output/'upload/reconstruction_4b_config.json',config)
    load_inputs(jobs_path,output/'upload/reconstruction_4b_config.json')
    write_rows(output/'private_key/references.private.jsonl',refs)
    summary={'protocol':PROTOCOL,'status':'prepared_not_executed','model_id':MODEL_ID,'cases':3,'jobs':6,
        'methods':['direct_rewrite','neutral_reconstruction'],'repaired_briefs':1,
        'training_side_only':True,'reserved_content_used':False,'training_jobs':0,'gpu_jobs_submitted':0,
        'reference_policy':'Original paragraph only in direct inputs; brief-only jobs contain no original paragraph or metadata.',
        'prompt_policy':'Shared neutral-prose, claim-preservation, guardrail and JSON contract; method-specific input instructions.',
        'relation_to_gpt6':'Same three development cases; repaired brief and single-record model prompts mean no exact cross-model causal comparison.',
        'limits':'Failure-selected examples, not held-out generalization, human authorship or Fama-style evidence.',
        'evaluation':'Review claims, commentary, readability, organization and same overlap diagnostics against private references after downloading outputs.'}
    write_json(output/'summary.json',summary)
    write_json(output/'lineage.json',{'parent_manifests':{
        'paired_packet':file_sha(packet/'manifest.json'),'instruction_check':file_sha(instruction/'manifest.json'),
        'binding_repair':file_sha(repair/'manifest.json')},
        'runtime_sha256':file_sha(Path(__file__).parents[1]/'evaluation/reconstruction_4b.py')})
    (output/'START_HERE.md').write_text('''# Gemma 4B six-output inference check

Prepared only; no model run has been submitted.
Import kaggle/reconstruction_4b.ipynb as a NEW PRIVATE Kaggle notebook.
Create a PRIVATE Kaggle Dataset titled "Neutral Reconstruction Gemma 4B Pilot"
with exactly the two files in upload/. Attach only that Dataset to the notebook.
Keep private_key/ and previous response folders outside the uploaded Dataset.

Enable HF_TOKEN in Kaggle Secrets, select GPU T4 x2 and enable Internet.
The first cell exposes only GPU 0. Use a fresh session and Save & Run All.
The Hugging Face account must already have access to google/gemma-3-4b-it.
No secret value is pasted into notebook code or dataset files.

Download the entire /kaggle/working/reconstruction_4b_v1 output folder:
state.json, runtime_metadata.json, token_audit.json, generation_config.json,
and generations.private.jsonl. Keep outputs private. Share notebook slug/version
for retrieval and automated comparison; do not share credentials.
No adapter, optimizer, training, detector query or reserved example is involved.
''')
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'code_sha256':file_sha(Path(__file__)),
        'files':{str(p.relative_to(output)):file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','instruction','repair','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.packet,args.instruction,args.repair,args.output)))


if __name__=='__main__':main()
