"""Audited, unchanged teacher-pair selection for an isolated reverse-direction pilot."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from .author_control_replication import verify_files
from .neutral_reconstruction import write_json,write_rows
from .neutral_reconstruction_instruction import rows
from .neutral_reconstruction_results import file_sha,metrics
from fama_style.training.teacher_pair_pilot import PROTOCOL,MODEL_ID,REVISION,HYPER,DECODING,sha,prompt,load_inputs


def build(parent,decisions,output):
    if output.exists():raise ValueError('refuse_overwrite_selection')
    verify_files(parent)
    candidates=rows(parent/'datasets/train.provisional.private.jsonl')+rows(parent/'datasets/development.private.jsonl')
    judgments=json.loads(decisions.read_text())
    by={r['example_id']:r for r in candidates};selected=[];audit=[]
    if len(by)!=len(candidates) or set(judgments['decisions'])!=set(by):raise ValueError('all_candidate_decisions_required')
    for rid,row in sorted(by.items()):
        decision=judgments['decisions'][rid]
        if row['source_sha256']!=sha(row['source']) or row['target_sha256']!=sha(row['target']):raise ValueError('candidate_text_changed')
        if decision['source_sha256']!=row['source_sha256'] or decision['target_sha256']!=row['target_sha256']:
            raise ValueError('audit_text_binding_changed')
        if row['split'] not in ('training','development') or row['publication_year']>=2020:raise ValueError('split_or_date_violation')
        record={'example_id':rid,'paper_id':row['paper_id'],'split':row['split'],'decision':decision,
                'original_needs_review':row['needs_review'],'original_needs_external_context':row['needs_external_context'],
                'original_source_quality_status':row['source_quality_status'],
                'source':row['source'],'target':row['target'],'pdf_sha256':row['pdf_sha256'],
                'semantic_gold_verified':False,'full_scale_training_ready':False}
        audit.append(record)
        if decision['decision']!='approve_limited_pilot':continue
        if not decision.get('alignment_notes') or not decision.get('context_disposition'):
            raise ValueError('explicit_alignment_and_context_decision_required')
        selected.append({k:row[k] for k in ('example_id','paper_id','author_group_id','authors','split','source','target','source_sha256','target_sha256')}|
                        {'prompt':prompt(row['source']),'pilot_eligible':True})
    selected.sort(key=lambda r:(r['split']!='training',r['example_id']))
    train=[r for r in selected if r['split']=='training'];dev=[r for r in selected if r['split']=='development']
    train_pdf={by[r['example_id']]['pdf_sha256'] for r in train};dev_pdf={by[r['example_id']]['pdf_sha256'] for r in dev}
    if train_pdf&dev_pdf:raise ValueError('same_pdf_across_splits')
    probe=[];papers=set()
    for row in train:
        if row['paper_id'] not in papers:probe.append(row['example_id']);papers.add(row['paper_id'])
        if len(probe)==2:break
    output.mkdir(parents=True);(output/'upload').mkdir();(output/'private_key').mkdir()
    data=output/'upload/teacher_pair_pilot.jsonl';write_rows(data,selected)
    config={'protocol':PROTOCOL,'model_id':MODEL_ID,'model_revision':REVISION,'hyperparameters':HYPER,
        'decoding':DECODING,'quantization':'NF4 double quantization; FP32 compute','dataset_sha256':sha(data.read_text()),
        'training_ids':[r['example_id'] for r in train],'development_ids':[r['example_id'] for r in dev],
        'training_probe_ids':probe,'gpu_execution_authorized':False}
    write_json(output/'upload/teacher_pair_config.json',config);load_inputs(data,output/'upload/teacher_pair_config.json')
    write_rows(output/'private_key/selection_audit.private.jsonl',audit)
    (output/'private_key/decisions.private.json').write_bytes(decisions.read_bytes())
    def describe(group):
        overlap=[metrics(r['source'],r['target'])['word_sequence_overlap'] for r in group]
        return {'pairs':len(group),'papers':len({r['paper_id'] for r in group}),
                'author_groups':len({r['author_group_id'] for r in group}),
                'median_source_reference_overlap':statistics.median(overlap),'min_overlap':min(overlap),'max_overlap':max(overlap)}
    summary={'protocol':PROTOCOL,'status':'prepared_not_executed','candidate_denominator':len(candidates),
        'training':describe(train),'development':describe(dev),'not_selected':len(candidates)-len(selected),
        'decision_counts':dict(Counter(d['decision'] for d in judgments['decisions'].values())),
        'development_by_paper':dict(Counter(r['paper_id'] for r in dev)),
        'approved_with_original_review_flag':sum(by[r['example_id']]['needs_review'] for r in selected),
        'audit_scope':'Automated comparison to archived human paragraph and supplied teacher rewrite; no new independent PDF/OCR certification.',
        'source_or_target_repairs':0,'new_synthetic_inputs':0,'reserved_records_read':0,
        'paper_and_author_separated':True,'fresh_development_claim':False,'full_scale_training_ready':False,
        'baseline_outputs_planned':10,'pilot_snapshots':3,'pilot_outputs_planned':31,
        'optimizer_updates_planned':16,'training_execution_authorized':False,'gpu_jobs_submitted':0}
    write_json(output/'summary.json',summary)
    write_json(output/'lineage.json',{'parent_manifest_sha256':file_sha(parent/'manifest.json'),
        'decision_sha256':file_sha(decisions),'runtime_source_sha256':file_sha(Path(__file__).parents[1]/'training/teacher_pair_pilot.py')})
    (output/'START_HERE.md').write_text('''# Teacher-pair reverse pilot: baseline first

Prepared only; no training authorization or GPU submission.
Create a PRIVATE Dataset titled "Teacher Pair Reverse Pilot" containing exactly
the two files in upload/. Keep private_key/, raw responses and other datasets out.
Import kaggle/teacher_pair_base_4b.ipynb as a NEW PRIVATE notebook, attach the
Dataset, enable HF_TOKEN in Secrets and Internet, and select GPU T4 x2.
Only GPU 0 is used. The baseline notebook performs inference and loss evaluation.
Download teacher_pair_base_v1 in full or provide the actual URL slug and version.

Do not run the training notebook yet. It defaults to a CPU check and its
RUN_PILOT switch remains false. After baseline review and explicit training
authorization, the prepared pilot can run two epochs / sixteen updates against
the exact same dataset. It also requires the complete baseline output.
No full training run, detector scoring or taxonomy input is part of this pilot.
''')
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'code_sha256':file_sha(Path(__file__)),
        'files':{str(p.relative_to(output)):file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','decisions','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();print(json.dumps(build(args.parent,args.decisions,args.output)))


if __name__=='__main__':main()
