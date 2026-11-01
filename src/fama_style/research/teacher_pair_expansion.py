"""Data-only, source-aligned expansion of the fixed teacher-pair experiment."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import statistics

from .author_control_replication import verify_files
from .neutral_reconstruction import write_json,write_rows
from .neutral_reconstruction_instruction import rows
from .neutral_reconstruction_results import file_sha,metrics
from fama_style.training.teacher_pair_pilot import sha,prompt

PROTOCOL='teacher_pair_curated_expansion_v2'
ADMIT='admit_limited_pilot'
HOLDS={'hold_ambiguous_condition','hold_unbound_context','hold_technical_meaning','hold_transcription_or_notation'}


def training_annotations(path):
    """Skip non-training annotation bodies before JSON parsing, including reserves."""
    result={}
    for line in path.read_text().splitlines():
        if not re.search(r'"split"\s*:\s*"training"',line):continue
        row=json.loads(line)
        if row['split']!='training':raise ValueError('training_annotation_selector_mismatch')
        if row['paragraph_id'] in result:raise ValueError('duplicate_upstream_training_id')
        result[row['paragraph_id']]=row['annotation']
    return result


def select(parent_rows,previous_rows,previous_decisions,new_decisions,annotations):
    by={r['example_id']:r for r in parent_rows}
    if len(by)!=len(parent_rows) or any(r['split']!='training' for r in parent_rows):raise ValueError('training_pool_only')
    old={r['example_id']:r for r in previous_rows if r['split']=='training'}
    dev=[r for r in previous_rows if r['split']=='development']
    held={rid for rid in by if previous_decisions[rid]['decision']=='hold_unresolved_upstream_flags'}
    if set(old)&held or set(old)|held!=set(by) or set(new_decisions)!=held:raise ValueError('complete_held_pool_decisions_required')
    selected=dict(old);audit=[]
    for rid,row in sorted(by.items()):
        if (row['source_sha256']!=sha(row['source']) or row['target_sha256']!=sha(row['target'])
                or type(row['publication_year']) is not int or row['publication_year']>=2020):
            raise ValueError('text_binding_or_date_changed')
        annotation=annotations[row['paragraph_id']]
        if annotation['plain_english']!=row['source'] or annotation['original_paragraph']!=row['target']:
            raise ValueError('upstream_text_binding_changed')
        if rid in old:
            if any(old[rid][key]!=row[key] for key in ('source','target','source_sha256','target_sha256','paper_id','authors')):
                raise ValueError('carried_pair_changed')
            decision={'decision':'carry_forward_v1','basis':'Previous versioned source-alignment review; pair kept unchanged.'}
        else:
            decision=new_decisions[rid]
            if decision['source_sha256']!=row['source_sha256'] or decision['target_sha256']!=row['target_sha256']:
                raise ValueError('new_decision_text_binding_changed')
            if decision['decision'] not in HOLDS|{ADMIT}:raise ValueError('unknown_expansion_decision')
            if not decision.get('alignment_notes') or not decision.get('context_disposition'):raise ValueError('explicit_alignment_and_context_notes_required')
            if decision['decision']==ADMIT:
                selected[rid]={key:row[key] for key in next(iter(old.values())) if key not in ('prompt','pilot_eligible')}
                selected[rid]['prompt']=prompt(row['source']);selected[rid]['pilot_eligible']=True
        audit.append({'example_id':rid,'paragraph_id':row['paragraph_id'],'paper_id':row['paper_id'],
            'source':row['source'],'target':row['target'],'source_sha256':row['source_sha256'],'target_sha256':row['target_sha256'],
            'decision':decision,'original_needs_review':row['needs_review'],'original_needs_external_context':row['needs_external_context'],
            'upstream_source_quality':annotation['source_quality'],'upstream_review_notes':annotation.get('review_notes',[]),
            'independent_semantic_verification':False,'full_scale_training_ready':False})
    train=[selected[rid] for rid in sorted(selected)]
    for field in ('paper_id','author_group_id'):
        if {r[field] for r in train}&{r[field] for r in dev}:raise ValueError('split_metadata_overlap')
    if {a.casefold().strip() for r in train for a in r['authors']}&{a.casefold().strip() for r in dev for a in r['authors']}:
        raise ValueError('author_overlap')
    seen=set()
    for row in train+dev:
        for text in (row['source'],row['target']):
            normalized=' '.join(text.casefold().split())
            if not normalized or normalized in seen:raise ValueError('duplicate_source_or_target')
            seen.add(normalized)
    return train,dev,audit


def describe(group):
    measurements=[metrics(r['source'],r['target']) for r in group]
    return {'pairs':len(group),'papers':len({r['paper_id'] for r in group}),
        'author_groups':len({r['author_group_id'] for r in group}),
        'median_source_reference_overlap':statistics.median(m['word_sequence_overlap'] for m in measurements),
        'min_overlap':min(m['word_sequence_overlap'] for m in measurements),'max_overlap':max(m['word_sequence_overlap'] for m in measurements),
        'pairs_with_sentence_count_proxy_change':sum(m['sentence_count_changed'] for m in measurements),
        'pairs_above_0_90_overlap':sum(m['word_sequence_overlap']>0.90 for m in measurements),
        'measurement_note':'Descriptive surface metrics only; no edit-distance selection threshold or proof of style contrast.'}


def build(parent,previous,upstream,decisions,output):
    if output.exists():raise ValueError('refuse_overwrite_expansion')
    for p in (parent,previous,upstream):verify_files(p)
    parent_manifest=json.loads((parent/'manifest.json').read_text())
    if parent_manifest['upstream_audit_manifest_sha256']!=file_sha(upstream/'manifest.json'):raise ValueError('wrong_upstream_audit')
    parent_rows=rows(parent/'datasets/train.provisional.private.jsonl')
    previous_path=previous/'upload/teacher_pair_pilot.jsonl';previous_rows=rows(previous_path)
    old_decisions=json.loads((previous/'private_key/decisions.private.json').read_text())['decisions']
    new=json.loads(decisions.read_text())
    if new['protocol']!=PROTOCOL:raise ValueError('wrong_expansion_protocol')
    train,dev,audit=select(parent_rows,previous_rows,old_decisions,new['decisions'],training_annotations(upstream/'audits.private.jsonl'))
    # Development PDF metadata comes only from the already selected eight audit rows.
    dev_ids={r['example_id'] for r in dev};dev_pdf=set();dev_audit_ids=set()
    for line in (previous/'private_key/selection_audit.private.jsonl').read_text().splitlines():
        if not any('"'+rid+'"' in line for rid in dev_ids):continue
        row=json.loads(line)
        if row['example_id'] in dev_ids:
            dev_pdf.add(row['pdf_sha256']);dev_audit_ids.add(row['example_id'])
    if dev_audit_ids!=dev_ids:raise ValueError('development_pdf_audit_coverage')
    parent_by={r['example_id']:r for r in parent_rows}
    if dev_pdf&{parent_by[r['example_id']]['pdf_sha256'] for r in train}:raise ValueError('pdf_crosses_splits')
    controls={r['example_id'] for r in rows(parent/'detector_controls/PRIVATE_KEY.jsonl')}
    for row in audit:row['prior_detector_control_metadata']=row['example_id'] in controls
    original_lines={json.loads(line)['example_id']:line for line in previous_path.read_bytes().splitlines(keepends=True)}
    output.mkdir(parents=True);(output/'private_key').mkdir()
    with (output/'training.private.jsonl').open('wb') as stream:
        for row in train:
            stream.write(original_lines[row['example_id']] if row['example_id'] in original_lines
                         else (json.dumps(row,sort_keys=True,ensure_ascii=False)+'\n').encode())
    development_bytes=b''.join(line for line in previous_path.read_bytes().splitlines(keepends=True)
                              if json.loads(line)['split']=='development')
    (output/'development.private.jsonl').write_bytes(development_bytes)
    write_rows(output/'private_key/training_audit.private.jsonl',audit)
    (output/'private_key/expansion_decisions.private.json').write_bytes(decisions.read_bytes())
    old_ids={r['example_id'] for r in previous_rows if r['split']=='training'}
    added=[r for r in train if r['example_id'] not in old_ids]
    summary={'protocol':PROTOCOL,'status':'data_curated_no_new_training_plan','parent_training_candidates':len(parent_rows),
        'previous_training_pairs':len(old_ids),'held_candidates_reviewed':len(new['decisions']),
        'newly_admitted':len(added),'remaining_held':len(parent_rows)-len(train),
        'new_decision_counts':dict(Counter(r['decision'] for r in new['decisions'].values())),
        'training':describe(train),'added_pairs':describe(added) if added else None,'development':describe(dev),
        'development_text_and_row_bytes_unchanged':True,'existing_training_row_bytes_unchanged':True,
        'added_pairs_with_previous_detector_control_exposure':sum(r['example_id'] in controls for r in added),
        'source_or_target_repairs':0,'new_annotations_or_paraphrases':0,'detector_scores_read':0,
        'reserved_records_deserialized_for_curation':0,'gpu_jobs_submitted':0,'training_authorized':False,
        'new_training_budget_selected':False,'tokenizer_preflight_for_added_rows':'not_run',
        'full_scale_training_ready':False,
        'review_limits':'Automated comparison to archived human transcription and upstream annotations; original flags retained; no new independent PDF/OCR or semantic certification.'}
    write_json(output/'summary.json',summary)
    old_config=json.loads((previous/'upload/teacher_pair_config.json').read_text())
    write_json(output/'lineage.json',{'parent_manifest_sha256':file_sha(parent/'manifest.json'),
        'previous_packet_manifest_sha256':file_sha(previous/'manifest.json'),'upstream_manifest_sha256':file_sha(upstream/'manifest.json'),
        'decision_sha256':file_sha(decisions),'development_sha256':sha(development_bytes.decode()),
        'preserved_development_ids':old_config['development_ids'],'preserved_training_probe_ids':old_config['training_probe_ids'],
        'generation_prompt_policy':'Same ordinary rewrite instruction as v1; no new author label, taxonomy or reference supplied to generator.'})
    (output/'README.md').write_text(f'''# Teacher-pair curation v2: data only

{len(train)} provisionally aligned training pairs; the same {len(dev)} development rows as v1.
No new training configuration, authorization or Kaggle run is created here.
The v1 notebook is frozen for exactly 16 training pairs; these v2 files are NOT
an upload replacement for its old Dataset. A separately specified bounded
follow-up must set the data/step comparison and tokenizer preflight first.
All text and original flags remain unchanged; per-row decisions are private_key/.
See docs/teacher_pair_pilot_v2/README.md for source-check scope and limitations.
''')
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'code_sha256':file_sha(Path(__file__)),
        'files':{str(p.relative_to(output)):file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','previous','upstream','decisions','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();print(json.dumps(build(args.parent,args.previous,args.upstream,args.decisions,args.output)))


if __name__=='__main__':main()
