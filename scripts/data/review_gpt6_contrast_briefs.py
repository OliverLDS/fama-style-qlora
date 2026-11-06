"""Replay brief clarifications and export Stage 2 with briefs only; CPU, no APIs."""
import argparse
from collections import Counter
import copy
import importlib.util
import json
from pathlib import Path

SPEC=importlib.util.spec_from_file_location('contrast_packet',Path(__file__).with_name('prepare_gpt6_contrast_rewrites.py'))
m=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)
AUDIT=m.OUT/'brief_quality_v1'
STAGE2=m.OUT/'stage_2_reconstruction_batches'


def row_hash(row):
    return m.sha(json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(',',':')))


def apply_edits(row,edits):
    result=copy.deepcopy(row)
    for edit in edits:
        keys=edit['path']
        if not keys or keys[0] not in {'brief','status','notes'} or not edit.get('reason'):
            raise ValueError('unsupported_brief_edit')
        container=result
        for key in keys[:-1]:container=container[key]
        key=keys[-1]
        if edit.get('operation','replace')=='append':
            if not isinstance(container,list) or key!=len(container) or edit['before'] is not None:
                raise ValueError('append_binding')
            container.append(edit['after'])
        else:
            if container[key]!=edit['before']:raise ValueError('brief_edit_binding')
            container[key]=edit['after']
    return result


def load_raw():
    m.check();selected={r['id']:r for r in m.lines(m.OUT/'selected_targets.private.jsonl')}
    rows=[];files={}
    for job in m.read(m.OUT/'stage_1_brief_batches/jobs.json'):
        name='responses/stage_1_briefs/'+job['response'];rs=m.lines(m.OUT/name)
        if len(rs)!=len(job['ids']):raise ValueError('brief_count')
        files[name]=m.digest(m.OUT/name)
        for row,rid in zip(rs,job['ids']):
            if row.get('id')!=rid:raise ValueError('brief_order')
            m.validate_brief(row,selected[rid]);rows.append(row)
    if len(rows)!=len(selected) or len({r['id'] for r in rows})!=len(rows):raise ValueError('brief_coverage')
    return rows,files,selected


def prepare():
    if STAGE2.exists() or (AUDIT/'manifest.json').exists():raise ValueError('refuse_overwrite_brief_release')
    raw,files,selected=load_raw()
    decisions=m.lines(AUDIT/'adjudications.private.jsonl');by={r['id']:r for r in decisions}
    if len(by)!=len(decisions) or set(by)!={r['id'] for r in raw}:raise ValueError('brief_decision_coverage')
    reviewed=[];payload=[];alerts=[]
    for row in raw:
        d=by[row['id']]
        if d['raw_response_sha256']!=row_hash(row) or d['target_sha256']!=row['target_sha256'] or not d['reason']:
            raise ValueError('brief_decision_binding')
        final=apply_edits(row,d['edits']);m.validate_brief(final,selected[row['id']])
        text=' '.join([final['brief']['task']]+final['brief']['key_points']+final['brief']['anchors'])
        c=m.contrast(text,selected[row['id']]['target'])
        if c['longest_ordinary_words']>=12:alerts.append(dict(id=row['id'],flag='long_copied_ordinary_span',words=c['longest_ordinary_words']))
        if final['status']=='source_problem':alerts.append(dict(id=row['id'],flag='source_concern_carried',notes=final['notes']))
        reviewed.append(dict(id=row['id'],raw_response=row,effective_response=final,adjudication=d,
            raw_response_sha256=row_hash(row),effective_response_sha256=row_hash(final),
            metrics=c,independent_semantic_verification=False))
        # This allowlist is deliberately independent of the full original audit row.
        payload.append(dict(id=final['id'],target_sha256=final['target_sha256'],brief=final['brief'],
                            brief_status=final['status'],brief_notes=final['notes']))
    jobs=m.write_jobs(STAGE2,payload,'reconstruction')
    (AUDIT/'reviewed_briefs.private.jsonl').write_text(m.jsonl(reviewed))
    summary=dict(protocol='gpt6_contrast_brief_quality_v1',brief_files=len(files),brief_records=len(raw),
        raw_status_counts=dict(Counter(r['status'] for r in raw)),effective_status_counts=dict(Counter(r['effective_response']['status'] for r in reviewed)),
        clarified_ids=[r['id'] for r in reviewed if r['adjudication']['edits']],
        edit_count=sum(len(r['adjudication']['edits']) for r in reviewed),
        carried_source_concern_ids=[r['id'] for r in reviewed if r['effective_response']['status']=='source_problem'],
        long_ordinary_phrase_flags=sum(a['flag']=='long_copied_ordinary_span' for a in alerts),
        stage_2_records=len(payload),stage_2_batches=len(jobs),batch_sizes=[j['records'] for j in jobs],
        original_prose_fields_forwarded=0,old_rewrite_fields_forwarded=0,source_files_modified=0,
        provider_calls=0,training_calls=0,independent_semantic_verification=False)
    m.dump(AUDIT/'summary.json',summary);m.dump(AUDIT/'advisory_flags.json',alerts)
    m.dump(STAGE2/'manifest.json',dict(parent_preparation_manifest_sha256=m.digest(m.OUT/'manifest.json'),
        reviewed_briefs_sha256=m.digest(AUDIT/'reviewed_briefs.private.jsonl'),raw_response_files=files,
        builder_sha256=m.digest(Path(__file__)),
        files={str(f.relative_to(STAGE2)):m.digest(f) for f in sorted(STAGE2.iterdir()) if f.is_file()}))
    (STAGE2/'START_HERE.md').write_text('''# Stage 2: fresh paragraphs from content briefs

Upload only batch_01.md through batch_06.md, one file per GPT-6 request, and request
High thinking effort. Ask: "Complete the attached job and return the named JSONL file."
Each attachment already contains the instructions and briefs. No extra files are needed.

Prefer a new chat without attaching the original paragraphs, Stage 1 attachments,
old rewrites or taxonomy data. Cross-chat memory cannot be controlled or certified.
The package explicitly supplies only briefs, IDs, hash strings and uncertainty notes.

Save rewrites_01.jsonl through rewrites_06.jsonl in ../responses/stage_2_rewrites/.
All 175 selected items are included: 30/30/30/30/30/25. P002 and P024 are in batch 1.
Two inherited source concerns are carried explicitly rather than silently guessed
or discarded; all selected targets remain intended for completion and assessment.

The objective is fresh neutral expression, not a close edit or recovery of the
original author wording. Modest emphasis/nuance differences are allowed. There is
no required percentage of changed words and no request for taxonomy annotation.
After reconstruction, contrast and material factual issues will be assessed separately.
No GPU training, API calls or detector scans are requested by this packet.
''')
    m.dump(AUDIT/'manifest.json',dict(preparation_manifest_sha256=m.digest(m.OUT/'manifest.json'),
        raw_response_files=files,builder_sha256=m.digest(Path(__file__)),
        files={str(f.relative_to(m.OUT)):m.digest(f) for folder in [AUDIT,STAGE2] for f in sorted(folder.rglob('*')) if f.is_file()}))
    print(json.dumps(summary,indent=2))


def check():
    m.check();raw,files,selected=load_raw();manifest=m.read(AUDIT/'manifest.json')
    if files!=manifest['raw_response_files'] or m.digest(Path(__file__))!=manifest['builder_sha256']:
        raise ValueError('brief_sources_or_builder_changed')
    if m.digest(m.OUT/'manifest.json')!=manifest['preparation_manifest_sha256']:raise ValueError('preparation_changed')
    for name,h in manifest['files'].items():
        if m.digest(m.OUT/name)!=h:raise ValueError('brief_release_changed')
    reviewed=m.lines(AUDIT/'reviewed_briefs.private.jsonl');expected={r['id']:r['effective_response'] for r in reviewed};ids=[]
    for item in reviewed:
        if apply_edits(item['raw_response'],item['adjudication']['edits'])!=item['effective_response']:raise ValueError('brief_edit_replay')
    for job in m.read(STAGE2/'jobs.json'):
        rows=m.lines(STAGE2/f"inputs_{job['batch']:02d}.jsonl")
        if [r['id'] for r in rows]!=job['ids']:raise ValueError('reconstruction_order')
        text=(STAGE2/job['attachment']).read_text()
        for row in rows:
            final=expected[row['id']]
            if set(row)!={'id','target_sha256','brief','brief_status','brief_notes'} or row['brief']!=final['brief'] or row['brief_status']!=final['status'] or row['brief_notes']!=final['notes'] or row['target_sha256']!=final['target_sha256']:
                raise ValueError('reconstruction_payload')
            if selected[row['id']]['target'] in text:raise ValueError('whole_original_leaked')
            ids.append(row['id'])
    if len(ids)!=len(selected) or set(ids)!=set(selected):raise ValueError('reconstruction_coverage')
    return dict(records=len(ids),batches=len(m.read(STAGE2/'jobs.json')),original_prose_fields_forwarded=0,source_files_modified=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','check']);args=parser.parse_args()
    if args.action=='prepare':prepare()
    else:print(json.dumps(check(),indent=2))
