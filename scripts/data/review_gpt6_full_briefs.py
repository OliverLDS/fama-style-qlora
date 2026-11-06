"""Assess the remaining 320 briefs and export reconstruction-only packets.

Raw responses and all earlier releases remain unchanged. CPU only; training split only.
"""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import statistics

SPEC = importlib.util.spec_from_file_location('full_preparation', Path(__file__).with_name('prepare_gpt6_full_reconstruction.py'))
f = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(f)
m = f.m
b = f.review.b
OUT = f.OUT / 'brief_quality_v1'
STAGE2 = f.OUT / 'stage_2_reconstruction_batches'


def edits_for(row):
    if row['id'] == 'P257':
        return [dict(path=['brief', 'key_points', 0],
                     before='Cumulative wealth ratios can replace compounded benchmark-minus-event differences to lessen interpretive distortion',
                     after='Ratios of event-firm and benchmark cumulative wealth can replace differences between those cumulative wealths to lessen interpretive distortion',
                     reason='The reference does not specify benchmark-minus-event ordering; remove the unsupported subtraction direction.')]
    if row['id'] == 'P495':
        return [dict(path=['status'], before='source_problem', after='ready',
                     reason='The reconstruction can use the stated date ranges and total without asserting how many annual observations they contain. The original six-year wording remains unresolved in provenance.'),
                dict(path=['notes'], before=['Source labels 1984–1990 as a six-year span; inclusive calendar years would number seven, so exact span is ambiguous.'],
                     after=['For reconstruction, use the reported date ranges and $532 billion total; do not add a count of annual observations.'],
                     reason='Carry an actionable scope limitation rather than asking the realizer to invent a calendar correction or write an extraction critique.')]
    return []


def load_raw():
    f.check()
    targets = {r['id']: r for r in m.lines(m.OUT / 'audit.private.jsonl')}
    rows, files = [], {}
    for job in m.read(f.OUT / 'stage_1_brief_batches/jobs.json'):
        path = f.OUT / 'responses/stage_1_briefs' / job['response']
        batch = m.lines(path)
        if [r.get('id') for r in batch] != job['ids']:
            raise ValueError('brief_batch_identity')
        files[str(path.relative_to(f.OUT))] = m.digest(path)
        for row in batch:
            m.validate_brief(row, targets[row['id']])
        rows.extend(batch)
    queued = {r['id'] for r in m.lines(f.OUT / 'coverage.private.jsonl') if r['stage'] == 'brief_queued'}
    if len(rows) != len(queued) or {r['id'] for r in rows} != queued:
        raise ValueError('brief_coverage')
    return rows, files, targets


def payload_for(row):
    # A deliberate allowlist: never forward the audit row or original prose.
    return dict(id=row['id'], target_sha256=row['target_sha256'], brief=row['brief'],
                brief_status=row['status'], brief_notes=row['notes'])


def validate_payload(payload, effective):
    if payload != payload_for(effective) or set(payload) != {'id', 'target_sha256', 'brief', 'brief_status', 'brief_notes'}:
        raise ValueError('stage_2_payload')


def prepare():
    if OUT.exists() or STAGE2.exists():
        raise ValueError('refuse_overwrite_brief_review')
    rows, files, targets = load_raw()
    reviewed, decisions, payloads = [], [], []
    for row in rows:
        edits = edits_for(row)
        effective = b.apply_edits(row, edits)
        m.validate_brief(effective, targets[row['id']])
        text = ' '.join([effective['brief']['task']] + effective['brief']['key_points'] + effective['brief']['anchors'])
        metrics = m.contrast(text, targets[row['id']]['target'])
        d = dict(id=row['id'], target_sha256=row['target_sha256'], raw_response_sha256=b.row_hash(row),
                 decision='clarify' if edits else 'retain', edits=edits,
                 reason='Replay the limited content/status clarification.' if edits else
                 'Compact neutral content notes; retain simplification and modest compression without restoring authorial framing.',
                 reviewer='Codex automated assessment', independent_semantic_verification=False)
        decisions.append(d)
        reviewed.append(dict(id=row['id'], raw_response=row, effective_response=effective, adjudication=d,
                             raw_response_sha256=b.row_hash(row), effective_response_sha256=b.row_hash(effective),
                             metrics=metrics, independent_semantic_verification=False))
        payloads.append(payload_for(effective))
    OUT.mkdir()
    jobs = m.write_jobs(STAGE2, payloads, 'reconstruction')
    (OUT / 'reviewed_briefs.private.jsonl').write_text(m.jsonl(reviewed))
    (OUT / 'adjudications.private.jsonl').write_text(m.jsonl(decisions))
    # Maintain all 500 identities, including the existing 175 and five target holds.
    coverage = m.lines(f.OUT / 'coverage.private.jsonl')
    for row in coverage:
        if row['stage'] == 'brief_queued':
            row['stage'] = 'brief_reviewed_reconstruction_pending'
    (OUT / 'coverage.private.jsonl').write_text(m.jsonl(coverage))
    ratios = [len((' '.join([r['effective_response']['brief']['task']] + r['effective_response']['brief']['key_points'] +
                           r['effective_response']['brief']['anchors'])).split()) / len(targets[r['id']]['target'].split()) for r in reviewed]
    summary = dict(records=len(rows), files=len(files), raw_status_counts=dict(Counter(r['status'] for r in rows)),
                   effective_status_counts=dict(Counter(r['effective_response']['status'] for r in reviewed)),
                   retained_without_edits=sum(not r['edits'] for r in decisions),
                   clarified_ids=[r['id'] for r in decisions if r['edits']], edit_count=sum(len(r['edits']) for r in decisions),
                   long_copied_ordinary_span_flags=sum(r['metrics']['longest_ordinary_words'] >= 12 for r in reviewed),
                   maximum_ordinary_span=max(r['metrics']['longest_ordinary_words'] for r in reviewed),
                   median_brief_to_target_word_ratio=statistics.median(ratios),
                   stage_2_records=len(payloads), stage_2_batches=len(jobs), batch_sizes=[j['records'] for j in jobs],
                   source_files_modified=0, original_prose_fields_forwarded=0, old_rewrite_fields_forwarded=0,
                   provider_calls=0, training_calls=0, training_ready=False,
                   cohort_target_count=500, collected_brief_count_including_reuse=495,
                   usable_reconstructions_already_collected=173, pending_reconstructions=len(payloads),
                   missing_target_repair_ids=sorted(f.TARGET_HOLDS), previous_source_concern_ids=['P134', 'P139'],
                   complete=False, independent_semantic_verification=False,
                   assessment_limits=[
                       'All briefs receive format, identity and copied-span checks and content-note assessment.',
                       'Target comparisons focus on numeric, compression and technical-relation risks; no exhaustive claim alignment or independent model verifier.',
                       'Low overlap of notes is not proof of high-contrast prose; assess Stage 2 outputs separately.',
                       'Minor compression/emphasis shifts are accepted. Original PDFs and historical assertions were not independently reverified.',
                       'P495 reconstruction avoids a year-count assertion; the target date/count wording itself has not been repaired.'])
    m.dump(OUT / 'summary.json', summary)
    (STAGE2 / 'START_HERE.md').write_text('''# Stage 2: remaining fresh reconstructions

Upload batch_01.md through batch_11.md from this folder, one job per request.
Each file includes the prompt and reviewed briefs. No other attachments are needed.
Use GPT-6 with requested High thinking effort. Ask it to complete the attached job
and return the named JSONL file. There are ten jobs of 30 and one of 20: all 320
returned briefs are included. No earlier 175-record job needs to be repeated.

Use a separate request/chat from Stage 1. Do not attach original paragraphs,
Stage 1 files, earlier rewrites or human targets. Cross-chat memory isolation is
not asserted or required; the supplied packet itself contains only briefs and IDs.

Save rewrites_01.jsonl through rewrites_11.jsonl in:
../responses/stage_2_rewrites/

The aim is newly composed neutral prose rather than recovery of original author
wording. Substantial expression changes matter; modest shifts in emphasis are
acceptable. Keep the central economic account and necessary factual anchors.
P257 has a limited subtraction-direction clarification. P495 can be reconstructed
using the stated dates/total without inferring a count of annual observations.

These jobs complete missing reconstructions for 320 targets, not the entire 500:
five targets await source repair, and two earlier reconstructions have concerns.
No training, provider calls or detector scans are started by this packet.
''')
    manifest = dict(builder_sha256=m.digest(Path(__file__)),
                    preparation_manifest_sha256=m.digest(f.OUT / 'manifest.json'), raw_response_files=files,
                    files={str(p.relative_to(f.OUT)): m.digest(p) for folder in [OUT, STAGE2]
                           for p in sorted(folder.rglob('*')) if p.is_file()})
    m.dump(OUT / 'manifest.json', manifest)
    return check()


def check():
    rows, files, targets = load_raw()
    manifest = m.read(OUT / 'manifest.json')
    if files != manifest['raw_response_files'] or m.digest(Path(__file__)) != manifest['builder_sha256']:
        raise ValueError('brief_sources_changed')
    if m.digest(f.OUT / 'manifest.json') != manifest['preparation_manifest_sha256']:
        raise ValueError('brief_parent_changed')
    for name, digest in manifest['files'].items():
        if m.digest(f.OUT / name) != digest:
            raise ValueError('brief_release_changed')
    raw = {r['id']: r for r in rows}
    effective = {}
    for r in m.lines(OUT / 'reviewed_briefs.private.jsonl'):
        if r['raw_response'] != raw[r['id']] or r['effective_response'] != b.apply_edits(raw[r['id']], r['adjudication']['edits']):
            raise ValueError('brief_edit_replay')
        effective[r['id']] = r['effective_response']
    ids = []
    for job in m.read(STAGE2 / 'jobs.json'):
        payloads = m.lines(STAGE2 / f"inputs_{job['batch']:02d}.jsonl")
        if [p['id'] for p in payloads] != job['ids']:
            raise ValueError('payload_order')
        packet = (STAGE2 / job['attachment']).read_text()
        for p in payloads:
            validate_payload(p, effective[p['id']])
            if targets[p['id']]['target'] in packet:
                raise ValueError('whole_target_leaked')
            ids.append(p['id'])
    if len(ids) != len(raw) or set(ids) != set(raw):
        raise ValueError('stage_2_coverage')
    return m.read(OUT / 'summary.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'check'])
    args = parser.parse_args()
    print(json.dumps(prepare() if args.action == 'prepare' else check(), indent=2))
