"""Prepare all 500 enhanced-brief / purpose-only reconstruction pairs.

Reuse five observed v4 briefs, collect the remaining 495, then reconstruct all 500.
No teacher calls, model imports, GPU jobs, target changes or response edits.
"""
import argparse
from collections import Counter
import copy
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from fama_style.research.content_brief_v4 import validate_brief_v4

SPEC = importlib.util.spec_from_file_location('observed_pilot', Path(__file__).with_name('compare_abstraction_reconstructions_v4.py'))
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)
m = pilot.m
OUT = ROOT / 'research_runs/data_scaling_v1/gpt6_purpose_only_0500_v1'
DOCS = ROOT / 'docs/data_scaling_v1'
TARGETS = ROOT / 'research_runs/data_scaling_v1/shared_target_repairs_v2/targets_0500.private.jsonl'
PILOT_BRIEFS = pilot.a.RAW
BATCH_SIZE = 30
RECONSTRUCTION_FIELDS = ('topic', 'purpose', 'key_points', 'anchors')


def purpose_only_payload(row):
    validate_brief_v4(row)
    return dict(id=row['id'], target_sha256=row['target_sha256'],
        brief={key: copy.deepcopy(row['brief'][key]) for key in RECONSTRUCTION_FIELDS},
        brief_status=row['status'], brief_notes=copy.deepcopy(row['notes']))


def partition(targets, reused):
    ids = [r['id'] for r in targets]
    if len(ids) != 500 or set(ids) != {f'P{i:03d}' for i in range(1, 501)}:
        raise ValueError('full_500_cohort_required')
    by_id = {r['id']: r for r in targets}
    if len({r['target_sha256'] for r in targets}) != 500 or any(r['split'] != 'training' or m.sha(r['target']) != r['target_sha256'] for r in targets):
        raise ValueError('training_target_binding')
    reused_ids = [r['id'] for r in reused]
    if len(reused_ids) != 5 or len(set(reused_ids)) != 5 or not set(reused_ids) <= set(ids):
        raise ValueError('five_pilot_briefs_required')
    for row in reused:
        validate_brief_v4(row)
        if row['target_sha256'] != by_id[row['id']]['target_sha256']:
            raise ValueError('pilot_target_binding')
    return [r for r in targets if r['id'] not in set(reused_ids)]


def write_jobs(folder, rows, phase):
    folder.mkdir()
    prompt_path = DOCS / ('BRIEF_ABSTRACTION_V4.md' if phase == 'briefs' else 'BRIEF_RECONSTRUCTION_V4.md')
    prompt = prompt_path.read_text()
    jobs = []
    for offset in range(0, len(rows), BATCH_SIZE):
        chunk = rows[offset:offset+BATCH_SIZE]
        number = len(jobs) + 1
        response = f'briefs_{number:02d}.jsonl' if phase == 'briefs' else f'rewrites_{number:02d}.jsonl'
        inputs = [dict(id=r['id'], target_sha256=r['target_sha256'], original_paragraph=r['target']) for r in chunk] if phase == 'briefs' else chunk
        data = f'inputs_{number:02d}.jsonl'
        attachment = f'batch_{number:02d}.md'
        (folder / data).write_text(m.jsonl(inputs))
        (folder / attachment).write_text(prompt + f'\nOutput data filename: {response}\nExpected records: {len(chunk)}\n\nINPUTS:\n```jsonl\n' + m.jsonl(inputs) + '```\n')
        jobs.append(dict(batch=number, ids=[r['id'] for r in chunk], records=len(chunk),
                         inputs=data, attachment=attachment, response=response))
    m.dump(folder / 'jobs.json', jobs)
    return jobs


def prepare():
    pilot.check()
    if OUT.exists():
        raise ValueError('refuse_overwrite_collection')
    targets, reused = m.lines(TARGETS), m.lines(PILOT_BRIEFS)
    queued = partition(targets, reused)
    OUT.mkdir()
    (OUT / 'reused').mkdir()
    (OUT / 'reused/pilot_briefs.private.jsonl').write_bytes(PILOT_BRIEFS.read_bytes())
    (OUT / 'targets_0500.private.jsonl').write_bytes(TARGETS.read_bytes())
    jobs = write_jobs(OUT / 'stage_1_brief_batches', queued, 'briefs')
    for name in ('stage_1_briefs', 'stage_2_rewrites'):
        (OUT / 'responses' / name).mkdir(parents=True)
    (OUT / 'brief_v4.schema.json').write_bytes((DOCS / 'brief_v4.schema.json').read_bytes())
    reused_ids = {r['id'] for r in reused}
    coverage = [dict(id=r['id'], target_sha256=r['target_sha256'], paper_id=r['paper_id'],
                     style_group=r['style_group'], brief_state='pilot_reused' if r['id'] in reused_ids else 'new_response_pending',
                     reconstruction_state='new_purpose_only_output_pending', training_ready=False) for r in targets]
    (OUT / 'coverage.private.jsonl').write_text(m.jsonl(coverage))
    policy = dict(total_pairs=500, brief_schema_version=4, enhanced_brief_fields=['topic', 'purpose', 'key_points', 'anchors', 'relationships'],
                  reconstruction_brief_fields=list(RECONSTRUCTION_FIELDS), excluded_reconstruction_fields=['relationships'],
                  reuse_pilot_briefs=5, new_briefs_to_collect=495, new_reconstructions_required=500,
                  reused_pilot_reconstructions=0, brief_output_edits=0, target_edits=0,
                  requested_model='GPT-6', requested_effort='High', actual_settings_independently_verified=False,
                  purpose_only_variant_previously_executed=False, provider_calls=0, training_calls=0)
    m.dump(OUT / 'policy.json', policy)
    summary = dict(total_pairs=500, author_group_counts=dict(Counter(r['style_group'] for r in targets)),
        completed_enhanced_briefs_reused=5, new_enhanced_brief_records=495, stage_1_jobs=len(jobs),
        stage_1_batch_sizes=[j['records'] for j in jobs], planned_purpose_only_reconstructions=500,
        stage_2_jobs_prepared=0, training_ready=False, provider_calls=0, training_calls=0)
    m.dump(OUT / 'summary.json', summary)
    (OUT / 'START_HERE.md').write_text('''# Full 500-pair collection: enhanced brief, purpose-only reconstruction

Scope is ALL 500 matched human targets. Every pair retains its enhanced brief,
including purpose and relationships. Reconstruction receives only topic, purpose,
key_points and anchors, plus the ordinary ID/status/notes wrapper. Relations stay
in the stored brief and are not sent to the reconstructor.

## Stage 1: collect the enhanced briefs

The five v4 pilot briefs are reused unchanged, with original hashes. Upload only
stage_1_brief_batches/batch_01.md through batch_17.md, one job per request, to GPT-6
with requested High effort. Each attachment includes the full prompt, relation
catalog and inputs. No additional attachments are required. These jobs cover the
other 495 targets: sixteen batches of 30, then 15.

Save briefs_01.jsonl through briefs_17.jsonl in:
responses/stage_1_briefs/ under this collection folder.

Keep completed responses. If one output is partial, preserve it and report missing
IDs; do not repeat completed work. Do not discard a large fraction of responses on
a surface threshold. Every target must receive an explicit assessment or repair
decision; genuine issues remain visible rather than silently omitted.
Cross-chat memory isolation and actual model/effort settings are not certified.

## Stage 2: reconstruct all 500 with purpose

After the responses return, tell Codex so the briefs can be checked. All enhanced
fields remain in the stored version. We will then prepare 17 reconstruction jobs
for ALL 500 targets, including the five pilot targets. Previous pilot A/B prose is
comparison data; neither condition is reused as this purpose-only version.

The Stage 2 packets are made only from actual returned briefs. Each reconstruction
receives topic, purpose, key_points and anchors, with no relationship annotations,
human paragraph or old prose. Batch membership may differ from Stage 1 because
the five reused briefs are inserted into their original 500-slot order.
Save the eventual rewrites_01.jsonl through rewrites_17.jsonl in responses/stage_2_rewrites/.

Local commands after all brief responses are available:
python3 scripts/data/prepare_gpt6_purpose_only_500.py check-briefs
python3 scripts/data/prepare_gpt6_purpose_only_500.py prepare-reconstruction

Formal validation is not independent semantic review. Candidate/source concerns
are recorded; no data are marked ready for training by packaging these files.
This purpose-only condition was selected by the user and has not yet been
evaluated as a separate condition. Earlier A/B findings are not evidence that it wins.
No teacher API calls, GPU jobs, detector scans or training are started here.
''')
    dependencies = [TARGETS, PILOT_BRIEFS, pilot.OUT / 'manifest.json',
        DOCS / 'BRIEF_ABSTRACTION_V4.md', DOCS / 'BRIEF_RECONSTRUCTION_V4.md', DOCS / 'brief_v4.schema.json',
        ROOT / 'src/fama_style/research/content_brief.py', ROOT / 'src/fama_style/research/content_brief_v3.py',
        ROOT / 'src/fama_style/research/content_brief_v4.py']
    m.dump(OUT / 'manifest.json', dict(builder_sha256=m.digest(Path(__file__)),
        dependencies={str(p.relative_to(ROOT)): m.digest(p) for p in dependencies},
        files={str(p.relative_to(OUT)): m.digest(p) for p in sorted(OUT.rglob('*')) if p.is_file()}))
    return check()


def check():
    manifest = m.read(OUT / 'manifest.json')
    if manifest['builder_sha256'] != m.digest(Path(__file__)):
        raise ValueError('collection_builder_changed')
    for name, digest in manifest['dependencies'].items():
        if m.digest(ROOT / name) != digest:
            raise ValueError('collection_dependency_changed')
    for name, digest in manifest['files'].items():
        if m.digest(OUT / name) != digest:
            raise ValueError('collection_artifact_changed')
    targets = m.lines(OUT / 'targets_0500.private.jsonl')
    reused = m.lines(OUT / 'reused/pilot_briefs.private.jsonl')
    queued = partition(targets, reused)
    actual_ids = []
    for job in m.read(OUT / 'stage_1_brief_batches/jobs.json'):
        inputs = m.lines(OUT / 'stage_1_brief_batches' / job['inputs'])
        if [r['id'] for r in inputs] != job['ids']:
            raise ValueError('stage_1_order')
        for row in inputs:
            if set(row) != {'id', 'target_sha256', 'original_paragraph'} or m.sha(row['original_paragraph']) != row['target_sha256']:
                raise ValueError('stage_1_binding')
        actual_ids.extend(job['ids'])
    if actual_ids != [r['id'] for r in queued]:
        raise ValueError('remaining_495_coverage')
    if (OUT / 'reused/pilot_briefs.private.jsonl').read_bytes() != PILOT_BRIEFS.read_bytes():
        raise ValueError('pilot_briefs_modified')
    return m.read(OUT / 'summary.json')


def load_all_briefs():
    check()
    targets = m.lines(OUT / 'targets_0500.private.jsonl')
    by_target = {r['id']: r for r in targets}
    rows = m.lines(OUT / 'reused/pilot_briefs.private.jsonl')
    response_files = {}
    for job in m.read(OUT / 'stage_1_brief_batches/jobs.json'):
        path = OUT / 'responses/stage_1_briefs' / job['response']
        batch = m.lines(path)
        if [r.get('id') for r in batch] != job['ids']:
            raise ValueError('brief_response_coverage')
        response_files[str(path.relative_to(OUT))] = m.digest(path)
        rows.extend(batch)
    if len(rows) != 500 or {r['id'] for r in rows} != set(by_target):
        raise ValueError('all_500_briefs_required')
    for row in rows:
        validate_brief_v4(row)
        if row['target_sha256'] != by_target[row['id']]['target_sha256']:
            raise ValueError('brief_response_target')
    by_id = {r['id']: r for r in rows}
    return [by_id[t['id']] for t in targets], response_files


def check_briefs():
    rows, files = load_all_briefs()
    return dict(total_briefs=500, collected_response_files=len(files), pilot_reused=5,
                statuses=dict(Counter(r['status'] for r in rows)), independent_semantic_verification=False)


def prepare_reconstruction():
    rows, files = load_all_briefs()
    folder = OUT / 'stage_2_reconstruction_batches'
    if folder.exists():
        raise ValueError('refuse_overwrite_full_reconstruction')
    payloads = [purpose_only_payload(r) for r in rows]
    jobs = write_jobs(folder, payloads, 'reconstruction')
    # Preserve the complete enhanced records alongside the projection, never in
    # the upload payload. Raw response files also stay at their original paths.
    (folder / 'retained_enhanced_briefs.private.jsonl').write_text(m.jsonl(rows))
    summary = dict(total_pairs=500, jobs=len(jobs), batch_sizes=[j['records'] for j in jobs],
        reconstruction_brief_fields=list(RECONSTRUCTION_FIELDS), relationship_fields_forwarded=0,
        brief_output_edits=0, statuses=dict(Counter(r['status'] for r in rows)), training_ready=False)
    m.dump(folder / 'summary.json', summary)
    (folder / 'START_HERE.md').write_text('''# Reconstruct all 500 with purpose only

Upload batch_01.md through batch_17.md, one per GPT-6 request with requested High
effort. Each attachment contains the prompt and only its purpose-only projection.
Do not attach retained_enhanced_briefs.private.jsonl, human paragraphs, Stage 1
files, relationship annotations or earlier rewrites. Prefer separate requests;
cross-chat memory isolation is not certified.

Save rewrites_01.jsonl through rewrites_17.jsonl in ../responses/stage_2_rewrites/.
These 17 jobs cover all 500 targets, including the five pilot targets: sixteen
batches of 30, then 20. Every complete enhanced brief remains saved separately.
Statuses and genuine source concerns remain visible. Packaging is not independent
semantic verification or authorization to train. No model calls are run here.
''')
    m.dump(folder / 'manifest.json', dict(parent_manifest_sha256=m.digest(OUT / 'manifest.json'),
        builder_sha256=m.digest(Path(__file__)), raw_response_files=files,
        files={str(p.relative_to(folder)): m.digest(p) for p in sorted(folder.iterdir()) if p.is_file()}))
    check_reconstruction()
    return summary


def check_reconstruction():
    rows, files = load_all_briefs()
    folder = OUT / 'stage_2_reconstruction_batches'
    manifest = m.read(folder / 'manifest.json')
    if manifest['raw_response_files'] != files or manifest['parent_manifest_sha256'] != m.digest(OUT / 'manifest.json') or manifest['builder_sha256'] != m.digest(Path(__file__)):
        raise ValueError('reconstruction_dependency_changed')
    for name, digest in manifest['files'].items():
        if m.digest(folder / name) != digest:
            raise ValueError('reconstruction_packet_changed')
    expected = [purpose_only_payload(r) for r in rows]
    actual = []
    targets = {r['id']: r['target'] for r in m.lines(OUT / 'targets_0500.private.jsonl')}
    for job in m.read(folder / 'jobs.json'):
        payloads = m.lines(folder / job['inputs'])
        if [r['id'] for r in payloads] != job['ids'] or any(set(r['brief']) != set(RECONSTRUCTION_FIELDS) for r in payloads):
            raise ValueError('purpose_only_fields_or_order')
        packet = (folder / job['attachment']).read_text()
        if any(targets[r['id']] in packet for r in payloads):
            raise ValueError('human_paragraph_leaked')
        actual.extend(payloads)
    if actual != expected or m.lines(folder / 'retained_enhanced_briefs.private.jsonl') != rows:
        raise ValueError('projection_or_enhanced_retention_changed')
    return m.read(folder / 'summary.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'check', 'check-briefs', 'prepare-reconstruction', 'check-reconstruction'])
    args = parser.parse_args()
    action = dict(prepare=prepare, check=check, **{'check-briefs': check_briefs,
        'prepare-reconstruction': prepare_reconstruction, 'check-reconstruction': check_reconstruction})[args.action]
    print(json.dumps(action(), indent=2))
