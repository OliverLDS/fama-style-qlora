"""Extend brief-first coverage to the complete frozen training cohort, without recollection.

Uses training data only. No providers, training, model downloads or target edits.
"""
import argparse
import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location('contrast_outputs', Path(__file__).with_name('review_gpt6_contrast_outputs.py'))
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)
m = review.m
OUT = m.OUT.parent / 'full_reconstruction_v1'
TARGET_HOLDS = {
    'P250': 'Conflicting mean values in archived source; do not guess median.',
    'P264': 'Paragraph begins mid-sentence across PDF pages; restore boundary first.',
    'P291': 'Publisher boilerplate is not author narrative; target replacement needed.',
    'P458': 'Separate conclusion heading from narrative in a versioned target repair.',
    'P484': 'Separate subsection label from narrative in a versioned target repair.',
}


def partition(targets, existing, holds):
    ids = [r['id'] for r in targets]
    if len(set(ids)) != len(ids) or not set(existing) <= set(ids) or not set(holds) <= set(ids):
        raise ValueError('invalid_coverage')
    if set(existing) & set(holds):
        raise ValueError('existing_and_missing_hold_overlap')
    return ([r for r in targets if r['id'] not in existing and r['id'] not in holds],
            [r for r in targets if r['id'] in holds])


def prepare():
    review.check()
    if OUT.exists():
        raise ValueError('refuse_overwrite_full_reconstruction')
    targets = m.lines(m.OUT / 'audit.private.jsonl')
    briefs = {r['id']: r for r in m.lines(review.b.AUDIT / 'reviewed_briefs.private.jsonl')}
    rewrites = {r['id']: r for r in m.lines(review.OUT / 'assessed_rewrites.private.jsonl')}
    if set(briefs) != set(rewrites):
        raise ValueError('reuse_coverage_mismatch')
    queue, blocked = partition(targets, briefs, TARGET_HOLDS)
    OUT.mkdir()
    m.write_jobs(OUT / 'stage_1_brief_batches', queue, 'briefs')
    for name in ('stage_1_briefs', 'stage_2_rewrites'):
        (OUT / 'responses' / name).mkdir(parents=True)
    reused = []
    route_a = []
    for t in targets:
        rid = t['id']
        if rid not in briefs:
            continue
        b, r = briefs[rid], rewrites[rid]
        if b['effective_response']['target_sha256'] != t['target_sha256'] or r['target_sha256'] != t['target_sha256']:
            raise ValueError('reuse_target_mismatch')
        usable = r['adjudication']['decision'] != 'source_concern' and b['effective_response']['status'] == 'ready'
        reused.append(dict(id=rid, target_sha256=t['target_sha256'], brief=b['effective_response'],
                           reconstruction=r['final_source'], source_sha256=r['source_sha256'],
                           usable=usable, training_ready=False,
                           brief_provenance=str((review.b.AUDIT / 'reviewed_briefs.private.jsonl').relative_to(m.ROOT)),
                           reconstruction_provenance=str((review.OUT / 'assessed_rewrites.private.jsonl').relative_to(m.ROOT))))
        if usable:
            route_a.append(dict(id=rid, split='training', source=b['effective_response']['brief'],
                                target=t['target'], target_sha256=t['target_sha256'],
                                style_group=t['style_group'], paper_id=t['paper_id'],
                                brief_sha256=review.b.row_hash(b['effective_response']['brief']),
                                training_ready=False, status='Candidate · brief-to-human preview'))
    m.dump(OUT / 'coverage.json', dict(targets=len(targets), existing_responses=len(reused),
           usable_reconstructions=sum(r['usable'] for r in reused),
           queued_missing_briefs=len(queue), blocked_missing_briefs=len(blocked),
           existing_source_concerns=[r['id'] for r in reused if not r['usable']],
           complete=False, training_ready=False))
    (OUT / 'reused_0175.private.jsonl').write_text(m.jsonl(reused))
    (OUT / 'route_a_preview.private.jsonl').write_text(m.jsonl(route_a))
    (OUT / 'target_holds.private.jsonl').write_text(m.jsonl([
        dict(id=r['id'], target_sha256=r['target_sha256'], reason=TARGET_HOLDS[r['id']],
             decision='repair_before_manual_annotation', target_unchanged=True) for r in blocked]))
    reused_by_id = {x['id']: x for x in reused}
    ledger = [dict(id=r['id'], target_sha256=r['target_sha256'],
                  stage=('reused_usable' if reused_by_id[r['id']]['usable'] else 'reused_source_concern')
                  if r['id'] in reused_by_id else
                  ('target_repair_pending' if r['id'] in TARGET_HOLDS else 'brief_queued')) for r in targets]
    (OUT / 'coverage.private.jsonl').write_text(m.jsonl(ledger))
    (OUT / 'START_HERE.md').write_text('''# Complete-cohort brief-first reconstruction

Reuse all 175 completed brief/reconstruction responses. Do not upload them again.
There are 320 ready missing targets in 11 Stage 1 jobs (ten of 30, one of 20).
Upload only stage_1_brief_batches/batch_01.md through batch_11.md, one job per request.
Save briefs_01.jsonl through briefs_11.jsonl in responses/stage_1_briefs/ here.
Use GPT-6 with requested High effort. Record actual deviations if known; cross-chat
memory isolation is not assumed. No need to confirm session independence.

Five further missing targets require source repair first, recorded in target_holds.private.jsonl.
Two already-collected cases also retain source concerns. They are not discarded,
and the full 500-target release is not complete yet.

After the briefs return, assess and repair them locally, then prepare Stage 2 jobs
containing only reviewed briefs and IDs/hashes. Do not attach originals, old rewrites,
or this Stage 1 packet to Stage 2. Keep the two stages in separate requests.
No Stage 2 jobs are fabricated before actual reviewed briefs exist.

The same brief and human target support Route A; fresh reconstruction and human
target support Route B. route_a_preview.private.jsonl contains the 173 currently
usable training-side brief/target candidates, not an approved training release.
All frozen prior datasets, targets and experiments remain unchanged.
''')
    dependencies = [m.OUT / 'audit.private.jsonl', review.b.AUDIT / 'reviewed_briefs.private.jsonl',
                    review.OUT / 'assessed_rewrites.private.jsonl', review.OUT / 'manifest.json']
    m.dump(OUT / 'manifest.json', dict(builder_sha256=m.digest(Path(__file__)),
        dependencies={str(p.relative_to(m.ROOT)): m.digest(p) for p in dependencies},
        files={str(p.relative_to(OUT)): m.digest(p) for p in sorted(OUT.rglob('*')) if p.is_file()}))
    return check()


def check():
    manifest = m.read(OUT / 'manifest.json')
    if manifest['builder_sha256'] != m.digest(Path(__file__)):
        raise ValueError('builder_changed')
    for name, digest in manifest['dependencies'].items():
        if m.digest(m.ROOT / name) != digest:
            raise ValueError('dependency_changed')
    for name, digest in manifest['files'].items():
        if m.digest(OUT / name) != digest:
            raise ValueError('packet_changed')
    coverage = m.lines(OUT / 'coverage.private.jsonl')
    jobs = m.read(OUT / 'stage_1_brief_batches/jobs.json')
    ids = [rid for job in jobs for rid in job['ids']]
    if len(coverage) != 500 or len({r['id'] for r in coverage}) != 500 or len(ids) != 320 or len(set(ids)) != 320:
        raise ValueError('coverage_changed')
    if set(ids) != {r['id'] for r in coverage if r['stage'] == 'brief_queued'}:
        raise ValueError('queue_mismatch')
    return m.read(OUT / 'coverage.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'check'])
    args = parser.parse_args()
    print(json.dumps(prepare() if args.action == 'prepare' else check(), indent=2))
