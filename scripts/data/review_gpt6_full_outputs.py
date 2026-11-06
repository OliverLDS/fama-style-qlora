"""Replay reconstruction repairs and stage matched brief/prose candidates; CPU only.

No development/test reads, provider calls, training or modification of raw responses.
"""
import argparse
from collections import Counter
import copy
import importlib.util
import json
from pathlib import Path
import statistics

SPEC = importlib.util.spec_from_file_location('full_briefs', Path(__file__).with_name('review_gpt6_full_briefs.py'))
b = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(b)
m = b.m
old = b.f.review
OUT = b.f.OUT / 'rewrite_quality_v1'


def load():
    b.check()
    targets = {r['id']: r for r in m.lines(m.OUT / 'audit.private.jsonl')}
    briefs = {r['id']: r['effective_response'] for r in m.lines(b.OUT / 'reviewed_briefs.private.jsonl')}
    rows, files = [], {}
    for job in m.read(b.STAGE2 / 'jobs.json'):
        path = b.f.OUT / 'responses/stage_2_rewrites' / job['response']
        batch = m.lines(path)
        if [r.get('id') for r in batch] != job['ids']:
            raise ValueError('rewrite_batch_identity')
        files[str(path.relative_to(b.f.OUT))] = m.digest(path)
        for row in batch:
            old.validate_response(row, targets[row['id']])
        rows.extend(batch)
    if len(rows) != len(briefs) or {r['id'] for r in rows} != set(briefs):
        raise ValueError('rewrite_coverage')
    return rows, files, targets, briefs


def repair(row, decision):
    if decision['id'] != row['id'] or decision['target_sha256'] != row['target_sha256'] or decision['raw_source_sha256'] != m.sha(row['plain_english']):
        raise ValueError('repair_identity')
    if decision['decision'] not in {'accept', 'repair'} or bool(decision['patches']) != (decision['decision'] == 'repair') or not decision['reason']:
        raise ValueError('repair_decision')
    return old.apply_patches(row['plain_english'], decision['patches'])


def merged_candidates(parent, final, targets):
    result = []
    parent_hash = m.digest(old.OUT / 'candidates_0500.private.jsonl')
    for prior in parent:
        row = copy.deepcopy(prior)
        rid = row['id']
        if rid in final:
            r = final[rid]
            row.update(source=r['final_source'], plain_english=r['final_source'], source_sha256=r['source_sha256'], training_ready=False)
            row['provenance'].update(parent_dataset=str((old.OUT / 'candidates_0500.private.jsonl').relative_to(m.ROOT)),
                parent_dataset_sha256=parent_hash, previous_source_sha256=prior['source_sha256'],
                input_origin='full_brief_reconstruction_v1',
                reconstruction_response_file=r['response_file'],
                assessment_file=str((OUT / 'assessed_rewrites.private.jsonl').relative_to(m.ROOT)),
                source_repair_author='Codex' if r['adjudication']['patches'] else None)
            row['quality'].update(text_pair_status='Candidate · fresh brief reconstruction',
                                  target_review_pending=False, release_status='candidate_only_not_training_ready')
        if row['target'] != targets[rid]['target'] or row['target_sha256'] != m.sha(row['target']):
            raise ValueError('human_target_changed')
        result.append(row)
    if len(result) != 500 or len({r['id'] for r in result}) != 500:
        raise ValueError('candidate_cohort')
    return result


def prepare():
    if (OUT / 'manifest.json').exists():
        raise ValueError('refuse_overwrite_output_assessment')
    old.check()
    rows, files, targets, briefs = load()
    decisions = m.lines(OUT / 'adjudications.private.jsonl')
    by = {r['id']: r for r in decisions}
    if len(by) != len(decisions) or set(by) != set(briefs):
        raise ValueError('decision_coverage')
    response_paths = {}
    for job in m.read(b.STAGE2 / 'jobs.json'):
        for rid in job['ids']:
            response_paths[rid] = 'responses/stage_2_rewrites/' + job['response']
    assessed = []
    for row in rows:
        rid = row['id']
        final = repair(row, by[rid])
        assessed.append(dict(id=rid, target_sha256=row['target_sha256'], raw_response=row,
            raw_source_sha256=m.sha(row['plain_english']), final_source=final, source_sha256=m.sha(final),
            response_file=str((b.f.OUT / response_paths[rid]).relative_to(m.ROOT)), adjudication=by[rid],
            old_metrics=targets[rid]['metrics'], raw_metrics=m.contrast(row['plain_english'], targets[rid]['target']),
            final_metrics=m.contrast(final, targets[rid]['target']), independent_semantic_verification=False))
    (OUT / 'assessed_rewrites.private.jsonl').write_text(m.jsonl(assessed))
    final = {r['id']: r for r in assessed}
    candidates = merged_candidates(m.lines(old.OUT / 'candidates_0500.private.jsonl'), final, targets)
    (OUT / 'candidates_0500.private.jsonl').write_text(m.jsonl(candidates))
    (OUT / 'candidates_0200.private.jsonl').write_text(m.jsonl(candidates[:200]))
    corrections = m.lines(OUT / 'brief_clarifications.private.jsonl')
    cb = {r['id']: r for r in corrections}
    if len(cb) != len(corrections) or not set(cb) <= set(briefs):
        raise ValueError('brief_correction_identity')
    combined_briefs = {r['id']: dict(id=r['id'], effective_response=r['effective_response'],
                        origin=str((old.b.AUDIT / 'reviewed_briefs.private.jsonl').relative_to(m.ROOT)), additional_edits=[])
                       for r in m.lines(old.b.AUDIT / 'reviewed_briefs.private.jsonl')}
    for rid, brief in briefs.items():
        edits = cb[rid]['edits'] if rid in cb else []
        if rid in cb and cb[rid]['before_sha256'] != b.b.row_hash(brief):
            raise ValueError('brief_correction_binding')
        effective = b.b.apply_edits(brief, edits)
        m.validate_brief(effective, targets[rid])
        combined_briefs[rid] = dict(id=rid, effective_response=effective,
            origin=str((b.OUT / 'reviewed_briefs.private.jsonl').relative_to(m.ROOT)), additional_edits=edits)
    (OUT / 'reviewed_briefs_0495.private.jsonl').write_text(m.jsonl([combined_briefs[r] for r in sorted(combined_briefs)]))
    route_a = []
    for row in candidates:
        rid = row['id']
        if row['quality']['text_pair_status'] != 'Candidate · fresh brief reconstruction':
            continue
        brief = combined_briefs[rid]['effective_response']
        if brief['status'] != 'ready' or brief['target_sha256'] != row['target_sha256']:
            raise ValueError('route_a_not_ready')
        source = json.dumps(brief['brief'], ensure_ascii=False, sort_keys=True)
        route_a.append(dict(id=rid, split='training', source=source, brief=brief['brief'], source_sha256=m.sha(source),
            target=row['target'], target_sha256=row['target_sha256'], paper_id=row['paper_id'], style_group=row['style_group'],
            training_ready=False, provenance=dict(brief_origin=combined_briefs[rid]['origin'],
            brief_revision_file=str((OUT / 'reviewed_briefs_0495.private.jsonl').relative_to(m.ROOT)),
            canonical_pdf_sha256=row['provenance']['canonical_pdf_sha256'], matched_mixed_id=row['provenance']['matched_mixed_id'])))
    (OUT / 'brief_human_candidates_0493.private.jsonl').write_text(m.jsonl(route_a))
    metrics = {}
    for key in ('word_similarity', 'ordinary_shared_fivegram_coverage', 'longest_ordinary_words', 'word_length_ratio'):
        metrics[key] = {label: statistics.median(r[field][key] for r in assessed)
                        for label, field in [('prior', 'old_metrics'), ('raw_reconstruction', 'raw_metrics'), ('final_reconstruction', 'final_metrics')]}
    summary = dict(records=len(rows), response_files=len(files), raw_status_counts=dict(Counter(r['rewrite_status'] for r in rows)),
        decisions=dict(Counter(r['decision'] for r in decisions)), repaired_ids=[r['id'] for r in decisions if r['patches']],
        patches=sum(len(r['patches']) for r in decisions), brief_clarification_ids=sorted(cb), metrics=metrics,
        lower_word_similarity=sum(r['final_metrics']['word_similarity'] < r['old_metrics']['word_similarity'] for r in assessed),
        contrast_followup_ids=[r['id'] for r in assessed if m.classify(r['final_metrics'])[0] == 'regenerate_brief_first'],
        short_relative_to_target_ids=[r['id'] for r in assessed if r['final_metrics']['word_length_ratio'] < .45],
        short_ratio_threshold_advisory_only=True,
        reconstruction_count_including_reuse=495, fresh_prose_candidates=493, brief_human_candidates=len(route_a),
        retained_prior_inputs=7, missing_target_repair_ids=sorted(b.f.TARGET_HOLDS), previous_source_concern_ids=['P134', 'P139'],
        human_targets_changed=0, source_versions_discarded=0, training_ready=False,
        provider_calls=0, training_calls=0, development_examples_read=0, reserved_test_examples_read=0,
        limitations=['All new paragraphs assessed with targeted reference checks, not exhaustive claim alignment or independent certification.',
            'Compression and modest additions/emphasis shifts are accepted; short relative length is advisory, not a rejection threshold.',
            'Surface contrast does not establish distinctive human/author style or student-model/detector gains.',
            'P495 date/count ambiguity is scoped, not silently repaired in the human target.'])
    m.dump(OUT / 'summary.json', summary)
    dependencies = [b.OUT / 'manifest.json', old.OUT / 'manifest.json', old.b.AUDIT / 'reviewed_briefs.private.jsonl',
                    old.OUT / 'candidates_0500.private.jsonl', m.OUT / 'audit.private.jsonl']
    m.dump(OUT / 'manifest.json', dict(builder_sha256=m.digest(Path(__file__)), raw_response_files=files,
        dependencies={str(p.relative_to(m.ROOT)): m.digest(p) for p in dependencies},
        files={str(p.relative_to(OUT)): m.digest(p) for p in sorted(OUT.iterdir()) if p.is_file()}))
    return check()


def check():
    rows, files, targets, briefs = load()
    manifest = m.read(OUT / 'manifest.json')
    if manifest['builder_sha256'] != m.digest(Path(__file__)) or manifest['raw_response_files'] != files:
        raise ValueError('rewrite_sources_changed')
    for name, digest in manifest['dependencies'].items():
        if m.digest(m.ROOT / name) != digest:
            raise ValueError('rewrite_dependency_changed')
    for name, digest in manifest['files'].items():
        if m.digest(OUT / name) != digest:
            raise ValueError('assessment_changed')
    raw = {r['id']: r for r in rows}
    assessed = m.lines(OUT / 'assessed_rewrites.private.jsonl')
    for r in assessed:
        if r['raw_response'] != raw[r['id']] or repair(raw[r['id']], r['adjudication']) != r['final_source'] or m.sha(r['final_source']) != r['source_sha256']:
            raise ValueError('repair_replay')
    expected = merged_candidates(m.lines(old.OUT / 'candidates_0500.private.jsonl'), {r['id']: r for r in assessed}, targets)
    if expected != m.lines(OUT / 'candidates_0500.private.jsonl'):
        raise ValueError('candidate_replay')
    if expected[:200] != m.lines(OUT / 'candidates_0200.private.jsonl'):
        raise ValueError('nested_prefix')
    combined = {r['id']: r for r in m.lines(OUT / 'reviewed_briefs_0495.private.jsonl')}
    old_briefs = {r['id']: r['effective_response'] for r in m.lines(old.b.AUDIT / 'reviewed_briefs.private.jsonl')}
    originals = dict(old_briefs, **briefs)
    for rid, r in combined.items():
        if b.b.apply_edits(originals[rid], r['additional_edits']) != r['effective_response']:
            raise ValueError('brief_revision_replay')
    route_a = m.lines(OUT / 'brief_human_candidates_0493.private.jsonl')
    expected_ids = {r['id'] for r in expected if r['quality']['text_pair_status'] == 'Candidate · fresh brief reconstruction'}
    if len(combined) != 495 or len(assessed) != 320 or len(route_a) != 493 or {r['id'] for r in route_a} != expected_ids:
        raise ValueError('route_coverage')
    for r in route_a:
        if r['target'] != targets[r['id']]['target'] or r['brief'] != combined[r['id']]['effective_response']['brief'] or json.loads(r['source']) != r['brief'] or m.sha(r['source']) != r['source_sha256'] or r['training_ready']:
            raise ValueError('route_a_binding')
    return m.read(OUT / 'summary.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'check'])
    args = parser.parse_args()
    print(json.dumps(prepare() if args.action == 'prepare' else check(), indent=2))
