"""CPU-only verification of a versioned partial Gemini qualification checkpoint."""
import argparse
from collections import Counter
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location('gemini_completion', Path(__file__).with_name('complete_gemini_track_500.py'))
g = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(g)


def replay(source, patches):
    for patch in patches:
        before = patch['before']
        if not before or source.count(before) != patch['occurrences'] or patch['occurrences'] < 1:
            raise ValueError('patch_binding')
        source = source.replace(before, patch['after'])
    return source


def audit(version='resume_20261105_v2'):
    g.check()
    folder = g.RUN / version
    manifest = g.m.read(folder / 'manifest.json')
    for name, expected in manifest['files'].items():
        if g.m.digest(g.RUN / name) != expected:
            raise ValueError('checkpoint_artifact_changed')
    state = g.m.read(folder / 'state.json')
    rows = g.m.lines(folder / 'qualified_completed_reviews_v2.private.jsonl')
    prior = g.m.lines(g.RUN / 'qualified_completed_reviews_v1.private.jsonl')
    if rows[:len(prior)] != prior:
        raise ValueError('prior_decisions_changed')
    drafts = {r['id']:r for r in g.m.lines(g.RUN / 'drafts.private.jsonl')}
    targets = {r['id']:r for r in g.m.lines(g.RUN / 'targets.private.jsonl')}
    needed = {rid for rid, r in drafts.items() if r['review_required']}
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)) or not set(ids) <= needed:
        raise ValueError('qualified_identity')
    for row in rows:
        draft = drafts[row['id']]; target = targets[row['id']]
        if row['draft_sha256'] != g.m.sha(draft['plain_english']) or row['target_sha256'] != target['target_sha256']:
            raise ValueError('decision_text_binding')
        final = replay(draft['plain_english'], row['applied_patches'])
        if final != row['final_source'] or g.m.sha(final) != row['source_sha256'] or row['status'] != 'approved_automated_pilot':
            raise ValueError('unbound_or_unapproved_result')
        if g.m.numbers(final) != g.m.numbers(target['target']) and not row['numeric_difference_explanation']:
            raise ValueError('unadjudicated_numeric_difference')
    reused = sum(not r['review_required'] for r in drafts.values())
    pending = needed - set(ids)
    if state['newly_qualified_after_adjudication'] != len(rows) or state['separate_reviews_pending'] != len(pending):
        raise ValueError('state_counts')
    if state['staged_accepted_total'] != reused + len(rows) or reused + len(needed) != 500:
        raise ValueError('cohort_coverage')
    if state['final_training_cohort_released'] or state['training_calls'] or state['paid_fallback']:
        raise ValueError('scope')
    return dict(qualified_reviewed_inputs=len(rows), accepted_reuses=reused,
                staged_input_qualified_total=reused+len(rows), reviews_pending=len(pending),
                total_inputs=500, repaired_reviewed_inputs=sum(bool(r['applied_patches']) for r in rows),
                shared_target_followups=len(state['pending_shared_target_issue_ids']), training_calls=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='resume_20261105_v2')
    args = parser.parse_args()
    import json
    print(json.dumps(audit(args.checkpoint), indent=2))
