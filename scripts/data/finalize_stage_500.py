"""Freeze audited neutral inputs; CPU only, with no provider or training calls.

Full-paragraph reviewer proposals are deliberately excluded: that reviewer
frequently copied the human target. Only hash-bound minimal decisions are used.
"""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location('stage500', Path(__file__).with_name('expand_stage_500.py'))
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def targets():
    rows = m.effective_targets()
    by = {r['id']: r for r in rows}
    audit = m.RUN / 'final_target_repairs.private.jsonl'
    for repair in m.lines(audit):
        row = by[repair['id']]
        if m.sha(row['target']) != repair['before_sha256'] or m.sha(repair['after']) != repair['after_sha256']:
            raise ValueError('final_target_repair_binding')
        for name, digest in repair['evidence'].items():
            if m.digest(m.RUN / name) != digest:
                raise ValueError('final_target_evidence_changed')
        row['extraction_target_sha256'] = row.get('extraction_target_sha256', row['target_sha256'])
        row['target'] = repair['after']
        row['target_sha256'] = repair['after_sha256']
        row['final_target_repair'] = dict(audit_sha256=m.digest(audit), reason=repair['reason'], evidence=repair['evidence'])
    replacement = m.RUN / 'replacement_v1'
    rows.extend(m.lines(replacement / 'targets.private.jsonl'))
    return rows


def validate_decision(decision, row, draft):
    if decision['draft_sha256'] != m.sha(draft['plain_english']) or decision['target_sha256'] != m.sha(row['target']):
        raise ValueError('final_decision_text_binding')
    source = draft['plain_english']
    for patch in decision['applied_patches']:
        if not patch['before'] or source.count(patch['before']) != patch['occurrences']:
            raise ValueError('final_patch_anchor_changed')
        source = source.replace(patch['before'], patch['after'])
    if source != decision['final_source'] or m.sha(source) != decision['source_sha256']:
        raise ValueError('final_patch_replay_changed')
    if m.numbers(source) != m.numbers(row['target']) and not decision['numeric_difference_explanation']:
        raise ValueError('unresolved_numeric_difference')
    if decision['status'] != 'approved_automated_pilot':
        raise ValueError('unapproved_final_decision')
    return source


def freeze():
    m.check()
    manifest = m.read(m.RUN / 'final_decisions_manifest.json')
    for name, digest in manifest['files'].items():
        if m.digest(m.RUN / name) != digest:
            raise ValueError('final_decision_evidence_changed')
    if manifest['finalizer_sha256'] != m.digest(Path(__file__)):
        raise ValueError('finalizer_changed')
    decisions = m.lines(m.RUN / 'final_input_decisions.private.jsonl')
    selected = {r['id']: r for r in targets()}
    exclusions = m.lines(m.RUN / 'excluded_non_narrative.private.jsonl')
    excluded_ids = {r['id'] for r in exclusions}
    if len(excluded_ids) != 3 or any(r['id'] in excluded_ids for r in decisions):
        raise ValueError('excluded_record_reintroduced')
    if len(decisions) != 215 or {r['id'] for r in decisions} != set(selected) - excluded_ids:
        raise ValueError('final_coverage')
    drafts = {r['id']: r for r in m.lines(m.RUN / 'generated_drafts.private.jsonl')}
    replacement = m.RUN / 'replacement_v1'
    for r in m.lines(replacement / 'drafts.private.jsonl'):
        drafts[r['id']] = dict(r, generator=m.MODEL, thinking_level='low',
                              generation_response_sha256=m.digest(replacement / 'generation/response_01.private.json'))
    approved = []
    decision_hash = m.digest(m.RUN / 'final_input_decisions.private.jsonl')
    for decision in decisions:
        row = selected[decision['id']]
        draft = drafts[row['id']]
        source = validate_decision(decision, row, draft)
        approved.append(m.normalize_pair(
            dict(id=row['id'] + '::stage500_gemini_v1', parent_id=row['parent_id'], paper_id=row['paper_id'],
                 style_group=row['style_group'], authors=row['authors']), source, row['target'],
            dict(source_file=row['source_file'], source_file_sha256=row['source_file_sha256'],
                 pdf_sha256=row['pdf_sha256'], line_refs=row['line_refs'], page_location=row['page_location'],
                 quality_flags=row['quality_flags'], generator=draft['generator'],
                 generation_response_sha256=draft['generation_response_sha256'],
                 generation_thinking_level=draft.get('thinking_level', 'historical_or_provider_default'),
                 final_decisions_sha256=decision_hash, minimal_review_response_sha256=decision['review_response_sha256'],
                 input_repaired=source != draft['plain_english'],
                 human_target_changed=bool(row.get('target_repair') or row.get('final_target_repair')),
                 target_repair=row.get('target_repair'), final_target_repair=row.get('final_target_repair'),
                 extraction_target_sha256=row.get('extraction_target_sha256', row['target_sha256']),
                 verifier_independent_of_generator=False, numeric_difference_explanation=decision['numeric_difference_explanation']),
            'Separate Gemini meaning-only patch review plus Codex PDF/numeric adjudication; automated pilot approval, not independent human verification'))
    reused = m.lines(m.RUN / 'reused.private.jsonl')
    extras = reused[200:] + approved
    groups = {g: m.interleave([r for r in extras if r['style_group'] == g]) for g in m.GROUPS}
    combined = reused[:200] + [r for i in range(max(map(len, groups.values()))) for g in m.GROUPS for r in groups[g][i:i+1]]
    counts = dict(Counter(r['style_group'] for r in combined))
    if len(combined) != 500 or counts != {g: 125 for g in m.GROUPS}:
        raise ValueError('final_count_or_balance')
    if len({m.canonical(r['target']) for r in combined}) != 500 or len({r['parent_target_id'] for r in combined}) != 500:
        raise ValueError('duplicate_target_or_parent')
    if m.OUT.exists() or (m.RUN / 'assembly').exists():
        raise ValueError('refuse_overwrite_frozen_release')
    prefix = m.ROOT / 'data/data_scaling_v1/stage_0200_salvage_v1/training.private.jsonl'
    if m.digest(prefix) != m.read(m.RUN / 'config.json')['prefix_sha256']:
        raise ValueError('prefix_changed')
    summary = dict(protocol='stage_0500_expansion_v1', approved_pairs=500, new_approved=215,
                   groups=counts, target=500, complete=True, holds=0, excluded_non_narrative=3,
                   replacements=3, original_200_unchanged=True, teacher_composition_changed=True,
                   semantic_review='automated pilot acceptance; same-family model checks and Codex adjudication, not independent verification',
                   input_repairs=sum(r['provenance']['input_repaired'] for r in approved),
                   new_human_target_repairs=sum(r['provenance']['human_target_changed'] for r in approved),
                   training_calls=0, free_tier_only=True)
    m.OUT.mkdir(parents=True)
    suffix = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in combined[200:]).encode()
    (m.OUT / 'training.private.jsonl').write_bytes(prefix.read_bytes() + suffix)
    m.write(m.OUT / 'summary.json', summary)
    m.write(m.OUT / 'manifest.json', dict(parent_200_sha256=m.digest(prefix), config_sha256=m.digest(m.RUN / 'config.json'),
        final_decisions_manifest_sha256=m.digest(m.RUN / 'final_decisions_manifest.json'),
        files={p.name: m.digest(p) for p in m.OUT.iterdir() if p.is_file()}))
    checkpoint = m.RUN / 'assembly'
    checkpoint.mkdir()
    m.write_lines(checkpoint / 'new_approved.private.jsonl', approved)
    m.write(checkpoint / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    freeze()
