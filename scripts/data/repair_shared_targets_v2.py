"""Version seven user-adjudicated cases and prepare five purpose-aware brief jobs.

Training-side private artifacts only; no provider calls, training or legacy edits.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location('full_outputs', Path(__file__).with_name('review_gpt6_full_outputs.py'))
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)
m = r.m
OUT = m.ROOT / 'research_runs/data_scaling_v1/shared_target_repairs_v2'
DOCS = m.ROOT / 'docs/data_scaling_v1'
PENDING = {'P250', 'P264', 'P291', 'P458', 'P484'}


def replace_once(text, edits):
    for e in edits:
        if not e.get('reason') or not e['before'] or text.count(e['before']) != 1:
            raise ValueError('repair_span_binding')
        text = text.replace(e['before'], e['after'], 1)
    return text


def target_versions(parent, decisions):
    by = {d['id']: d for d in decisions}
    if len(by) != len(decisions) or set(by) != PENDING | {'P134', 'P139'}:
        raise ValueError('seven_decisions_required')
    rows = copy.deepcopy(parent)
    for row in rows:
        rid = row['id']
        if rid not in by:
            continue
        d = by[rid]
        if m.sha(row['target']) != d['before_target_sha256'] or row['paper_id'] != d['paper_id']:
            raise ValueError('target_binding')
        text = replace_once(row['target'], d['target_edits'])
        if m.sha(text) != d['after_target_sha256']:
            raise ValueError('target_repair_hash')
        row.update(target=text, original_paragraph=text, target_sha256=m.sha(text), training_ready=False)
        row['provenance'].update(target_repair_version='shared_target_repairs_v2',
            previous_target_sha256=d['before_target_sha256'], target_kind=d['target_kind'],
            user_adjudication_date='2026-11-06', matched_target_slot=rid,
            shared_target_registry=str((OUT / 'targets_0500.private.jsonl').relative_to(m.ROOT)))
        if d.get('replacement_paragraph_id'):
            row['provenance'].update(replaced_parent_target_id=row['parent_target_id'],
                replaced_matched_mixed_id=row['provenance']['matched_mixed_id'])
            row['parent_target_id'] = d['replacement_paragraph_id']
            row['parent_id'] = d['replacement_paragraph_id']
            row['provenance']['matched_mixed_id'] = rid + '::shared_target_repairs_v2'
        if rid in PENDING:
            row['quality'].update(text_pair_status='Target repaired · fresh source pending',
                target_review_pending=False, source_alignment_pending=True,
                release_status='candidate_only_not_training_ready')
    if len(rows) != 500 or len({x['id'] for x in rows}) != 500 or len({x['target_sha256'] for x in rows}) != 500:
        raise ValueError('cohort_identity_or_duplicate_target')
    return rows


def prepare():
    if (OUT / 'manifest.json').exists():
        raise ValueError('refuse_overwrite_shared_repair')
    r.check()
    parent = m.lines(r.OUT / 'candidates_0500.private.jsonl')
    decisions = m.lines(OUT / 'decisions.private.jsonl')
    rows = target_versions(parent, decisions)
    recovery = m.lines(OUT / 'source_recoveries.private.jsonl')
    prior_responses = {x['id']: x for x in m.lines(r.old.OUT / 'assessed_rewrites.private.jsonl')}
    by = {x['id']: x for x in recovery}
    if len(by) != 2 or set(by) != {'P134', 'P139'}:
        raise ValueError('two_source_recoveries_required')
    combined = m.lines(r.OUT / 'reviewed_briefs_0495.private.jsonl')
    brief_by = {x['id']: x for x in combined}
    for row in rows:
        rid = row['id']
        if rid not in by:
            continue
        x = by[rid]
        old_source = prior_responses[rid]['final_source']
        if m.sha(old_source) != x['before_source_sha256']:
            raise ValueError('source_recovery_binding')
        source = replace_once(old_source, x['source_edits'])
        if m.sha(source) != x['after_source_sha256']:
            raise ValueError('source_recovery_hash')
        row.update(source=source, plain_english=source, source_sha256=m.sha(source))
        row['provenance'].update(input_origin='brief_reconstruction_recovered_v2', source_repair_author='Codex',
            source_recovery_audit=str((OUT / 'source_recoveries.private.jsonl').relative_to(m.ROOT)))
        row['quality'].update(text_pair_status='Candidate · fresh brief reconstruction',
            target_review_pending=False, source_alignment_pending=False,
            release_status='candidate_only_not_training_ready')
        original = brief_by[rid]['effective_response']
        if r.b.b.row_hash(original) != x['before_brief_sha256']:
            raise ValueError('brief_recovery_binding')
        effective = r.b.b.apply_edits(original, x['brief_edits'])
        if x['target_sha256'] != row['target_sha256']:
            raise ValueError('brief_target_rebind')
        effective['target_sha256'] = row['target_sha256']
        m.validate_brief(effective, row)
        brief_by[rid].update(effective_response=effective, recovery_audit=x,
                            brief_schema_version=1)
    shared = []
    for row in rows:
        rid = row['id']
        decision = next((x for x in decisions if x['id'] == rid), None)
        shared.append(dict(id=rid, paper_id=row['paper_id'], style_group=row['style_group'], authors=row['authors'],
            parent_target_id=row['parent_target_id'], canonical_pdf_sha256=row['provenance']['canonical_pdf_sha256'],
            target=row['target'], target_sha256=row['target_sha256'], split='training',
            target_version='shared_target_repairs_v2', target_kind=decision['target_kind'] if decision else 'prior_target_unchanged',
            previous_target_sha256=decision['before_target_sha256'] if decision else row['target_sha256'],
            cohort_membership_replaced=rid == 'P291'))
    (OUT / 'targets_0500.private.jsonl').write_text(m.jsonl(shared))
    (OUT / 'gpt6_candidates_0500.private.jsonl').write_text(m.jsonl(rows))
    (OUT / 'gpt6_candidates_0200.private.jsonl').write_text(m.jsonl(rows[:200]))
    (OUT / 'reviewed_briefs_0495.private.jsonl').write_text(m.jsonl([brief_by[k] for k in sorted(brief_by)]))
    # A shared target overlay does not certify old Gemini inputs against new content.
    overlay = [dict(id=d['id'], old_target_sha256=d['before_target_sha256'], new_target_sha256=d['after_target_sha256'],
                    action='rebind_and_reassess_source' if d['before_target_sha256'] != d['after_target_sha256'] else 'target_unchanged_user_adjudication',
                    ready_for_training=False) for d in decisions]
    (OUT / 'gemini_target_overlay.private.jsonl').write_text(m.jsonl(overlay))
    route_a = []
    for row in rows:
        if row['id'] in PENDING:
            continue
        brief = brief_by[row['id']]['effective_response']
        if brief['target_sha256'] != row['target_sha256'] or brief['status'] != 'ready':
            raise ValueError('brief_target_alignment')
        source = json.dumps(brief['brief'], ensure_ascii=False, sort_keys=True)
        route_a.append(dict(id=row['id'], split='training', brief=brief['brief'], brief_schema_version=1,
            source=source, source_sha256=m.sha(source), target=row['target'], target_sha256=row['target_sha256'],
            style_group=row['style_group'], paper_id=row['paper_id'], training_ready=False,
            provenance=dict(shared_target_registry=str((OUT / 'targets_0500.private.jsonl').relative_to(m.ROOT)),
                            canonical_pdf_sha256=row['provenance']['canonical_pdf_sha256'])))
    (OUT / 'brief_human_candidates_0495.private.jsonl').write_text(m.jsonl(route_a))
    jobs = OUT / 'stage_1_brief_batches'
    jobs.mkdir()
    selected = [row for row in rows if row['id'] in PENDING]
    payload = [dict(id=row['id'], target_sha256=row['target_sha256'], original_paragraph=row['target']) for row in selected]
    (jobs / 'inputs_01.jsonl').write_text(m.jsonl(payload))
    prompt = (DOCS / 'BRIEF_EXTRACTION_V2.md').read_text()
    (jobs / 'batch_01.md').write_text(prompt + '\nOutput filename: briefs_01.jsonl\nExpected records: 5\n\nINPUTS:\n```jsonl\n' + m.jsonl(payload) + '```\n')
    m.dump(jobs / 'jobs.json', [dict(batch=1, ids=[x['id'] for x in selected], records=5,
                                   attachment='batch_01.md', response='briefs_01.jsonl', brief_schema_version=2)])
    for name in ('stage_1_briefs', 'stage_2_rewrites'):
        (OUT / 'responses' / name).mkdir(parents=True, exist_ok=True)
    (OUT / 'START_HERE.md').write_text('''# Five repaired-target briefs, with purpose

Upload only stage_1_brief_batches/batch_01.md to GPT-6 with requested High effort.
It includes all five repaired/replaced targets still lacking fresh reconstruction:
P250, P264, P291, P458 and P484. Save briefs_01.jsonl in responses/stage_1_briefs/.
Do not upload the earlier 495 briefs again. Their work is preserved and reused.

P134 has verified chronology corrections; P139 retains the user-accepted target.
Both prior brief/reconstruction responses have local commentary removal and are
recovered, with exact edits retained. No new manual annotations are needed for them.

After the five briefs return, assess them and create a separate brief-only Stage 2
packet with the new purpose-aware reconstruction prompt. Do not attach originals
or this Stage 1 batch to Stage 2. Cross-chat memory isolation is not assumed.

The shared target registry is for both teacher tracks. Gemini target bindings must
be updated and their source alignment assessed; existing Gemini reviews are not
silently transferred to corrected targets. Historical versions remain unchanged.
All files are candidate/audit artifacts, not an authorized training release.
''')
    summary = dict(decisions=7, changed_human_targets=sum(d['before_target_sha256'] != d['after_target_sha256'] for d in decisions),
        cohort_membership_replacements=1, original_pdfs_modified=0, historical_datasets_modified=0,
        current_fresh_prose_candidates=495, current_brief_human_candidates=495,
        missing_fresh_source_ids=sorted(PENDING), new_manual_brief_jobs=1, new_manual_brief_records=5,
        purpose_enrichment_of_legacy_495='not_requested_not_fabricated', gemini_target_reassessment_pending=True,
        independent_semantic_verification=False, training_ready=False, provider_calls=0, training_calls=0)
    m.dump(OUT / 'summary.json', summary)
    deps = [r.OUT / 'manifest.json', r.OUT / 'candidates_0500.private.jsonl', r.OUT / 'reviewed_briefs_0495.private.jsonl',
            r.old.OUT / 'assessed_rewrites.private.jsonl', DOCS / 'BRIEF_EXTRACTION_V2.md', DOCS / 'BRIEF_RECONSTRUCTION_V2.md',
            DOCS / 'brief_v2.schema.json', m.ROOT / 'src/fama_style/research/content_brief.py',
            m.ROOT / 'research_runs/taxonomy_discovery_v1/responses/task_3_consolidation/taxonomy_v0_1_proposal_v2.json']
    for d in decisions:
        deps.append(m.ROOT / d['pdf'])
    m.dump(OUT / 'manifest.json', dict(builder_sha256=m.digest(Path(__file__)),
        dependencies={str(p.relative_to(m.ROOT)): m.digest(p) for p in set(deps)},
        files={str(p.relative_to(OUT)): m.digest(p) for p in sorted(OUT.rglob('*')) if p.is_file()}))
    return check()


def check():
    manifest = m.read(OUT / 'manifest.json')
    if manifest['builder_sha256'] != m.digest(Path(__file__)):
        raise ValueError('repair_builder_changed')
    for name, digest in manifest['dependencies'].items():
        if m.digest(m.ROOT / name) != digest:
            raise ValueError('repair_dependency_changed')
    for name, digest in manifest['files'].items():
        if m.digest(OUT / name) != digest:
            raise ValueError('repair_artifact_changed')
    parent = m.lines(r.OUT / 'candidates_0500.private.jsonl')
    decisions = m.lines(OUT / 'decisions.private.jsonl')
    target_rows = target_versions(parent, decisions)
    candidates = m.lines(OUT / 'gpt6_candidates_0500.private.jsonl')
    if [x['target'] for x in target_rows] != [x['target'] for x in candidates] or any(x['training_ready'] for x in candidates):
        raise ValueError('candidate_targets_or_status')
    if candidates[:200] != m.lines(OUT / 'gpt6_candidates_0200.private.jsonl'):
        raise ValueError('repair_prefix')
    if len(m.lines(OUT / 'brief_human_candidates_0495.private.jsonl')) != 495:
        raise ValueError('route_a_coverage')
    jobs = m.read(OUT / 'stage_1_brief_batches/jobs.json')
    payloads = m.lines(OUT / 'stage_1_brief_batches/inputs_01.jsonl')
    if len(payloads) != 5 or set(jobs[0]['ids']) != PENDING or {x['id'] for x in payloads} != PENDING:
        raise ValueError('five_missing_briefs')
    if any(set(x) != {'id', 'target_sha256', 'original_paragraph'} for x in payloads):
        raise ValueError('brief_payload')
    shared = m.lines(OUT / 'targets_0500.private.jsonl')
    if [(x['id'], x['target_sha256']) for x in shared] != [(x['id'], x['target_sha256']) for x in candidates]:
        raise ValueError('shared_target_registry')
    return m.read(OUT / 'summary.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'check'])
    args = parser.parse_args()
    print(json.dumps(prepare() if args.action == 'prepare' else check(), indent=2))
