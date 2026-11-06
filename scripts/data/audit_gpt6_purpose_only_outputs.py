"""Stage all 500 observed outputs, audit risks, and prepare two focused repairs.

No raw-response edits, filtering, provider calls, training or reserved-test reads.
"""
import argparse
from collections import Counter
import copy
import importlib.util
import json
from pathlib import Path
import re
import statistics

SPEC = importlib.util.spec_from_file_location('brief_audit', Path(__file__).with_name('audit_gpt6_purpose_only_briefs.py'))
b = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(b)
c, m = b.c, b.m
OUT = c.OUT / 'output_audit_v1'
FOLLOWUPS = {
    'P139': dict(level='target_conflict', reason='The output adds premium in price, opposite the user-accepted target comparison of a discount relative to stocks. The brief omitted that comparison.', action='recollect_brief_then_reconstruct'),
    'P400': dict(level='timing_conflict', reason='The output describes subsequent-return forecasting; the paper section is Using Sentiment to Explain Current Returns. The brief did not retain that timing.', action='recollect_brief_with_context_then_reconstruct'),
    'P085': dict(level='scope_watch', reason='The closing IPO-population framing narrows the target new-list population; financing frictions also supply a mechanism not explicit in this paragraph.', action='retain_observed_with_scope_advisory'),
    'P313': dict(level='scope_watch', reason='IPO-firm wording may narrow the broader new-list phenomenon, although the cited age evidence concerns going public.', action='retain_observed_with_scope_advisory'),
    'P329': dict(level='scope_watch', reason='The main statistics still concern new listings, but IPO activity is used in the introduction.', action='retain_observed_with_scope_advisory'),
    'P353': dict(level='scope_watch', reason='The opening IPO entrants narrows a target explicitly covering IPO and non-IPO listings.', action='retain_observed_with_scope_advisory'),
    'P056': dict(level='model_flag_nonblocking', reason='The warning conflates current demand with later returns. Source context sorts monthly returns using prior-year sentiment; the paragraph retains the signs and describes future returns.', action='retain_observed_warning_adjudicated'),
    'P206': dict(level='model_flag_nonblocking', reason='The output preserves the conditional small-b adjustment claim. Lack of a complete parameter definition in an excerpt does not by itself invalidate that conditional account.', action='retain_observed_warning_adjudicated'),
    'P289': dict(level='brief_concern_not_repeated', reason='The output does not repeat the brief permanent-pool overstatement; it describes repeated cohorts and dividend histories.', action='retain_observed'),
    'P213': dict(level='brief_concern_not_blocking', reason='The output keeps newly listed firms as the main population; internet IPOs describe a subset of the acceleration.', action='retain_observed'),
    'P233': dict(level='brief_concern_not_repeated', reason='The output uses new listings and exchange cohorts rather than narrowing the whole population to IPOs.', action='retain_observed')
}


def normalized_numerals(text):
    return {v.replace(',', '').rstrip('%') for v in re.findall(r'(?<![A-Za-z])\d+(?:[.,]\d+)*(?:%)?', text)}


def load():
    b.check()
    c.check_reconstruction()
    targets = m.lines(c.OUT / 'targets_0500.private.jsonl')
    by_target = {r['id']: r for r in targets}
    briefs, _ = c.load_all_briefs()
    validate = c.pilot.a.p.p3.v.r.old.validate_response
    rows, files, row_files = [], {}, {}
    for job in m.read(c.OUT / 'stage_2_reconstruction_batches/jobs.json'):
        path = c.OUT / 'responses/stage_2_rewrites' / job['response']
        batch = m.lines(path)
        if [r.get('id') for r in batch] != job['ids']:
            raise ValueError('rewrite_coverage_or_order')
        for row in batch:
            validate(row, by_target[row['id']])
            row_files[row['id']] = str(path.relative_to(c.ROOT))
        rows.extend(batch)
        files[str(path.relative_to(c.OUT))] = m.digest(path)
    if len(rows) != 500 or [r['id'] for r in rows] != [r['id'] for r in targets]:
        raise ValueError('all_500_observed_outputs_required')
    return rows, targets, {r['id']: r for r in briefs}, files, row_files


def candidate(row, target, brief, filename):
    if row['id'] != target['id'] or row['target_sha256'] != target['target_sha256'] or brief['target_sha256'] != target['target_sha256']:
        raise ValueError('candidate_identity')
    followup = FOLLOWUPS.get(row['id'])
    status = 'Candidate · purpose-only reconstruction'
    if followup and followup['level'] in {'target_conflict', 'timing_conflict'}:
        status = 'Candidate · targeted regeneration pending'
    return dict(id=row['id'], paragraph_id=row['id'], parent_target_id=target['parent_target_id'],
        split='training', paper_id=target['paper_id'], authors=target['authors'], style_group=target['style_group'],
        source=row['plain_english'], plain_english=row['plain_english'], source_sha256=m.sha(row['plain_english']),
        target=target['target'], original_paragraph=target['target'], target_sha256=target['target_sha256'],
        raw_response=copy.deepcopy(row), enhanced_brief=copy.deepcopy(brief['brief']), training_ready=False,
        quality=dict(text_pair_status=status, release_status='candidate_only_not_training_ready',
                     automated_assessment='formal_and_surface_checks_with_targeted_reading',
                     independent_semantic_verification=False, followup=followup,
                     raw_model_status=row['rewrite_status'], raw_model_notes=copy.deepcopy(row['review_notes'])),
        provenance=dict(teacher_track='GPT-6', requested_model='GPT-6', requested_effort='High',
            actual_settings_independently_verified=False, input_origin='purpose_only_reconstruction_v1',
            canonical_pdf_sha256=target['canonical_pdf_sha256'], matched_target_slot=row['id'],
            target_kind=target['target_kind'], raw_response_file=filename,
            shared_target_registry=str(c.TARGETS.relative_to(c.ROOT)),
            collection_manifest_sha256=m.digest(c.OUT / 'manifest.json')))


def context_evidence():
    wanted = {('A_BW2006', 17): 'P056', ('B_BW2007', 16): 'P400'}
    found, sources = {}, {}
    for name in ('author_curation_v1', 'author_curation_v2'):
        path = c.ROOT / 'research_runs' / name / 'native_pages.private.jsonl'
        if not path.exists():
            continue
        for line in path.open():
            row = json.loads(line)
            key = (row.get('paper_id'), row.get('pdf_page'))
            if key in wanted and row.get('text') and key not in found:
                found[key] = dict(id=wanted[key], paper_id=key[0], pdf_page=key[1], text=row['text'],
                    method='Previously archived native PDF text; not a new full visual verification.',
                    source_file=str(path.relative_to(c.ROOT)))
                sources[str(path.relative_to(c.ROOT))] = m.digest(path)
    if set(found) != set(wanted):
        raise ValueError('required_return_timing_context_missing')
    return list(found.values()), sources


def prepare_targeted_job(targets):
    folder = OUT / 'targeted_recollection_v1'
    folder.mkdir()
    selected = [r for r in targets if r['id'] in {'P139', 'P400'}]
    payloads = [dict(id=r['id'], target_sha256=r['target_sha256'], original_paragraph=r['target']) for r in selected]
    prompt = (c.DOCS / 'BRIEF_ABSTRACTION_V4.md').read_text()
    addendum = '''\n## Scoped content corrections for these two cases

P139: The user has adjudicated the supplied human paragraph as accepted wording.
Retain its model comparison: bonds command a liquidity premium AND are described
as trading at a discount relative to stocks. Put that stated relative comparison
in the content/anchors. Do not replace it with a price premium or reinterpret it
as a yield statement. Present it as the reported model result, without inventing
an explanation for the comparison or adding a critique of the accepted source.

P400: Verified source section metadata is "Using Sentiment to Explain Current
Returns" (PDF page 16). This paragraph introduces tests relating sentiment changes
and current stock returns, not subsequent-return forecasting. Retain this timing
in the content brief. Treat the section title as context metadata, not prose to
copy into the reconstructed paragraph. The original target text itself is unchanged.

These corrections concern economic content, not wording or a requested sentence
plan. Continue using compact subject notes, purpose labels, factual anchors and
the same optional relation schema. Return both records as briefs_01.jsonl.
'''
    (folder / 'inputs_01.jsonl').write_text(m.jsonl(payloads))
    (folder / 'batch_01.md').write_text(prompt + addendum + '\nOutput data filename: briefs_01.jsonl\nExpected records: 2\n\nINPUTS:\n```jsonl\n' + m.jsonl(payloads) + '```\n')
    for name in ('stage_1_briefs', 'stage_2_rewrites'):
        (folder / 'responses' / name).mkdir(parents=True)
    (folder / 'START_HERE.md').write_text('''# Two targeted upstream corrections

Upload only batch_01.md to GPT-6 with requested High effort. It contains P139
and P400, the original accepted targets and scoped factual/timing instructions.
No original response or human target is edited. No bulk collection is repeated.
Save briefs_01.jsonl in responses/stage_1_briefs/ inside this folder.

After the corrected briefs return, check them and prepare a separate purpose-only
reconstruction request, excluding relations and original paragraphs. Do not supply
the old erroneous rewrites as reconstruction examples. A later version can replace
only these two candidates while preserving the full 500-slot cohort and raw history.
No calls or training are executed by preparing this job.
''')


def prepare():
    if OUT.exists():
        raise ValueError('refuse_overwrite_output_audit')
    rows, targets, briefs, files, row_files = load()
    by_target = {r['id']: r for r in targets}
    prior_path = c.ROOT / 'research_runs/data_scaling_v1/shared_target_repairs_v2/gpt6_candidates_0500.private.jsonl'
    prior = {r['id']: r for r in m.lines(prior_path)}
    diagnostics, candidates, route_a = [], [], []
    for row in rows:
        rid = row['id']
        t, brief = by_target[rid], briefs[rid]
        text = row['plain_english']
        metrics = m.contrast(text, t['target'])
        bc = brief['brief']
        content = ' '.join([bc['topic']] + bc['key_points'] + list(bc['anchors'].values()))
        flags = []
        if row['rewrite_status'] != 'rewritten' or row['review_notes']:
            flags.append('model_review_flag')
        if re.search(r'@A\d+|\bK\d+\b', text):
            flags.append('unexpanded_reference_candidate')
        if re.search(r'\b(brief|provided paragraph|supplied text|source paragraph|not specified|not supplied)\b', text, re.I):
            flags.append('annotation_word_candidate')
        if metrics['word_similarity'] >= .60 or metrics['longest_ordinary_words'] >= 12:
            flags.append('overlap_followup')
        if metrics['word_length_ratio'] < .40:
            flags.append('short_relative_to_target')
        added = sorted(normalized_numerals(text) - normalized_numerals(content))
        if added:
            flags.append('new_numeral_string')
        if rid in FOLLOWUPS:
            flags.append('targeted_adjudication')
        old = prior[rid]
        comparable = old['target_sha256'] == t['target_sha256'] and old['quality']['text_pair_status'] == 'Candidate · fresh brief reconstruction'
        diagnostics.append(dict(id=rid, source_sha256=m.sha(text), target_sha256=t['target_sha256'],
            metrics=metrics, advisory_flags=flags, new_numeral_strings=added,
            prior_fresh_comparable=comparable, prior_metrics=m.contrast(old['source'], t['target']) if comparable else None,
            targeted_assessment=FOLLOWUPS.get(rid), independent_semantic_verification=False))
        candidates.append(candidate(row, t, brief, row_files[rid]))
        encoded = json.dumps(bc, ensure_ascii=False, sort_keys=True)
        route_a.append(dict(id=rid, split='training', paper_id=t['paper_id'], authors=t['authors'], style_group=t['style_group'],
            source=encoded, source_sha256=m.sha(encoded), enhanced_brief=copy.deepcopy(bc), brief_schema_version=4,
            target=t['target'], target_sha256=t['target_sha256'], training_ready=False,
            quality=dict(qualification='candidate_not_independently_verified', followup=FOLLOWUPS.get(rid)),
            provenance=dict(canonical_pdf_sha256=t['canonical_pdf_sha256'], matched_target_slot=rid,
                            brief_collection_manifest_sha256=m.digest(c.OUT / 'manifest.json'))))
    evidence, evidence_sources = context_evidence()
    OUT.mkdir()
    (OUT / 'diagnostics.private.jsonl').write_text(m.jsonl(diagnostics))
    (OUT / 'candidates_0500.private.jsonl').write_text(m.jsonl(candidates))
    (OUT / 'candidates_0200.private.jsonl').write_text(m.jsonl(candidates[:200]))
    (OUT / 'brief_human_candidates_0500.private.jsonl').write_text(m.jsonl(route_a))
    (OUT / 'context_evidence.private.jsonl').write_text(m.jsonl(evidence))
    prepare_targeted_job(targets)
    matched = [r for r in diagnostics if r['prior_fresh_comparable']]
    summary = dict(observed_outputs=500, response_files=17, raw_status_counts=dict(Counter(r['rewrite_status'] for r in rows)),
        prose_candidates=500, brief_human_candidates=500, output_edits=0, records_discarded=0,
        author_group_counts=dict(Counter(r['style_group'] for r in targets)),
        median_metrics={key: statistics.median(r['metrics'][key] for r in diagnostics)
                        for key in ('word_similarity', 'ordinary_shared_fivegram_coverage', 'longest_ordinary_words', 'word_length_ratio')},
        overlap_screen_flags=sum('overlap_followup' in r['advisory_flags'] for r in diagnostics),
        short_relative_to_target=sum('short_relative_to_target' in r['advisory_flags'] for r in diagnostics),
        comparable_prior_fresh_pairs=len(matched),
        prior_fresh_median_word_similarity=statistics.median(r['prior_metrics']['word_similarity'] for r in matched),
        new_matched_median_word_similarity=statistics.median(r['metrics']['word_similarity'] for r in matched),
        lower_word_similarity_than_prior=sum(r['metrics']['word_similarity'] < r['prior_metrics']['word_similarity'] for r in matched),
        higher_word_similarity_than_prior=sum(r['metrics']['word_similarity'] > r['prior_metrics']['word_similarity'] for r in matched),
        targeted_regeneration_ids=['P139', 'P400'], model_flags_adjudicated_nonblocking=['P056', 'P206'],
        scope_watch_ids=['P085', 'P313', 'P329', 'P353'], brief_concern_not_repeated_ids=['P213', 'P233', 'P289'],
        explicit_relation_lists_empty=478, training_ready=False, independent_semantic_verification=False,
        review_method='All-record formal/identity/surface screening; triggered-case and fixed author-stratified reading with targeted native-source context checks.',
        requested_model='GPT-6', requested_effort='High', actual_settings_independently_verified=False,
        limitations=['Not exhaustive claim alignment or independent semantic/style certification.',
            'Low target overlap is a useful contrast diagnostic, not proof of distinctive style, detector reduction or student improvement.',
            'Compression, explanatory additions and moderate scope/emphasis changes are retained; no score-based data purge.',
            'Two confirmed target/context conflicts require focused upstream regeneration before freezing a complete release.'],
        provider_calls=0, training_calls=0, detector_calls=0, reserved_test_examples_read=0)
    m.dump(OUT / 'summary.json', summary)
    dependencies = {str(prior_path.relative_to(c.ROOT)): m.digest(prior_path),
        str((c.OUT / 'stage_2_reconstruction_batches/manifest.json').relative_to(c.ROOT)): m.digest(c.OUT / 'stage_2_reconstruction_batches/manifest.json'),
        str((c.OUT / 'brief_audit_v1/manifest.json').relative_to(c.ROOT)): m.digest(c.OUT / 'brief_audit_v1/manifest.json'),
        str((c.DOCS / 'BRIEF_ABSTRACTION_V4.md').relative_to(c.ROOT)): m.digest(c.DOCS / 'BRIEF_ABSTRACTION_V4.md'),
        **evidence_sources}
    m.dump(OUT / 'manifest.json', dict(builder_sha256=m.digest(Path(__file__)), raw_response_files=files,
        dependencies=dependencies, files={str(p.relative_to(OUT)): m.digest(p) for p in sorted(OUT.rglob('*')) if p.is_file()}))
    return check()


def check():
    rows, targets, briefs, files, row_files = load()
    manifest = m.read(OUT / 'manifest.json')
    if manifest['builder_sha256'] != m.digest(Path(__file__)) or manifest['raw_response_files'] != files:
        raise ValueError('output_audit_source_changed')
    for name, digest in manifest['dependencies'].items():
        if m.digest(c.ROOT / name) != digest:
            raise ValueError('output_audit_dependency_changed')
    for name, digest in manifest['files'].items():
        if m.digest(OUT / name) != digest:
            raise ValueError('output_audit_artifact_changed')
    by_target = {r['id']: r for r in targets}
    actual = m.lines(OUT / 'candidates_0500.private.jsonl')
    expected = [candidate(r, by_target[r['id']], briefs[r['id']], row_files[r['id']]) for r in rows]
    if actual != expected or actual[:200] != m.lines(OUT / 'candidates_0200.private.jsonl'):
        raise ValueError('unchanged_candidate_or_prefix_binding')
    route_a = m.lines(OUT / 'brief_human_candidates_0500.private.jsonl')
    if len(route_a) != 500 or [r['id'] for r in route_a] != [r['id'] for r in rows]:
        raise ValueError('brief_candidate_coverage')
    for r in route_a:
        if r['enhanced_brief'] != briefs[r['id']]['brief'] or json.loads(r['source']) != r['enhanced_brief'] or m.sha(r['source']) != r['source_sha256'] or r['target'] != by_target[r['id']]['target'] or r['training_ready']:
            raise ValueError('brief_candidate_binding')
    return m.read(OUT / 'summary.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'check'])
    args = parser.parse_args()
    print(json.dumps(prepare() if args.action == 'prepare' else check(), indent=2))
