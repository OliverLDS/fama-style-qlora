"""Import manual GPT-6 counterparts and replay a private versioned quality audit.

CPU only. Lexical alarms prioritize assessment; they do not establish semantic
correctness. This script stages candidates and never exports training-ready data.
"""
import argparse
from collections import Counter
import copy
from difflib import SequenceMatcher
import importlib.util
import json
from pathlib import Path
import re
import statistics

SPEC = importlib.util.spec_from_file_location('gpt6_preparation', Path(__file__).with_name('prepare_gpt6_track_500.py'))
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)
RUN = p.DEFAULT_OUT
OUT = RUN / 'response_quality_v1'


def numbers(text):
    # Deliberately lexical: words, repeated citations and typography are adjudicated.
    return Counter(re.findall(r'(?<![\w])\d+(?:[.,]\d+)*(?:%|/\d+)?', text))


def metrics(source, target):
    a = re.findall(r'\w+', source.casefold()); b = re.findall(r'\w+', target.casefold())
    sn = numbers(source); tn = numbers(target)
    ratio = len(source.split()) / len(target.split())
    flags = []
    if sn != tn:
        flags.append('numeric_surface_difference')
    if ratio < .75:
        flags.append('shorter_rewrite_screen')
    return dict(source_words=len(source.split()), target_words=len(target.split()), word_length_ratio=ratio,
                token_sequence_similarity=SequenceMatcher(None, a, b, autojunk=False).ratio(),
                exact_normalized_copy=p.norm(source) == p.norm(target),
                missing_numeric_tokens=dict(tn-sn), extra_numeric_tokens=dict(sn-tn), flags=flags)


def validate_response(row, expected):
    if not isinstance(row, dict) or set(row) != set(p.SCHEMA['required']):
        raise ValueError('response_fields')
    if row['id'] != expected['id'] or row['target_sha256'] != expected['target_sha256']:
        raise ValueError('response_identity_or_target_hash')
    if not isinstance(row['plain_english'], str) or not row['plain_english'].strip():
        raise ValueError('empty_source')
    if row['rewrite_status'] not in p.SCHEMA['properties']['rewrite_status']['enum']:
        raise ValueError('rewrite_status')
    if not isinstance(row['review_notes'], list) or not all(isinstance(n, str) for n in row['review_notes']):
        raise ValueError('review_notes')
    if row['rewrite_status'] == 'pending_review' and not row['review_notes']:
        raise ValueError('unexplained_pending_review')


def load_responses(run=RUN):
    p.validate(run)
    by_id = {}; files = {}
    for job in p.read(run / 'jobs.json'):
        inputs = p.lines(run / 'manual_batches_v1' / f"inputs_{job['batch']:02d}.jsonl")
        path = run / job['response']; rows = p.lines(path)
        if len(rows) != len(inputs):
            raise ValueError('batch_record_count')
        files[job['response']] = p.digest(path)
        for row, expected in zip(rows, inputs):
            validate_response(row, expected)
            if row['id'] in by_id:
                raise ValueError('duplicate_response_id')
            by_id[row['id']] = dict(row, raw_file=job['response'], raw_file_sha256=files[job['response']],
                                   raw_source_sha256=p.sha(row['plain_english']))
    if len(by_id) != 227:
        raise ValueError('expected_227_missing_counterparts')
    return by_id, files


def apply_patches(source, patches):
    result = source
    for patch in patches:
        before, after = patch['before'], patch['after']
        if not isinstance(before, str) or not before or result.count(before) != 1 or not isinstance(after, str):
            raise ValueError('nonunique_or_invalid_patch')
        if len(before.split()) > max(25, .6 * len(source.split())) or before == source:
            raise ValueError('whole_paragraph_or_large_replacement')
        if not patch.get('reason'):
            raise ValueError('patch_reason')
        result = result.replace(before, after, 1)
    return result


def screen():
    if OUT.exists():
        raise ValueError('refuse_overwrite_quality_import')
    responses, files = load_responses()
    parents = p.lines(p.ROOT / p.MASTER)
    out = []
    for rid, row in responses.items():
        parent = parents[int(rid[1:])-1]
        result = dict(row, mixed_id=parent['id'], style_group=parent['style_group'],
                      original_paragraph=parent['target'], metrics=metrics(row['plain_english'], parent['target']))
        if row['rewrite_status'] == 'pending_review':
            result['metrics']['flags'].append('generator_flagged_uncertainty')
        out.append(result)
    OUT.mkdir()
    (OUT / 'screening.private.jsonl').write_text(p.jsonl(out))
    p.write(OUT / 'import_manifest.json', dict(protocol='gpt6_track_0500_quality_v1',
        preparation_manifest_sha256=p.digest(RUN / 'manifest.json'), response_files=files,
        screening_sha256=p.digest(OUT / 'screening.private.jsonl'),
        auditor_sha256=p.digest(Path(__file__)), provider_calls=0, training_calls=0))
    print(json.dumps(dict(records=len(out), priority_screen_records=sum(bool(r['metrics']['flags']) for r in out),
                          reported_status_counts=dict(Counter(r['rewrite_status'] for r in out)))))


def check_import():
    manifest = p.read(OUT / 'import_manifest.json')
    responses, files = load_responses()
    if files != manifest['response_files'] or p.digest(OUT / 'screening.private.jsonl') != manifest['screening_sha256']:
        raise ValueError('import_changed')
    if p.digest(RUN / 'manifest.json') != manifest['preparation_manifest_sha256'] or p.digest(Path(__file__)) != manifest['auditor_sha256']:
        raise ValueError('audit_dependency_changed')
    return responses


def adjudicate():
    responses = check_import()
    if (OUT / 'quality_manifest.json').exists():
        raise ValueError('refuse_overwrite_quality_adjudication')
    decisions = p.lines(OUT / 'adjudications.private.jsonl'); by = {r['id']: r for r in decisions}
    if len(by) != len(decisions) or set(by) != set(responses):
        raise ValueError('adjudication_coverage')
    checked = []; patched = {}; targets = {}; unresolved = []
    screened = {r['id']: r for r in p.lines(OUT / 'screening.private.jsonl')}
    for rid, raw in responses.items():
        d = by[rid]; row = screened[rid]
        if d['raw_source_sha256'] != raw['raw_source_sha256'] or d['target_sha256'] != raw['target_sha256']:
            raise ValueError('decision_text_binding')
        if d['decision'] not in {'accept_automated', 'repair_automated', 'target_review_pending'} or not d['reason']:
            raise ValueError('decision_schema')
        if bool(d['patches']) != (d['decision'] == 'repair_automated'):
            raise ValueError('patch_decision_mismatch')
        if row['metrics']['flags'] and not d['screen_adjudication']:
            raise ValueError('unexplained_screen_flag')
        final = apply_patches(raw['plain_english'], d['patches'])
        if d['decision'] == 'target_review_pending':
            unresolved.append(rid)
        patched[rid] = final
        checked.append(dict(row, final_source=final, source_sha256=p.sha(final),
                            adjudication=d, independent_semantic_verification=False))
    parents = p.lines(p.ROOT / p.MASTER)
    reused = {r['id']: r for r in p.lines(RUN / 'reused_pairs.private.jsonl')}
    lineage = p.lines(RUN / 'target_lineage.private.jsonl'); candidates = []
    for parent, entry in zip(parents, lineage):
        rid = entry['id']; row = copy.deepcopy(parent)
        source = reused[rid]['source'] if rid in reused else patched[rid]
        row.update(id=parent['id']+'::gpt6_track_v1', paragraph_id=parent['id']+'::gpt6_track_v1',
                   source=source, plain_english=source, source_sha256=p.sha(source), training_ready=False)
        row['provenance'] = dict(teacher_track='GPT-6', matched_mixed_id=parent['id'],
            parent_mixed_sha256=p.digest(p.ROOT / p.MASTER), previous_pair_provenance=parent['provenance'],
            canonical_pdf_sha256=entry['canonical_pdf_sha256'],
            input_origin='accepted_historical_reuse' if rid in reused else 'manual_GPT6_counterpart',
            requested_model=None if rid in reused else 'GPT-6',
            requested_thinking_effort=None if rid in reused else 'High',
            actual_effort_independently_verified=False,
            generation_evidence=reused[rid] if rid in reused else {k:responses[rid][k] for k in ['raw_file', 'raw_file_sha256', 'raw_source_sha256', 'rewrite_status', 'review_notes']},
            source_repair_author='Codex' if rid in by and by[rid]['patches'] else None,
            quality_decision=None if rid in reused else by[rid]['decision'],
            candidate_only=True, independent_semantic_verification=False)
        row['quality'] = dict(parent['quality'], release_status='candidate_only_not_training_ready',
                              target_review_pending=rid in unresolved)
        if row['target'] != parent['target'] or row['target_sha256'] != parent['target_sha256']:
            raise ValueError('frozen_target_changed')
        candidates.append(row)
    if len(candidates) != 500 or len({r['parent_target_id'] for r in candidates}) != 500:
        raise ValueError('candidate_identity')
    (OUT / 'reviewed_responses.private.jsonl').write_text(p.jsonl(checked))
    (OUT / 'candidates_0500.private.jsonl').write_text(p.jsonl(candidates))
    (OUT / 'candidates_0200.private.jsonl').write_text(p.jsonl(candidates[:200]))
    similarities = [r['metrics']['token_sequence_similarity'] for r in checked]
    summary = dict(protocol='gpt6_track_0500_quality_v1', response_files=8, new_responses=227,
        reused_inputs_unchanged=273, candidate_pairs=500,
        schema_and_target_bindings_valid=True, all_requested_targets_retained=True,
        decision_counts=dict(Counter(d['decision'] for d in decisions)),
        repaired_input_ids=[d['id'] for d in decisions if d['patches']],
        patch_count=sum(len(d['patches']) for d in decisions), target_review_pending_ids=unresolved,
        raw_reported_status_counts=dict(Counter(r['rewrite_status'] for r in responses.values())),
        screening_flags=dict(Counter(f for r in checked for f in r['metrics']['flags'])),
        median_raw_source_target_similarity=statistics.median(similarities),
        median_raw_word_length_ratio=statistics.median(r['metrics']['word_length_ratio'] for r in checked),
        exact_normalized_raw_copies=sum(r['metrics']['exact_normalized_copy'] for r in checked),
        unique_normalized_new_sources=len({p.norm(r['plain_english']) for r in responses.values()}),
        nested_200_byte_prefix=True, training_ready_release=False, provider_calls=0, training_calls=0,
        limitations=['Automated text/difference assessment, not independent human or model-based semantic certification.',
                    'Numeric surfaces and length flags prioritize review; they are not rejection rules.',
                    'Historical reused inputs retain earlier acceptance and provenance; not freshly re-reviewed here.',
                    'Actual per-batch UI thinking effort is not independently observed.',
                    'Target issues are inherited from the frozen mixed cohort and affect both matched tracks.'])
    p.write(OUT / 'summary.json', summary)
    p.write(OUT / 'quality_manifest.json', dict(import_manifest_sha256=p.digest(OUT / 'import_manifest.json'),
        files={str(f.relative_to(OUT)):p.digest(f) for f in sorted(OUT.rglob('*')) if f.is_file()}))
    print(json.dumps(summary, indent=2))


def check():
    check_import()
    manifest = p.read(OUT / 'quality_manifest.json')
    if p.digest(OUT / 'import_manifest.json') != manifest['import_manifest_sha256']:
        raise ValueError('import_manifest_changed')
    for name, h in manifest['files'].items():
        if p.digest(OUT / name) != h:
            raise ValueError('quality_artifact_changed')
    large = OUT / 'candidates_0500.private.jsonl'; small = OUT / 'candidates_0200.private.jsonl'
    if not large.read_bytes().startswith(small.read_bytes()) or len(p.lines(small)) != 200:
        raise ValueError('nested_candidate_prefix')
    print('Checked: response hashes, audit replay bindings, 500 target identities, and nested 200 candidates; no training release.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['screen', 'adjudicate', 'check'])
    args = parser.parse_args()
    {'screen':screen, 'adjudicate':adjudicate, 'check':check}[args.action]()
