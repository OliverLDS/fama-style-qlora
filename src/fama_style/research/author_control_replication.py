"""Frozen-feature comparison-author replication; CPU-only, private prose outputs.

Reuses the original target anchors, segmentation, filters and feature extractor.
No classifier fitting, feature search, generation, or training is performed.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import statistics
import unicodedata

from . import author_comparison_corpus as corpus
from . import author_comparison_diagnostics as diagnostics
from fama_style.rhetoric.experiment import fresh, write_json, write_lines

PROTOCOL = 'author_control_replication_v1'
FEATURES = ('relative_clause_markers', 'passive_ed_proxy')
PATTERNS = (r'\b(?:which|that|whose|whom)\b',
            r'\b(?:is|are|was|were|be|been)\s+(?:\w+ly\s+)?\w+ed\b')


def folded(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text.lower()) if not unicodedata.combining(c))


def validate_controls(registry, original):
    controls = registry['controls']
    if registry['protocol'] != PROTOCOL or len(controls) != 7:
        raise ValueError('seven_new_controls_required')
    if len({p['paper_id'] for p in controls}) != 7:
        raise ValueError('duplicate_control_paper')
    targets = {p['paper_id']: p for p in original['papers'] if p['label'] == 1}
    if Counter(p['matched_target'] for p in controls) != Counter(targets.keys()):
        raise ValueError('target_coverage')
    excluded = set(original['excluded_author_surnames'])
    excluded.update(folded(a).split()[-1] for p in original['papers'] if p['label'] == 0 for a in p['authors'])
    old_ids = {p['paper_id'] for p in original['papers']}
    seen_authors = set()
    for p in controls:
        if p['paper_id'] in old_ids or p['label'] != 0 or not p['authors']:
            raise ValueError('new_control_metadata')
        names = set(corpus.words(folded(' '.join(p['authors']))))
        if names & excluded:
            raise ValueError('excluded_old_or_project_author')
        authors = {folded(a) for a in p['authors']}
        if authors & seen_authors:
            raise ValueError('reused_new_control_author')
        seen_authors.update(authors)
        if p['topic'] != targets[p['matched_target']]['topic']:
            raise ValueError('topic_mismatch')
        if p['columns'] not in (1, 2) or not p['ocr_pages'] or len(set(p['ocr_pages'])) != len(p['ocr_pages']):
            raise ValueError('layout_or_page_coverage')
    if registry['matching_policy'] != original['paragraph_match_policy']:
        raise ValueError('quality_matching_policy_changed')


def verify_files(directory, manifest_name='manifest.json'):
    manifest = json.loads((directory / manifest_name).read_text())
    for name, digest in manifest['files'].items():
        if corpus.file_sha(directory / name) != digest:
            raise ValueError('frozen_file_changed:' + name)
    return manifest


def frozen_names(original):
    return {w.lower() for p in original['papers'] for a in p['authors'] for w in corpus.words(a)}


def verify_prereg(prereg, extractor_path):
    if prereg['protocol'] != PROTOCOL or prereg['feature_extractor_sha256'] != corpus.file_sha(extractor_path):
        raise ValueError('feature_extractor_changed')
    if [(h['feature'], h['pattern'], h['expected_target_minus_control']) for h in prereg['hypotheses']] != [
            (f, p, 'negative') for f, p in zip(FEATURES, PATTERNS)]:
        raise ValueError('hypotheses_changed')


def remove_duplicates(candidates, original_rows):
    frequencies = Counter(r['text_sha256'] for r in candidates)
    old = {r['text_sha256'] for r in original_rows}
    rejected = [r for r in candidates if frequencies[r['text_sha256']] > 1 or r['text_sha256'] in old]
    excluded = {r['id'] for r in rejected}
    return [r for r in candidates if r['id'] not in excluded], rejected


def build(registry_path, ocr_path, prereg_path, output, lexicon_path=Path('/usr/share/dict/words')):
    root = corpus.ROOT
    original_dir = root / 'data/author_comparison_v1'
    original_manifest = verify_files(original_dir)
    original = json.loads((original_dir / 'registry.private.json').read_text())
    registry = json.loads(registry_path.read_text())
    prereg = json.loads(prereg_path.read_text())
    verify_prereg(prereg, Path(diagnostics.__file__))
    if original_manifest['code_sha256'] != corpus.file_sha(Path(corpus.__file__)):
        raise ValueError('segmentation_filters_changed')
    if original_manifest['lexicon_sha256'] != corpus.file_sha(lexicon_path):
        raise ValueError('lexicon_changed')
    if registry['target_registry_sha256'] != corpus.file_sha(original_dir / 'registry.private.json'):
        raise ValueError('original_registry_changed')
    if corpus.file_sha(root / 'data/fama_corpus_v1/train.jsonl') != corpus.TRAIN_SHA:
        raise ValueError('frozen_training_changed')
    validate_controls(registry, original)
    # Any source replacement must be declared before computing the new features.
    actual_ids = [p['paper_id'] for p in registry['controls']]
    amendments = []
    if actual_ids != prereg['new_control_ids']:
        path = prereg_path.parent / 'source_quality_amendment.private.json'
        amendment = json.loads(path.read_text())
        if amendment['new_control_ids'] != actual_ids or not amendment['before_feature_computation']:
            raise ValueError('undeclared_source_replacement')
        amendments.append({'file': path.name, 'sha256': corpus.file_sha(path)})
    old_rows = corpus.read_lines(original_dir / 'paragraphs.private.jsonl')
    anchors = [r for r in old_rows if r['label'] == 1]
    if len(anchors) != 73:
        raise ValueError('original_73_anchors_required')
    pages = corpus.read_lines(ocr_path)
    if len({(p['paper_id'], p['pdf_page']) for p in pages}) != len(pages):
        raise ValueError('duplicate_ocr_page')
    by_paper = defaultdict(list)
    for p in pages:
        by_paper[p['paper_id']].append(p)
    if set(by_paper) != set(actual_ids):
        raise ValueError('ocr_paper_coverage')
    lexicon = {w.strip().lower() for w in lexicon_path.read_text().splitlines()}
    candidates, rejected, reports, pdf_hashes = [], [], [], {}
    for paper in registry['controls']:
        ps = by_paper[paper['paper_id']]
        if sorted(p['pdf_page'] for p in ps) != sorted(paper['ocr_pages']):
            raise ValueError('ocr_page_coverage')
        if any(p.get('columns', 1) != paper['columns'] for p in ps):
            raise ValueError('ocr_column_layout')
        pdf_hashes[paper['paper_id']] = corpus.file_sha(root / paper['pdf'])
        raw, dropped = corpus.paragraph_candidates(paper, ps)
        counts = Counter()
        accepted = []
        for r in raw:
            r['id'] = 'P' + corpus.sha(paper['paper_id'] + '|' + json.dumps(r['line_refs'], sort_keys=True))[:20]
            reason, text = corpus.quality_reason(r, registry['matching_policy'], lexicon)
            if reason:
                counts[reason] += 1
                rejected.append({**r, 'reason': reason})
                continue
            accepted.append({**r, 'text': text, 'text_sha256': corpus.sha(text), 'word_count': len(corpus.words(text)),
                'label': 0, 'topic': paper['topic'], 'authors': paper['authors'], 'version_year': paper['version_year'],
                'pdf_sha256': pdf_hashes[paper['paper_id']],
                'normalization': ['Unicode NFKC', 'whitespace collapse', 'dictionary-supported line-break joins only'],
                'normalization_events': corpus.normalize_candidate(r['raw_ocr'], lexicon)[1],
                'quality_flags': {'inline_technical_notation': bool(re.search(r'[<>+=ασμ]|\b[A-Za-z]\d\b', text)),
                    'unrecognized_lowercase_words': [w for w in corpus.words(text) if len(w) > 3 and w.islower() and not corpus.supported_word(w, lexicon)]},
                'segmentation_verified': False, 'independent_transcription_verified': False, 'training_ready': False})
        candidates.extend(accepted)
        reports.append({'paper_id': paper['paper_id'], 'candidate_blocks': len(raw), 'eligible_before_duplicates': len(accepted),
                        'rejections': dict(counts), 'dropped_lines': dropped})
    pool, duplicates = remove_duplicates(candidates, old_rows)
    rejected.extend({**r, 'reason': 'exact_duplicate_new_or_original_corpus'} for r in duplicates)
    selected, matches = [], []
    for paper in registry['controls']:
        target = [r for r in anchors if r['paper_id'] == paper['matched_target']]
        control = [r for r in pool if r['paper_id'] == paper['paper_id']]
        # Collect feasible fixed-caliper matches even when the prespecified minimum
        # fails. The failure is retained as an adequacy gate, never waived.
        feasible = corpus.choose_matches(target, control, {**registry['matching_policy'], 'min_pairs_per_topic': 0})
        if not feasible:
            raise ValueError('no_feasible_matches_for_topic')
        for x, y, cost in feasible:
            mid = 'M' + corpus.sha(PROTOCOL + '|' + x['id'] + '|' + y['id'])[:20]
            selected.append({**y, 'match_id': mid})
            matches.append({'match_id': mid, 'topic': paper['topic'], 'target_id': x['id'], 'control_id': y['id'], 'cost': cost,
                'word_ratio': max(x['word_count'], y['word_count']) / min(x['word_count'], y['word_count']),
                'position_difference': abs(x['window_position'] - y['window_position'])})
    matched = {m['target_id'] for m in matches}
    coverage = [{'topic': p['topic'], 'target_paper': p['matched_target'], 'new_control': p['paper_id'],
        'target_anchors': sum(r['paper_id'] == p['matched_target'] for r in anchors),
        'matched_anchors': sum(m['topic'] == p['topic'] for m in matches),
        'meets_prespecified_minimum': sum(m['topic'] == p['topic'] for m in matches) >= registry['matching_policy']['min_pairs_per_topic'],
        'unmatched_target_ids': [r['id'] for r in anchors if r['paper_id'] == p['matched_target'] and r['id'] not in matched]}
        for p in registry['controls']]
    summary = {'protocol': PROTOCOL, 'status': 'new_control_corpus_built_before_feature_analysis',
        'adequate_topic_coverage': all(p['meets_prespecified_minimum'] for p in coverage),
        'original_target_anchors': len(anchors), 'new_control_paragraphs': len(selected), 'new_control_papers': 7,
        'topics': 7, 'ocr_pages': len(pages), 'eligible_control_pool': len(pool), 'exact_duplicate_candidates_excluded': len(duplicates),
        'per_paper': reports, 'coverage': coverage, 'matched_word_ratio_max': max(m['word_ratio'] for m in matches),
        'target_anchor_selection_changed': False, 'rhetorical_scores_used_for_selection': False,
        'training_ready_rows': 0, 'validation_rows_read': 0, 'reserved_test_rows_read': 0, 'gpu_jobs_started': 0, 'language_model_calls': 0}
    fresh(output)
    write_lines(output / 'original_target_anchors.private.jsonl', anchors)
    write_lines(output / 'new_controls.private.jsonl', selected)
    write_lines(output / 'eligible_pool.private.jsonl', pool)
    write_lines(output / 'rejected_blocks.private.jsonl', rejected)
    write_json(output / 'matches.private.json', matches)
    write_json(output / 'registry.private.json', registry)
    write_json(output / 'summary.json', summary)
    write_json(output / 'manifest.json', {'protocol': PROTOCOL, 'code_sha256': corpus.file_sha(Path(__file__)),
        'feature_extractor_sha256': corpus.file_sha(Path(diagnostics.__file__)), 'segmenter_sha256': corpus.file_sha(Path(corpus.__file__)),
        'ocr_engine_sha256': corpus.file_sha(root / 'scripts/data/extract_comparison_ocr.swift'),
        'preregistration_sha256': corpus.file_sha(prereg_path), 'amendments': amendments,
        'original_manifest_sha256': corpus.file_sha(original_dir / 'manifest.json'), 'registry_sha256': corpus.file_sha(registry_path),
        'ocr_sha256': corpus.file_sha(ocr_path), 'pdf_sha256': pdf_hashes, 'lexicon_sha256': corpus.file_sha(lexicon_path),
        'frozen_train_sha256': corpus.TRAIN_SHA, 'files': {p.name: corpus.file_sha(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({'new_controls': len(selected), 'target_anchors': len(anchors), 'coverage': coverage}))


def summarize_features(feature_rows, matches):
    """Paper-weighted topic differences; all original anchors primary, matched subset sensitivity."""
    matched = {m['target_id'] for m in matches}
    topics = sorted({r['topic'] for r in feature_rows})
    result = {}
    for feature in FEATURES:
        per_topic = []
        for topic in topics:
            rows = [r for r in feature_rows if r['topic'] == topic]
            a = [r for r in rows if r['group'] == 'target']
            old = [r for r in rows if r['group'] == 'old_control']
            new = [r for r in rows if r['group'] == 'new_control']
            subset = [r for r in a if r['id'] in matched]
            if not all((a, old, new, subset)):
                raise ValueError('missing_topic_group_or_matched_anchor')
            mean = lambda rs: statistics.mean(r['features'][feature] for r in rs)
            target, first, second, sensitivity = map(mean, (a, old, new, subset))
            per_topic.append({'topic': topic, 'target_mean': target, 'old_control_mean': first, 'new_control_mean': second,
                'target_minus_old': target - first, 'target_minus_new': target - second,
                'matched_anchor_subset_minus_new': sensitivity - second,
                'target_minus_equal_weight_control_paper_mean': target - (first + second) / 2,
                'target_n': len(a), 'old_control_n': len(old), 'new_control_n': len(new), 'matched_target_n': len(subset)})
        differences = [r['target_minus_new'] for r in per_topic]
        subset_diffs = [r['matched_anchor_subset_minus_new'] for r in per_topic]
        negative = sum(d < 0 for d in differences)
        median = statistics.median(differences)
        result[feature] = {'unit': 'occurrences per 100 metadata-masked words; paragraph means within paper',
            'new_control_negative_topics': negative, 'topics': len(topics), 'new_control_median_difference': median,
            'meets_preregistered_direction_criterion': len(topics) == 7 and negative >= 6 and median < 0,
            'old_control_negative_topics': sum(r['target_minus_old'] < 0 for r in per_topic),
            'matched_subset_negative_topics': sum(d < 0 for d in subset_diffs),
            'matched_subset_median_difference': statistics.median(subset_diffs),
            'equal_control_paper_weight_negative_topics': sum(r['target_minus_equal_weight_control_paper_mean'] < 0 for r in per_topic),
            'per_topic': per_topic}
    return result


def analyze(corpus_dir, output):
    root = corpus.ROOT
    manifest = verify_files(corpus_dir)
    if manifest['feature_extractor_sha256'] != corpus.file_sha(Path(diagnostics.__file__)):
        raise ValueError('feature_extractor_changed')
    old_dir = root / 'data/author_comparison_v1'
    verify_files(old_dir)
    if manifest['original_manifest_sha256'] != corpus.file_sha(old_dir / 'manifest.json'):
        raise ValueError('original_corpus_changed')
    old = json.loads((old_dir / 'registry.private.json').read_text())
    new = json.loads((corpus_dir / 'registry.private.json').read_text())
    names = frozen_names(old)
    papers = {p['paper_id']: p for p in old['papers'] + new['controls']}
    original = corpus.read_lines(old_dir / 'paragraphs.private.jsonl')
    anchors = corpus.read_lines(corpus_dir / 'original_target_anchors.private.jsonl')
    if anchors != [r for r in original if r['label'] == 1]:
        raise ValueError('target_anchors_changed')
    source = original + corpus.read_lines(corpus_dir / 'new_controls.private.jsonl')
    old_features = {r['id']: r for r in corpus.read_lines(root / 'research_runs/author_comparison_v1/diagnostics_v1/features.private.jsonl')}
    feature_rows = []
    for r in source:
        f = diagnostics.feature_values({**r, 'columns': papers[r['paper_id']]['columns']}, names)
        if r['id'] in old_features:
            stored = old_features[r['id']]
            if any(f[name] != stored['features'][name] for name in FEATURES):
                raise ValueError('original_features_changed')
        feature_rows.append({'id': r['id'], 'paper_id': r['paper_id'], 'topic': r['topic'],
            'group': 'target' if r['label'] == 1 else ('old_control' if r['paper_id'] in {p['paper_id'] for p in old['papers']} else 'new_control'),
            'features': {name: f[name] for name in FEATURES}})
    matches = json.loads((corpus_dir / 'matches.private.json').read_text())
    corpus_summary = json.loads((corpus_dir / 'summary.json').read_text())
    hypotheses = summarize_features(feature_rows, matches)
    for result in hypotheses.values():
        result['adequate_topic_coverage'] = corpus_summary['adequate_topic_coverage']
        result['replication_supported'] = result['meets_preregistered_direction_criterion'] and corpus_summary['adequate_topic_coverage']
    results = {'protocol': PROTOCOL, 'status': 'executed_frozen_feature_cpu_replication',
        'new_control_papers': 7, 'reused_target_papers': 7, 'old_control_papers': 7,
        'original_target_anchors': len(anchors), 'new_control_paragraphs': len(source) - len(original),
        'coverage': corpus_summary['coverage'], 'adequate_topic_coverage': corpus_summary['adequate_topic_coverage'],
        'hypotheses': hypotheses,
        'interpretation_limit': 'Sensitivity to new comparison authors; targets reused. Proxy counts are neither verified syntax nor an author-style classifier.',
        'fitted_classifiers': 0, 'feature_search': False, 'validation_rows_read': 0, 'reserved_test_rows_read': 0,
        'language_model_calls': 0, 'gpu_jobs_started': 0, 'adapter_training_called': False}
    fresh(output)
    write_lines(output / 'features.private.jsonl', feature_rows)
    write_json(output / 'summary.json', results)
    write_json(output / 'manifest.json', {'protocol': PROTOCOL, 'code_sha256': corpus.file_sha(Path(__file__)),
        'corpus_manifest_sha256': corpus.file_sha(corpus_dir / 'manifest.json'),
        'files': {p.name: corpus.file_sha(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({name: {k: v for k, v in results['hypotheses'][name].items() if k != 'per_topic'} for name in FEATURES}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    for flag in ('registry', 'ocr', 'prereg', 'output'):
        b.add_argument('--' + flag, type=Path, required=True)
    a = sub.add_parser('analyze')
    a.add_argument('--corpus', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.command == 'build':
        build(args.registry, args.ocr, args.prereg, args.output)
    else:
        analyze(args.corpus, args.output)


if __name__ == '__main__':
    main()
