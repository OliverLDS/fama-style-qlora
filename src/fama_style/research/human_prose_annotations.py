"""Prepare private pre-2020 economics passages and manual annotation packets.

No generation, detector optimization, ML imports, model downloads or training.
Original corpora stay unchanged; every output package must use a fresh directory.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import unicodedata

from . import author_comparison_corpus as corpus
from .author_control_replication import verify_files
from fama_style.rhetoric.experiment import fresh, write_json, write_lines

ROOT = corpus.ROOT
PROTOCOL = 'human_prose_annotation_v1'
SEED = '20261028'
CATALOG = ROOT / 'research_runs/taxonomy_discovery_v1/stage1_audit_v1/stage2_ready/batch_A/taxonomy_categories.json'
SPLITS = ('training', 'development', 'reserved_test')
EXCLUDED = {'fama', 'french', 'jensen', 'blume'}


def author_key(name):
    return re.sub(r'[^a-z]', '', ''.join(c for c in unicodedata.normalize('NFKD', name.lower()) if not unicodedata.combining(c)))


def validate_registry(registry):
    papers = registry['papers']
    if registry['protocol'] != PROTOCOL or len(papers) != 20 or len({p['paper_id'] for p in papers}) != 20:
        raise ValueError('twenty_unique_papers_required')
    if registry['publication_cutoff_exclusive_year'] != 2020:
        raise ValueError('frozen_pre_2020_cutoff')
    if registry['paragraphs_per_paper'] != 5:
        raise ValueError('five_paragraphs_per_paper_required')
    authors = {}
    for p in papers:
        year = p['publication_year']
        if not isinstance(year, int) or isinstance(year, bool) or not 1800 <= year < 2020:
            raise ValueError('publication_after_cutoff_or_missing')
        if not p.get('publication_year_checked') or not p.get('publication_evidence_url', '').startswith('https://'):
            raise ValueError('publication_evidence_required')
        if p['split'] not in SPLITS or not p['authors']:
            raise ValueError('split_or_authors_missing')
        for name in p['authors']:
            tokens = set(corpus.words(unicodedata.normalize('NFKD', name.lower())))
            if tokens & EXCLUDED:
                raise ValueError('project_author_excluded')
            key = author_key(name)
            if key in authors and authors[key] != p['split']:
                raise ValueError('author_crosses_split')
            authors[key] = p['split']
    expected = {'training': 12, 'development': 4, 'reserved_test': 4}
    if dict(Counter(p['split'] for p in papers)) != expected:
        raise ValueError('frozen_12_4_4_paper_split_required')


def output_schema(taxonomy_version, taxonomy_sha):
    string = {'type': 'string'}
    nullable = {'type': ['string', 'null']}
    arr = lambda item: {'type': 'array', 'items': item}
    obj = lambda props: {'type': 'object', 'properties': props, 'required': list(props), 'additionalProperties': False}
    enum = lambda values: {'type': 'string', 'enum': values.split('|')}
    ids = lambda prefix: arr({'type': 'string', 'pattern': '^' + prefix + '[1-9][0-9]*$'})
    evidence = arr(obj({'quote': {'type': 'string', 'minLength': 1}, 'occurrence': {'type': 'integer', 'minimum': 1}}))
    alignment = enum('preserved|omitted|changed|uncertain')
    observation = obj({'canonical_id': nullable, 'assignment_status': enum('assigned|unresolved|not_covered'),
        'candidate_ids': arr(string), 'dimension': string, 'value': enum('present|absent|unresolved'),
        'evidence': evidence, 'claim_ids': ids('C'), 'sentence_ids': ids('S'), 'relation_ids': ids('R'), 'note': string})
    schema = obj({'sentences': arr(obj({'sentence_id': {'type': 'string', 'pattern': '^S[1-9][0-9]*$'},
        'exact_text': {'type': 'string', 'minLength': 1}, 'claim_ids': ids('C')})),
        'paragraph_structure': arr(observation), 'sentence_structure': arr(observation), 'language_features': arr(observation)})
    ledger = obj({'entities': arr(obj({'entity_id': {'type': 'string', 'pattern': '^E[1-9][0-9]*$'},
        'label': string, 'kind': enum('actor|institution|variable|concept|other'), 'roles': arr(string), 'original_evidence': evidence})),
        'claims': arr(obj({'claim_id': {'type': 'string', 'pattern': '^C[1-9][0-9]*$'}, 'proposition': string,
        'actor_ids': ids('E'), 'other_entity_ids': ids('E'), 'epistemic_status': enum('finding|hypothesis|expectation|definition|assertion|unresolved'),
        'conditions': arr(string), 'quantities': arr(obj({'expression': string, 'scope': string})),
        'negation': arr(string), 'modality': arr(string), 'quantifiers': arr(string), 'comparisons': arr(string),
        'attributions': arr(string), 'original_evidence': evidence, 'rewrite_evidence': evidence, 'alignment_status': alignment})),
        'relations': arr(obj({'relation_id': {'type': 'string', 'pattern': '^R[1-9][0-9]*$'},
        'kind': enum('causal|conditional|comparison|association|explanatory|logical_implication|temporal|other|uncertain'),
        'from_claim_ids': ids('C'), 'to_claim_ids': ids('C'), 'direction_and_scope': string,
        'original_evidence': evidence, 'rewrite_evidence': evidence, 'alignment_status': alignment})),
        'unmatched_source_claims': arr(obj({'proposition': string, 'rewrite_evidence': evidence, 'reason': string}))})
    result = obj({'annotation_version': {'const': PROTOCOL}, 'batch_id': string, 'paragraph_id': string,
        'original_paragraph': {'type': 'string', 'minLength': 1}, 'plain_english': nullable,
        'rewrite_status': enum('simplified|already_clear|pending_review'), 'rewrite_notes': string,
        'source_quality': obj({'status': enum('usable|uncertain|unusable'), 'issues': arr(string), 'needs_external_context': {'type': 'boolean'}}),
        'semantic_ledger': ledger,
        'source_schema': {'anyOf': [schema, {'type': 'null'}]}, 'target_schema': {'anyOf': [schema, {'type': 'null'}]},
        'transformations': arr(obj({'operation': enum('KEEP|MERGE|SPLIT|REORDER|MOVE_QUALIFICATION|CHANGE_SUBJECT|REPACK_CLAUSES|ADJUST_LEXIS'),
        'claim_ids': ids('C'), 'source_sentence_ids': ids('S'), 'target_sentence_ids': ids('S'),
        'relation_ids': ids('R'), 'explanation': string, 'uncertain': {'type': 'boolean'}})),
        'taxonomy_version': {'const': taxonomy_version}, 'taxonomy_sha256': {'const': taxonomy_sha},
        'needs_review': {'type': 'boolean'}, 'review_notes': arr(string),
        'provenance': obj({'visible_model_label': nullable, 'thinking_setting': nullable, 'annotation_date': nullable,
        'paired_context': {'const': True}, 'independently_verified': {'const': False}})})
    return {'$schema': 'https://json-schema.org/draft/2020-12/schema', '$id': 'urn:fama-style:human-prose-annotation:v1', **result}


def compact_catalog(full):
    # Keep all category definitions, applicability and semantic restrictions.
    # Remove redundant boilerplate by exposing shared rules once; no ID is merged.
    array_keys = ('inclusion_rules', 'exclusion_rules', 'evidence_requirements', 'preconditions', 'prohibited_effects')
    common = {}
    for key in array_keys:
        counts = Counter(v for x in full['canonical_items'] for v in x.get(key, []))
        common[key] = sorted(v for v, n in counts.items() if n >= 2)
    items = []
    for x in full['canonical_items']:
        entry = {k: v for k, v in x.items() if k not in array_keys}
        for key in array_keys:
            # Indexed references reproduce every original rule when expanded.
            entry[key] = [({'shared_rule': common[key].index(v)} if v in common[key] else {'rule': v}) for v in x.get(key, [])]
        items.append(entry)
    return {'taxonomy_version': full['taxonomy_version'], 'taxonomy_sha256': full['taxonomy_sha256'],
        'status': full['status'], 'format': 'lossless_shared_rule_view_v1',
        'instructions': 'Each shared_rule is a zero-based index into shared_rules for the SAME field. Definitions, categories and restrictions are unchanged. Not a validated universal taxonomy.',
        'shared_rules': common, 'canonical_items': items}


def expand_catalog(compact):
    items = []
    for x in compact['canonical_items']:
        row = dict(x)
        for field in compact['shared_rules']:
            row[field] = [compact['shared_rules'][field][r['shared_rule']] if 'shared_rule' in r else r['rule'] for r in row[field]]
        items.append(row)
    return items


def prose_reason(row):
    text = row['text']
    if not 40 <= row['word_count'] <= 300:
        return 'annotation_word_budget_40_300'
    if not text[:1].isupper() or not corpus.terminal(text):
        return 'incomplete_boundary'
    if text.count('(') != text.count(')') or text.count('[') != text.count(']'):
        return 'unbalanced_delimiters'
    if re.search(r'\b(?:lemma|theorem|corollary|proof|qed)\b', text, re.I):
        return 'formal_result_or_proof'
    if re.search(r'\b(?:equation|eq\.)\s*\(?\d', text, re.I):
        return 'depends_on_displayed_equation'
    if re.search(r'[=∑∫�|<>]', text):
        return 'formal_notation_or_corruption'
    return None


def select_five(rows, excluded_ids):
    pool = [r for r in rows if not prose_reason(r) and r['id'] not in excluded_ids]
    if len(pool) < 5:
        raise ValueError('insufficient_full_prose:' + (rows[0]['paper_id'] if rows else 'empty') + ':' + str(len(pool)))
    ordered = sorted(pool, key=lambda r: (r['word_count'], r['id']))
    chosen = []
    for i in range(5):
        group = ordered[i * len(ordered) // 5:(i + 1) * len(ordered) // 5]
        middle = statistics.median(r['word_count'] for r in group)
        def rank(r):
            # Quality/context and length, never detector or authorship feature scores.
            context = bool(re.match(r'^(?:These|Those|This|It|They|Such|Hence|Thus|Therefore)\b', r['text']))
            unknown = len(r.get('quality_flags', {}).get('unrecognized_lowercase_words', [])) / max(1, r['word_count'])
            return (context, unknown, abs(r['word_count'] - middle), r['id'])
        chosen.append(min(group, key=rank))
    return chosen


def collect_pool(registry, ocr_path, lexicon_path):
    lexicon = {w.strip().lower() for w in lexicon_path.read_text().splitlines()}
    imported = {}
    for directory in ('author_comparison_v1', 'author_control_replication_v1'):
        verify_files(ROOT / 'data' / directory)
        for row in corpus.read_lines(ROOT / 'data' / directory / 'eligible_pool.private.jsonl'):
            if row['label'] == 0:
                imported.setdefault(row['paper_id'], []).append(row)
    pages = corpus.read_lines(ocr_path)
    reports, result = [], []
    if len({(p['paper_id'], p['pdf_page']) for p in pages}) != len(pages):
        raise ValueError('duplicate_ocr_page')
    for paper in registry['papers']:
        rows = list(imported.get(paper['paper_id'], [])) if paper['pool_origin'] != 'new_local_ocr' else []
        if paper['pool_origin'] == 'new_local_ocr' or paper.get('additional_ocr_pages'):
            ps = [p for p in pages if p['paper_id'] == paper['paper_id']]
            if sorted(p['pdf_page'] for p in ps) != paper.get('additional_ocr_pages', paper['ocr_pages']):
                raise ValueError('ocr_page_coverage')
            if any(p['columns'] != paper['columns'] for p in ps):
                raise ValueError('ocr_layout')
            raw, _ = corpus.paragraph_candidates(paper, ps)
            for r in raw:
                reason, text = corpus.quality_reason(r, {'word_range': [40, 300]}, lexicon)
                if reason:
                    continue
                rows.append({**r, 'id': 'P' + corpus.sha(paper['paper_id'] + '|' + json.dumps(r['line_refs'], sort_keys=True))[:20],
                    'text': text, 'text_sha256': corpus.sha(text), 'word_count': len(corpus.words(text)),
                    'normalization_events': corpus.normalize_candidate(r['raw_ocr'], lexicon)[1],
                    'quality_flags': {'unrecognized_lowercase_words': [w for w in corpus.words(text) if len(w) > 3 and w.islower() and not corpus.supported_word(w, lexicon)]},
                    'segmentation_verified': False, 'independent_transcription_verified': False})
        result.extend(rows)
        reports.append({'paper_id': paper['paper_id'], 'eligible_upstream': len(rows),
            'annotation_filter_reasons': dict(Counter(prose_reason(r) or 'eligible' for r in rows))})
    frequencies = Counter(r['text_sha256'] for r in result)
    return [r for r in result if frequencies[r['text_sha256']] == 1], reports


def prepare(registry_path, ocr_path, output, exclusions_path=None, preview=False, selection_path=None, repairs_path=None):
    registry = json.loads(registry_path.read_text())
    validate_registry(registry)
    lexicon_path = Path('/usr/share/dict/words')
    pool, reports = collect_pool(registry, ocr_path, lexicon_path)
    excluded = json.loads(exclusions_path.read_text()) if exclusions_path else []
    excluded_ids = {r['source_id'] for r in excluded}
    selection = json.loads(selection_path.read_text()) if selection_path else {}
    repairs = json.loads(repairs_path.read_text()) if repairs_path else []
    by_source = {r['id']: r for r in pool}
    for repair in repairs:
        row = by_source[repair['source_id']]
        if corpus.sha(row['text']) != repair['before_sha256'] or repair['kind'] != 'extraction_only':
            raise ValueError('repair_source_changed_or_not_extraction')
        if not repair['evidence_file'] or not repair['verification_note']:
            raise ValueError('repair_evidence_required')
        if corpus.file_sha(ROOT / repair['evidence_file']) != repair['evidence_sha256']:
            raise ValueError('repair_evidence_changed')
        row['upstream_text_sha256'] = corpus.sha(row['text'])
        row['text'] = repair['after_text']
        row['text_sha256'] = corpus.sha(row['text'])
        row['word_count'] = len(corpus.words(row['text']))
    selected, keys = [], []
    papers = sorted(registry['papers'], key=lambda p: (SPLITS.index(p['split']), p['paper_id']))
    for paper in papers:
        if corpus.file_sha(ROOT / paper['pdf']) != paper['pdf_sha256']:
            raise ValueError('source_pdf_changed')
        candidates = [r for r in pool if r['paper_id'] == paper['paper_id']]
        if selection:
            ids = selection['papers'][paper['paper_id']]['source_ids']
            if len(ids) != 5 or len(set(ids)) != 5:
                raise ValueError('review_requires_five_unique_rows')
            chosen = [by_source[x] for x in ids]
            if any(r['paper_id'] != paper['paper_id'] or r['id'] in excluded_ids or prose_reason(r) for r in chosen):
                raise ValueError('review_selected_ineligible_row')
        else:
            chosen = select_five(candidates, excluded_ids)
        for row in chosen:
            pid = 'HP_' + corpus.sha(SEED + '|' + row['id'])[:16]
            selected.append({'paragraph_id': pid, 'text': row['text']})
            keys.append({'paragraph_id': pid, 'source_id': row['id'], 'paper_id': paper['paper_id'], 'split': paper['split'],
                'original_sha256': corpus.sha(row['text']), 'word_count': row['word_count'], 'pdf_page_start': row['pdf_page_start'],
                'pdf_page_end': row['pdf_page_end'], 'line_refs': row['line_refs'], 'source_pdf_sha256': corpus.file_sha(ROOT / paper['pdf']),
                'normalization_events': row['normalization_events'],
                'upstream_text_sha256': row.get('upstream_text_sha256', corpus.sha(row['text'])),
                'source_text_changed': 'upstream_text_sha256' in row,
                'independent_transcription_verified': False, 'upstream_segmentation_verified': row.get('segmentation_verified', False)})
    if len(selected) != 100 or len({r['paragraph_id'] for r in selected}) != 100 or len({k['original_sha256'] for k in keys}) != 100:
        raise ValueError('one_hundred_unique_paragraphs_required')
    # Future import can verify both original text and batch membership exactly.
    fresh(output)
    write_lines(output / 'selected_originals.private.jsonl', selected)
    write_json(output / 'PRIVATE_SELECTION_AND_SPLITS.json', keys)
    write_json(output / 'source_registry.private.json', registry)
    write_json(output / 'SPLIT_MANIFEST.json', {
        'protocol': PROTOCOL, 'frozen_before_annotations': True,
        'unit': 'all selected papers by a named author stay in one split',
        'reserved_use': 'No prompt, schema or model selection using reserved responses',
        'papers': [{'paper_id': p['paper_id'], 'authors': p['authors'], 'split': p['split'],
            'paragraph_ids': [k['paragraph_id'] for k in keys if k['paper_id'] == p['paper_id']]}
            for p in papers]})
    write_json(output / 'selection_exclusions.private.json', excluded)
    write_json(output / 'selection_review.private.json', selection)
    write_json(output / 'extraction_repairs.private.json', repairs)
    write_json(output / 'pool_quality.private.json', reports)
    if not preview:
        full = json.loads(CATALOG.read_text())
        compact = compact_catalog(full)
        if expand_catalog(compact) != full['canonical_items']:
            raise ValueError('taxonomy_view_not_lossless')
        common = output / 'common';common.mkdir()
        write_json(common / 'taxonomy_catalog.compact.json', compact)
        write_json(common / 'annotation_output.schema.json', output_schema(full['taxonomy_version'], full['taxonomy_sha256']))
        shutil.copyfile(ROOT / 'docs/human_prose_annotation_v1/ANNOTATION_PROMPT.md', common / 'ANNOTATION_PROMPT.md')
        index = []
        for i, paper in enumerate(papers, 1):
            batch = f'batch_{i:02d}';directory = output / 'batches' / batch;directory.mkdir(parents=True)
            items = [r for r, k in zip(selected, keys) if k['paper_id'] == paper['paper_id']]
            write_lines(directory / 'paragraphs.jsonl', items)
            filename = f'responses_{i:02d}.jsonl'
            header = f'Batch ID: {batch}\nRequired response filename: {filename}\nRequired coverage report: coverage_{i:02d}.md\n\n'
            (directory / 'TASK.md').write_text(header + (common / 'ANNOTATION_PROMPT.md').read_text(), encoding='utf-8')
            response_dir = output.parent / 'responses' / paper['split'];response_dir.mkdir(parents=True, exist_ok=True)
            index.append({'batch_id': batch, 'paper_id': paper['paper_id'], 'split': paper['split'], 'paragraphs': 5,
                'input': str(directory.relative_to(output) / 'paragraphs.jsonl'), 'prompt': str(directory.relative_to(output) / 'TASK.md'),
                'save_response': str(Path('..') / 'responses' / paper['split'] / filename)})
        write_json(output / 'BATCH_INDEX.json', index)
        shutil.copyfile(ROOT / 'docs/human_prose_annotation_v1/START_HERE.md', output / 'START_HERE.md')
    summary = {'protocol': PROTOCOL, 'status': 'selection_preview' if preview else 'manual_annotation_batches_prepared_no_responses',
        'selected_paragraphs': 100, 'papers': 20, 'authors': len({author_key(a) for p in papers for a in p['authors']}),
        'publication_year_range': [min(p['publication_year'] for p in papers), max(p['publication_year'] for p in papers)],
        'publication_cutoff_exclusive_year': 2020, 'paragraphs_per_paper': 5,
        'split_paragraph_counts': dict(Counter(k['split'] for k in keys)),
        'word_count_range': [min(k['word_count'] for k in keys), max(k['word_count'] for k in keys)],
        'selection': 'Within-paper length-strata preview, followed by recorded text-quality selection; no detector or author-feature score.' if selection else 'Five within-paper length strata; prefer fewer obvious context dependencies and OCR dictionary anomalies; no detector or author-feature score.',
        'representative_of_all_human_writing': False, 'original_author_wording_deliberately_rewritten': False,
        'extraction_repairs': len(repairs),
        'prior_exposure_note': 'Fourteen papers appeared in earlier authorship diagnostics. Six are added here; four added author groups form the reserved set. Manual annotations are joint, not independent.',
        'frozen_fama_training_rows_read': 0, 'existing_development_rows_read': 0, 'existing_reserved_test_rows_read': 0,
        'language_model_calls': 0, 'gpu_jobs_started': 0, 'training_called': False}
    write_json(output / 'summary.json', summary)
    write_json(output / 'manifest.json', {'protocol': PROTOCOL, 'builder_sha256': corpus.file_sha(Path(__file__)),
        'registry_sha256': corpus.file_sha(registry_path), 'ocr_sha256': corpus.file_sha(ocr_path),
        'lexicon_sha256': corpus.file_sha(lexicon_path), 'taxonomy_source_sha256': corpus.file_sha(CATALOG),
        'exclusions_sha256': corpus.file_sha(exclusions_path) if exclusions_path else None,
        'selection_review_sha256': corpus.file_sha(selection_path) if selection_path else None,
        'extraction_repairs_sha256': corpus.file_sha(repairs_path) if repairs_path else None,
        'files': {str(p.relative_to(output)): corpus.file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    print(json.dumps(summary))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry', type=Path, required=True);p.add_argument('--ocr', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True);p.add_argument('--exclusions', type=Path)
    p.add_argument('--preview', action='store_true')
    p.add_argument('--selection', type=Path);p.add_argument('--repairs', type=Path)
    a = p.parse_args();prepare(a.registry, a.ocr, a.output, a.exclusions, a.preview, a.selection, a.repairs)


if __name__ == '__main__':
    main()
