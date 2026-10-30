"""Audit manual human-prose pairs without training, generation or reserved scoring.

A closed JSON-schema subset checks transport; exact evidence and link checks
establish mechanical consistency only. Semantic equivalence remains unverified.
"""
import argparse
import copy
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import re
import shutil
import statistics

from . import author_comparison_corpus as corpus
from . import human_prose_annotations as preparation
from .author_control_replication import verify_files
from .signal import overlap
from .claims import tokens
from fama_style.rhetoric.experiment import fresh, write_json, write_lines

PROTOCOL = 'human_prose_review_v1'
LEVELS = {'paragraph_structure': 'paragraph', 'sentence_structure': 'sentence', 'language_features': 'language'}


def schema_errors(value, schema, path=''):
    """Validate every constraint used by the frozen packet's closed schema.

    This is deliberately not advertised as a general JSON Schema implementation.
    Unknown keywords fail closed instead of being silently ignored.
    """
    allowed = {'$schema', '$id', 'type', 'properties', 'required', 'additionalProperties',
               'items', 'anyOf', 'const', 'enum', 'pattern', 'minLength', 'minimum', 'minItems'}
    if set(schema) - allowed:
        raise ValueError('unsupported_schema_keyword')
    errors = []
    def error(code): errors.append({'code': code, 'path': path})
    if 'anyOf' in schema:
        if not any(not schema_errors(value, s, path) for s in schema['anyOf']): error('schema_union')
        return errors
    if 'const' in schema and (type(value) is not type(schema['const']) or value != schema['const']): error('schema_const')
    if 'enum' in schema and value not in schema['enum']: error('schema_enum')
    kind = schema.get('type')
    check = {'object': lambda: isinstance(value, dict), 'array': lambda: isinstance(value, list),
             'string': lambda: isinstance(value, str), 'integer': lambda: type(value) is int,
             'boolean': lambda: type(value) is bool, 'null': lambda: value is None}
    kinds = kind if isinstance(kind, list) else [kind] if kind else []
    if kinds and not any(check[k]() for k in kinds):
        error('schema_type');return errors
    if isinstance(value, dict) and 'properties' in schema:
        if set(schema.get('required', [])) - set(value): error('schema_missing_field')
        if schema.get('additionalProperties') is False and set(value) - set(schema['properties']): error('schema_extra_field')
        for key in value.keys() & schema['properties'].keys():
            errors.extend(schema_errors(value[key], schema['properties'][key], path + '/' + key))
    if isinstance(value, list):
        if len(value) < schema.get('minItems', 0): error('schema_min_items')
        if 'items' in schema:
            for i, child in enumerate(value): errors.extend(schema_errors(child, schema['items'], path + '/' + str(i)))
    if isinstance(value, str):
        if len(value) < schema.get('minLength', 0): error('schema_min_length')
        if 'pattern' in schema and not re.search(schema['pattern'], value): error('schema_pattern')
    if type(value) is int and value < schema.get('minimum', value): error('schema_minimum')
    return errors


def occurrence_span(text, anchor):
    quote, occurrence = anchor['quote'], anchor['occurrence']
    if not quote or type(occurrence) is not int or occurrence < 1: return None
    offset = 0
    for _ in range(occurrence):
        start = text.find(quote, offset)
        if start < 0: return None
        offset = start + len(quote)
    return start, offset


def normalized(text): return ' '.join(text.split())


def apply_program_repairs(row, patches):
    """Allow only direction normalization or a missing declared sentence binding.

    No prose, evidence, claims, operators, relations or labels can be changed.
    """
    result = copy.deepcopy(row)
    for patch in patches:
        operation = result['transformations'][patch['operation_index']]
        field = patch['field']
        if field not in ('operation', 'source_sentence_ids') or operation[field] != patch['before']:
            raise ValueError('unsupported_or_stale_program_repair')
        if field == 'operation':
            a, b = len(set(operation['source_sentence_ids'])), len(set(operation['target_sentence_ids']))
            expected = 'MERGE' if a > b >= 1 else 'SPLIT' if b > a >= 1 else None
            if patch['after'] != expected or patch['before'] not in ('MERGE', 'SPLIT'):
                raise ValueError('direction_repair_not_supported_by_bindings')
        else:
            old, new = set(patch['before']), set(patch['after'])
            sentences = {s['sentence_id']: set(s['claim_ids']) for s in result['source_schema']['sentences']}
            if not old < new or new - set(sentences):
                raise ValueError('sentence_repair_not_an_expansion')
            needed = set(operation['claim_ids'])
            before = set().union(*(sentences[s] for s in old)) if old else set()
            after = set().union(*(sentences[s] for s in new))
            if needed <= before or not needed <= after:
                raise ValueError('sentence_repair_has_no_missing_claim_basis')
            if any(not (sentences[s] & (needed - before)) for s in new-old):
                raise ValueError('sentence_repair_adds_unrelated_sentence')
        operation[field] = patch['after']
    return result


def number_mentions(text):
    return Counter(m.replace(',', '') for m in re.findall(r'(?<![A-Za-z])\d+(?:,\d{3})*(?:\.\d+)?', text))


def pair_metrics(row):
    source, target = row['plain_english'], row['original_paragraph']
    if not source: return None
    a, b = number_mentions(source), number_mentions(target)
    source_s = row['source_schema']['sentences'] if row['source_schema'] else []
    target_s = row['target_schema']['sentences'] if row['target_schema'] else []
    def together(sentences):
        return {tuple(sorted(pair)) for s in sentences for pair in combinations(set(s['claim_ids']), 2)}
    aa, bb = together(source_s), together(target_s)
    return {'lexical_overlap_sequence_ratio': overlap(source, target),
        'normalized_token_copy': tokens(source) == tokens(target),
        'source_words': len(corpus.words(source)), 'target_words': len(corpus.words(target)),
        'source_sentences': len(source_s), 'target_sentences': len(target_s),
        'sentence_count_changed': len(source_s) != len(target_s),
        'claim_coexpression_pairs_changed': len(aa ^ bb),
        'numeric_mentions_only_in_source': list((a-b).elements()), 'numeric_mentions_only_in_target': list((b-a).elements())}


def audit_record(row, catalog):
    errors, warnings, program_errors = [], [], []
    counts = Counter()
    def err(code, path): errors.append({'code': code, 'path': path})
    def warn(code, path): warnings.append({'code': code, 'path': path})
    def anchors(items, text, path, required=False):
        if required and not items: err('missing_evidence', path)
        spans = []
        for i, anchor in enumerate(items):
            counts['evidence_anchors_checked'] += 1
            span = occurrence_span(text or '', anchor)
            if span is None: err('evidence_not_exact', path + '/' + str(i))
            else: spans.append(span)
        return spans
    ledger = row['semantic_ledger'];source = row['plain_english'];target = row['original_paragraph']
    entity_ids = [x['entity_id'] for x in ledger['entities']]
    claim_ids = [x['claim_id'] for x in ledger['claims']]
    relation_ids = [x['relation_id'] for x in ledger['relations']]
    for field, ids in [('entities', entity_ids), ('claims', claim_ids), ('relations', relation_ids)]:
        if len(ids) != len(set(ids)): err('duplicate_' + field + '_id', '/semantic_ledger/' + field)
    entities, claims, relations = set(entity_ids), set(claim_ids), set(relation_ids)
    if not claims: err('empty_claim_inventory', '/semantic_ledger/claims')
    def refs(values, namespace, path):
        if len(values) != len(set(values)): warn('duplicate_reference', path)
        if set(values) - namespace: err('unresolved_reference', path)
    for i, e in enumerate(ledger['entities']): anchors(e['original_evidence'], target, '/entities/' + str(i), True)
    for i, c in enumerate(ledger['claims']):
        path = '/semantic_ledger/claims/' + str(i)
        refs(c['actor_ids'] + c['other_entity_ids'], entities, path + '/entity_ids')
        anchors(c['original_evidence'], target, path + '/original_evidence', True)
        anchors(c['rewrite_evidence'], source, path + '/rewrite_evidence', c['alignment_status'] == 'preserved')
        if not c['proposition'].strip(): err('empty_proposition', path)
        if c['alignment_status'] != 'preserved': warn('annotator_claim_alignment_' + c['alignment_status'], path)
    for i, rel in enumerate(ledger['relations']):
        path = '/semantic_ledger/relations/' + str(i)
        refs(rel['from_claim_ids'] + rel['to_claim_ids'], claims, path + '/claim_ids')
        if not rel['from_claim_ids'] or not rel['to_claim_ids']: err('empty_relation_endpoint', path)
        anchors(rel['original_evidence'], target, path + '/original_evidence', True)
        anchors(rel['rewrite_evidence'], source, path + '/rewrite_evidence', rel['alignment_status'] == 'preserved')
        if rel['alignment_status'] != 'preserved': warn('annotator_relation_alignment_' + rel['alignment_status'], path)
    for i, c in enumerate(ledger['unmatched_source_claims']):
        anchors(c['rewrite_evidence'], source, '/unmatched_source_claims/' + str(i), True)
        warn('annotator_unsupported_source_claim', '/unmatched_source_claims/' + str(i))
    if row['source_quality']['status'] != 'usable': warn('annotator_source_quality_' + row['source_quality']['status'], '/source_quality')
    if row['source_quality']['needs_external_context']: warn('annotator_needs_external_context', '/source_quality')
    if row['needs_review']: warn('annotator_needs_review', '/needs_review')
    if row['rewrite_status'] == 'pending_review' and source: warn('pending_review_with_rewrite', '/rewrite_status')
    if row['rewrite_status'] != 'pending_review' and not source: err('missing_rewrite', '/plain_english')
    if row['rewrite_status'] == 'already_clear' and source != target: warn('already_clear_text_changed', '/rewrite_status')
    if source == target and row['rewrite_status'] == 'simplified': warn('simplified_but_identical', '/rewrite_status')
    sentence_sets = {};assigned = defaultdict(Counter)
    for side, text in [('source_schema', source), ('target_schema', target)]:
        schema = row[side]
        if schema is None:
            if text: err('missing_schema', '/' + side)
            sentence_sets[side] = {};continue
        sentences = schema['sentences'];ids = [s['sentence_id'] for s in sentences]
        if len(ids) != len(set(ids)): err('duplicate_sentence_id', '/' + side)
        if ids != ['S' + str(i+1) for i in range(len(ids))]: warn('nonsequential_sentence_ids', '/' + side)
        if normalized(' '.join(s['exact_text'] for s in sentences)) != normalized(text or ''):
            err('sentence_coverage_mismatch', '/' + side)
        sentence_sets[side] = {s['sentence_id']: set(s['claim_ids']) for s in sentences}
        offset, bound_claims = 0, set()
        for i, s in enumerate(sentences):
            refs(s['claim_ids'], claims, '/' + side + '/sentences/' + str(i))
            bound_claims.update(s['claim_ids'])
            start = (text or '').find(s['exact_text'], offset)
            if start < 0: err('sentence_not_exact_or_out_of_order', '/' + side + '/sentences/' + str(i))
            else: offset = start + len(s['exact_text'])
        expected_claims = claims if side == 'target_schema' else {c['claim_id'] for c in ledger['claims'] if c['alignment_status'] != 'omitted'}
        if bound_claims != expected_claims: err('claim_sentence_coverage_mismatch', '/' + side)
        for field, level in LEVELS.items():
            for i, obs in enumerate(schema[field]):
                path = '/' + side + '/' + field + '/' + str(i)
                refs(obs['claim_ids'], claims, path + '/claim_ids')
                refs(obs['relation_ids'], relations, path + '/relation_ids')
                refs(obs['sentence_ids'], set(ids), path + '/sentence_ids')
                anchors(obs['evidence'], text, path + '/evidence', obs['value'] == 'present')
                cid = obs['canonical_id'];status = obs['assignment_status']
                if status == 'assigned':
                    if cid not in catalog: err('unknown_canonical_id', path)
                    elif catalog[cid]['level'] != level or catalog[cid]['dimension'] != obs['dimension']: err('taxonomy_level_dimension_mismatch', path)
                    else:
                        if obs['value'] == 'present': assigned[side][cid] += 1
                    if obs['candidate_ids']: err('assigned_with_candidates', path)
                else:
                    if cid is not None: err('unassigned_has_canonical_id', path)
                    warn('taxonomy_' + status, path)
                for candidate in obs['candidate_ids']:
                    if candidate not in catalog or catalog[candidate]['level'] != level or catalog[candidate]['dimension'] != obs['dimension']:
                        err('invalid_candidate_label', path)
                if obs['sentence_ids']:
                    available = set().union(*(sentence_sets[side].get(s, set()) for s in obs['sentence_ids']))
                    if set(obs['claim_ids']) - available: warn('observation_claim_outside_bound_sentences', path)
    for i, operation in enumerate(row['transformations']):
        path = '/transformations/' + str(i)
        refs(operation['claim_ids'], claims, path + '/claim_ids')
        refs(operation['relation_ids'], relations, path + '/relation_ids')
        a, b = set(operation['source_sentence_ids']), set(operation['target_sentence_ids'])
        refs(list(a), set(sentence_sets.get('source_schema', {})), path + '/source_sentence_ids')
        refs(list(b), set(sentence_sets.get('target_schema', {})), path + '/target_sentence_ids')
        counts['operations_checked'] += 1
        if operation['operation'] == 'MERGE' and not len(a) > len(b) >= 1:
            program_errors.append({'code': 'merge_cardinality_not_supported', 'path': path})
        if operation['operation'] == 'SPLIT' and not len(b) > len(a) >= 1:
            program_errors.append({'code': 'split_cardinality_not_supported', 'path': path})
        for side, selected in [('source_schema', a), ('target_schema', b)]:
            available = set().union(*(sentence_sets.get(side, {}).get(s, set()) for s in selected)) if selected else set()
            if set(operation['claim_ids']) - available:
                program_errors.append({'code': 'operation_claim_outside_sentences', 'path': path + '/' + side})
        if operation['uncertain']: warn('uncertain_transformation', path)
    counts.update(claims=len(claims), relations=len(relations), taxonomy_observations=sum(len(row[s][f]) for s in ('source_schema', 'target_schema') if row[s] for f in LEVELS))
    metrics = pair_metrics(row)
    if metrics and (metrics['numeric_mentions_only_in_source'] or metrics['numeric_mentions_only_in_target']): warn('numeric_surface_change_needs_scope_check', '/plain_english')
    return {'errors': errors, 'warnings': warnings, 'program_errors': program_errors, 'counts': dict(counts),
        'metrics': metrics, 'label_presence': {side: sorted(values) for side, values in assigned.items()},
        'training_ready': False, 'semantic_equivalence_verified': False}


def aggregate(records, split):
    group = [r for r in records if r['split'] == split]
    good = [r for r in group if r['schema_valid']]
    audits = [r['audit'] for r in good]
    metrics = [a['metrics'] for a in audits if a['metrics']]
    counts = Counter()
    for a in audits: counts.update(a['counts'])
    return {'rows': len(group), 'schema_valid_rows': len(good), 'mechanically_consistent_rows': sum(not a['errors'] for a in audits),
        'rows_with_program_errors': sum(bool(a['program_errors']) for a in audits),
        'rows_flagged_by_annotator': sum(r['annotation']['needs_review'] for r in good),
        'rewrite_statuses_annotator': dict(Counter(r['annotation']['rewrite_status'] for r in good)),
        'source_quality_annotator': dict(Counter(r['annotation']['source_quality']['status'] for r in good)),
        'alignment_statuses_annotator': dict(Counter(c['alignment_status'] for r in good for c in r['annotation']['semantic_ledger']['claims'])),
        'mechanical_error_codes': dict(Counter(e['code'] for a in audits for e in a['errors'])),
        'program_error_codes': dict(Counter(e['code'] for a in audits for e in a['program_errors'])),
        'warning_codes': dict(Counter(e['code'] for a in audits for e in a['warnings'])),
        'mechanical_counts': dict(counts),
        'lexical_overlap': {'median': statistics.median(m['lexical_overlap_sequence_ratio'] for m in metrics) if metrics else None,
            'minimum': min((m['lexical_overlap_sequence_ratio'] for m in metrics), default=None),
            'maximum': max((m['lexical_overlap_sequence_ratio'] for m in metrics), default=None),
            'rows_at_least_0_9': sum(m['lexical_overlap_sequence_ratio'] >= .9 for m in metrics),
            'normalized_copies': sum(m['normalized_token_copy'] for m in metrics)},
        'sentence_count_changed': sum(m['sentence_count_changed'] for m in metrics),
        'claim_coexpression_changed': sum(m['claim_coexpression_pairs_changed'] > 0 for m in metrics),
        'semantic_equivalence_verified_rows': 0, 'training_ready_rows': 0}


def run(package, responses, output, session_path, repairs_path=None):
    verify_files(package)
    schema = json.loads((package/'common/annotation_output.schema.json').read_text())
    compact = json.loads((package/'common/taxonomy_catalog.compact.json').read_text())
    catalog = {x['canonical_id']: x for x in preparation.expand_catalog(compact)}
    index = json.loads((package/'BATCH_INDEX.json').read_text())
    keys = {x['paragraph_id']: x for x in json.loads((package/'PRIVATE_SELECTION_AND_SPLITS.json').read_text())}
    session = json.loads(session_path.read_text())
    patches = json.loads(repairs_path.read_text()) if repairs_path else []
    by_paragraph = defaultdict(list)
    for patch in patches: by_paragraph[patch['paragraph_id']].append(patch)
    records, coverage, preserved, issues = [], [], {}, []
    for batch in index:
        name = Path(batch['save_response']).name
        path = responses/batch['split']/name
        expected = {r['paragraph_id']: r['text'] for r in corpus.read_lines(package/batch['input'])}
        raw = path.read_bytes();preserved[str(path.relative_to(responses))] = corpus.file_sha(path)
        rows = [];parse_errors=[]
        for n, line in enumerate(raw.decode('utf-8-sig').splitlines(), 1):
            if not line.strip(): continue
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: parse_errors.append(n)
        ids = [r.get('paragraph_id') for r in rows if isinstance(r, dict)]
        valid_schema = [not schema_errors(r, schema) for r in rows]
        exact = all(isinstance(r, dict) and r.get('original_paragraph') == expected.get(r.get('paragraph_id')) for r in rows)
        entry = {'batch_id': batch['batch_id'], 'split': batch['split'], 'rows': len(rows),
            'json_parse_error_lines': parse_errors, 'ids_complete_unique': len(ids)==len(expected) and set(ids)==set(expected),
            'originals_exact': exact, 'schema_valid_rows': sum(valid_schema)}
        coverage.append(entry)
        if parse_errors or not entry['ids_complete_unique'] or not exact: issues.append(batch['batch_id'])
        report = path.with_name(name.replace('responses_', 'coverage_').replace('.jsonl', '.md'))
        if report.exists(): preserved[str(report.relative_to(responses))] = corpus.file_sha(report)
        # Reserved annotations receive transport/schema/original-hash checks only.
        # No evidence checks, lexical metrics, labels, semantic review or selection.
        if batch['split'] == 'reserved_test': continue
        for row, valid in zip(rows, valid_schema):
            key = keys.get(row.get('paragraph_id')) if isinstance(row, dict) else None
            if key is None: continue
            if row['paragraph_id'] in by_paragraph:
                if not valid: raise ValueError('cannot_repair_schema_invalid_row')
                row = apply_program_repairs(row, by_paragraph[row['paragraph_id']])
                if schema_errors(row, schema): raise ValueError('repair_broke_schema')
            record = {'paragraph_id': row['paragraph_id'], 'paper_id': key['paper_id'], 'split': batch['split'],
                'batch_id': batch['batch_id'], 'schema_valid': valid,
                'applied_program_repair_ids': [p['patch_id'] for p in by_paragraph.get(row['paragraph_id'], [])],
                'schema_errors': schema_errors(row, schema), 'annotation': row,
                'audit': audit_record(row, catalog) if valid else None}
            records.append(record)
    if issues: raise ValueError('transport_or_original_integrity_failure:' + ','.join(issues))
    summary = {'protocol': PROTOCOL, 'status': 'mechanical_audit_complete_semantic_quality_unverified',
        'expected_rows': 100, 'received_rows': sum(x['rows'] for x in coverage), 'batches': len(coverage),
        'transport_complete': not issues, 'all_originals_preserved': all(x['originals_exact'] for x in coverage),
        'all_schema_valid_rows': sum(x['schema_valid_rows'] for x in coverage),
        'training': aggregate(records, 'training'), 'development': aggregate(records, 'development'),
        'reserved': {'rows': sum(x['rows'] for x in coverage if x['split']=='reserved_test'),
            'scope': 'transport/schema/original integrity only; no content metrics or model-selection analysis',
            'content_rows_audited': 0},
        'session': session, 'independent_semantic_verification': False,
        'program_repairs': len(patches), 'raw_responses_modified': False,
        'new_model_calls': 0, 'gpu_jobs_started': 0, 'training_called': False, 'existing_fama_test_rows_used': 0}
    # Training-only label contrasts are discovery descriptors, not author signatures.
    contrasts=[]
    training=[r for r in records if r['split']=='training' and r['schema_valid'] and not r['audit']['errors']]
    for label, category in sorted(catalog.items()):
        src=sum(label in r['audit']['label_presence'].get('source_schema', []) for r in training)
        tgt=sum(label in r['audit']['label_presence'].get('target_schema', []) for r in training)
        contrasts.append({'canonical_id': label, 'level': category['level'], 'dimension': category['dimension'],
            'source_rows': src, 'target_rows': tgt, 'target_minus_source': tgt-src, 'rows': len(training)})
    fresh(output);(output/'raw').mkdir()
    for name, digest in preserved.items():
        dest=output/'raw'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(responses/name,dest)
        if corpus.file_sha(dest)!=digest: raise ValueError('response_changed_during_import')
    shutil.copyfile(session_path,output/'session_confirmation.private.json')
    write_json(output/'program_repairs.private.json',patches)
    write_json(output/'coverage.json',coverage);write_lines(output/'audits.private.jsonl',records)
    write_json(output/'training_label_contrasts.json',contrasts);write_json(output/'summary.json',summary)
    write_json(output/'manifest.json',{'protocol': PROTOCOL, 'code_sha256':corpus.file_sha(Path(__file__)),
        'package_manifest_sha256':corpus.file_sha(package/'manifest.json'),'response_sha256':preserved,
        'program_repairs_sha256':corpus.file_sha(repairs_path) if repairs_path else None,
        'files':{str(p.relative_to(output)):corpus.file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    print(json.dumps({k:summary[k] for k in ('received_rows','all_originals_preserved','all_schema_valid_rows','training','development','reserved')}))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('package','responses','output','session'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--repairs',type=Path)
    a=p.parse_args();run(a.package,a.responses,a.output,a.session,a.repairs)


if __name__=='__main__':main()
