"""Versioned CPU-only evidence rebinding for a private four-input repair overlay.

Exact anchors and plan compilation establish mechanical consistency, not semantic
proof, model performance, author style, or training eligibility.
"""
import argparse
import copy
import json
import re
from pathlib import Path

from fama_style.rhetoric import operation_expansion as base
from fama_style.rhetoric import paper_plan_comparison as parent
from fama_style.rhetoric import elementary_edits as edits
from fama_style.rhetoric.concise_plan import TRAIN_SHA, file_sha, read_lines
from fama_style.rhetoric.experiment import fresh, write_json, write_lines
from fama_style.prompting.economics import make_prompt

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = 'repaired_source_bindings_v1'


def anchored(text, quote, owner=None):
    """Return an exact global occurrence, optionally restricted to an owner span."""
    matches = list(re.finditer(re.escape(quote), text)) if quote else []
    candidates = [(n, m) for n, m in enumerate(matches, 1)
                  if owner is None or owner['start'] <= m.start() < m.end() <= owner['end']]
    if len(candidates) != 1:
        raise ValueError('missing_or_ambiguous_owned_evidence')
    n, match = candidates[0]
    return {'quote': quote, 'occurrence': n}, {'start': match.start(), 'end': match.end(), 'quote': quote}


def bind_details(source, spec):
    """Require relation endpoints and all constraint parts to belong to their claims."""
    owners = {c['claim_id']: anchored(source, c['source_quote'])[1] for c in spec['claims']}
    if len(owners) != len(spec['claims']): raise ValueError('duplicate_claim_id')
    relations = []; endpoints = []
    for r in spec['relations']:
        anchors = []; bounds = {}
        for side in ('from', 'to'):
            refs = r[side + '_claim_refs']
            if len(refs) != 1 or refs[0] not in owners: raise ValueError('single_known_endpoint_required')
            ev, span = anchored(source, r[side + '_quote'], owners[refs[0]])
            anchors.append(ev); bounds[side] = {'claim_id': refs[0], 'span': span}
        relations.append({k: r[k] for k in ('relation_id', 'kind', 'from_claim_refs', 'to_claim_refs', 'certainty')} | {'source_evidence': anchors})
        endpoints.append({'relation_id': r['relation_id'], **bounds})
    facets = []
    for f in spec['source_facets']:
        if f['kind'] not in set(base.KINDS) | {'technical_term'}: raise ValueError('unknown_facet_kind')
        _, span = anchored(source, f['quote'], owners[f['claim_id']])
        facets.append({k: f[k] for k in ('claim_id', 'kind', 'role')} | {'source_span': span})
    constraints = []
    roles = {'sequence': ['first_stage', 'second_stage'], 'conditional': ['antecedent', 'consequent'],
             'scoped_modality': ['modal_clause', 'asserted_clause']}
    for c in spec['intra_claim_constraints']:
        if c['kind'] not in roles or [p['role'] for p in c['parts']] != roles[c['kind']]: raise ValueError('constraint_roles')
        parts = [{'role': p['role'], 'source_span': anchored(source, p['quote'], owners[c['claim_id']])[1]} for p in c['parts']]
        if parts[0]['source_span']['end'] > parts[1]['source_span']['start']: raise ValueError('constraint_order')
        constraints.append({k: c[k] for k in ('constraint_id', 'claim_id', 'kind', 'rule')} | {'parts': parts})
    return relations, endpoints, facets, constraints


def rebind(row, spec, old_annotation, old_payload):
    source = row['plain_english']; target = row['completion']
    if row['split'] != 'train' or row['id'] != spec['source_id'] or row['paper_key'] != spec['paper_key']: raise ValueError('identity_or_split')
    if base.sha(source) != spec['expected_source_sha256'] or base.sha(target) != spec['expected_target_sha256']: raise ValueError('pair_changed')
    case_id = 'B' + base.sha(PROTOCOL + '|' + spec['parent_case_id'] + '|' + base.sha(source))[:16]
    units = parent.grouping.output_units(source)
    if units['ambiguous']: raise ValueError('ambiguous_source_units')
    record = {'case_id': case_id, 'pair_sha256': base.pair_sha(row), 'source': source, 'reference': target,
              'source_units': [{'unit_id': 'S'+str(i), 'text': t} for i, t in enumerate(units['units'], 1)],
              'target_units': copy.deepcopy(old_payload['target_record_units'])}
    if len(spec['source_groups']) != len(record['source_units']): raise ValueError('source_group_count')
    claims = []; old = {c['claim_id']: c for c in old_annotation['claims']}
    if set(old) != {c['claim_id'] for c in spec['claims']}: raise ValueError('claim_correspondence_changed')
    for c in spec['claims']:
        prior = old[c['claim_id']]; ev, _ = anchored(source, c['source_quote'])
        protections = {p['kind']: p['target_evidence'] for p in prior['protected']}
        claims.append({k: c[k] for k in ('claim_id', 'epistemic_status', 'composite')} |
                      {'source_evidence': [ev], 'target_evidence': copy.deepcopy(prior['target_evidence']),
                       'protected': [{'kind': k, 'source_evidence': [ev], 'target_evidence': copy.deepcopy(protections.get(k, prior['target_evidence']))} for k in c['protection_kinds']]})
    relations, endpoints, facets, constraints = bind_details(source, spec)
    annotation = {'case_id': case_id, 'pair_sha256': record['pair_sha256'], 'status': 'already_structured',
                  'family': 'none', 'variant': 'none', 'claims': claims,
                  'source_units': [{'unit_id': u['unit_id'], 'claim_refs': g} for u, g in zip(record['source_units'], spec['source_groups'])],
                  'target_units': copy.deepcopy(old_annotation['target_units']), 'relations': relations, 'program': None,
                  'notes': 'Fresh source review; legacy status is compiler compatibility, not absence of packing contrast. No semantic-equivalence certification.'}
    compiled = edits.compile_edits(record, annotation)['generation_payload']
    if compiled['edits'] != old_payload['edits']: raise ValueError('packing_shape_changed')
    evidence = {k: compiled[k] for k in ('source_units', 'claim_index', 'source_relations')}
    prompt_hashes = {mode: base.sha(make_prompt(source, mode, evidence, compiled['edits'] if mode == 'planned' else None)) for mode in ('ordinary', 'planned')}
    return {'case_id': case_id, 'parent_case_id': spec['parent_case_id'], 'source_id': row['id'], 'paper_key': row['paper_key'],
            'source_sha256': base.sha(source), 'target_sha256': base.sha(target), 'record': record, 'annotation': annotation,
            'evidence': evidence, 'plan': compiled['edits'], 'endpoint_bindings': endpoints, 'facets': facets,
            'intra_claim_constraints': constraints, 'review_changes': spec['changes_reviewed'],
            'context_flags': row['alignment_revision']['remaining_context'], 'alignment_caveats': spec['alignment_caveats'],
            'prompt_validation_hashes': prompt_hashes, 'training_ready': False, 'automatic_generation_ready': False}


def verify_files(directory, hashes):
    for name, digest in hashes.items():
        if file_sha(directory / name) != digest: raise ValueError('parent_artifact_changed')


def build(spec_path, output):
    spec_path = spec_path.resolve()
    repair = ROOT / 'data/meaning_alignment_repair_v1'
    verification = ROOT / 'research_runs/meaning_alignment_repair_v1/release_verification.json'
    verify_files(repair, json.loads(verification.read_text())['release_file_hashes'])
    train = ROOT / 'data/fama_corpus_v1/train.jsonl'
    if file_sha(train) != TRAIN_SHA: raise ValueError('frozen_train_changed')
    package = ROOT / 'research_runs/paper_plan_comparison_v1/package_v1'
    parents = {r['id']: r for r in parent.check_package(package)['rows']}
    op_package = ROOT / 'research_runs/operation_expansion_v1/package_v1'
    original = {r['case_id']: r for r in base.check_package(op_package)['records']}
    audit = ROOT / 'research_runs/operation_expansion_v1/response_audit_v1'
    verify_files(audit, json.loads((audit/'manifest.json').read_text())['files'])
    annotations = {r['case_id']: r['annotation'] for r in read_lines(audit/'annotations.checked.private.jsonl')}
    rows = {r['id']: r for r in read_lines(repair/'repaired_pairs.private.jsonl')}
    spec = json.loads(spec_path.read_text())
    if spec['protocol'] != PROTOCOL or len(spec['records']) != 4 or {s['source_id'] for s in spec['records']} != set(rows): raise ValueError('four_repaired_cases_required')
    cards = []
    for s in spec['records']:
        prior = parents[s['parent_case_id']]; row = rows[s['source_id']]
        if prior['source_id'] != row['id'] or prior['reference'] != row['completion']: raise ValueError('parent_target_changed')
        payload = {**prior['payload'], 'target_record_units': original[s['parent_case_id']]['record']['target_units']}
        cards.append(rebind(row, s, annotations[s['parent_case_id']], payload))
    summary = {'protocol': PROTOCOL, 'rows': len(cards), 'claims': sum(len(c['annotation']['claims']) for c in cards),
               'plans_revalidated': len(cards), 'clean_bounded_diagnostic_cases': sum(not c['context_flags'] and not c['alignment_caveats'] for c in cards),
               'qualified_diagnostic_cases': sum(bool(c['context_flags'] or c['alignment_caveats']) for c in cards),
               'facets': sum(len(c['facets']) for c in cards), 'intra_claim_constraints': sum(len(c['intra_claim_constraints']) for c in cards),
               'training_ready_rows': 0, 'automatic_generation_ready_rows': 0, 'model_calls': 0, 'gpu_jobs_started': 0,
               'targets_changed': 0, 'validation_rows_read': 0, 'reserved_test_rows_read': 0,
               'semantic_equivalence_verified': False, 'author_style_improvement_measured': False,
               'cases': [{'paper_key': c['paper_key'], 'claims': len(c['annotation']['claims']), 'output_slots': sum(len(e['output_slots']) for e in c['plan']),
                          'context_flag_retained': bool(c['context_flags']), 'alignment_caveat_retained': bool(c['alignment_caveats'])} for c in cards]}
    fresh(output); write_lines(output/'paired_cards.private.jsonl', cards)
    ledgers = []
    for c in cards:
        d = output/c['paper_key']; d.mkdir()
        (d/'source.txt').write_text(c['record']['source'], encoding='utf-8')
        write_json(d/'evidence.json', c['evidence']); write_json(d/'plan.json', c['plan'])
        ledgers.append({k: c[k] for k in ('case_id', 'source_id', 'paper_key', 'source_sha256', 'evidence', 'endpoint_bindings', 'facets', 'intra_claim_constraints')})
        write_json(d/'binding_manifest.json', {'source_sha256': c['source_sha256'], 'prompt_validation_hashes': c['prompt_validation_hashes'],
                   'context_flags': c['context_flags'], 'alignment_caveat_codes': [v['code'] for v in c['alignment_caveats']],
                   'training_ready': False, 'automatic_generation_ready': False})
    write_lines(output/'source_ledgers.private.jsonl', ledgers); write_json(output/'summary.json', summary)
    if file_sha(train) != TRAIN_SHA: raise ValueError('frozen_train_changed')
    write_json(output/'manifest.json', {'protocol': PROTOCOL, 'code_sha256': file_sha(Path(__file__)),
       'inputs': {str(p.relative_to(ROOT)): file_sha(p) for p in (spec_path, verification, repair/'manifest.json', package/'manifest.json', op_package/'manifest.json', audit/'manifest.json', train)},
       'files': {str(p.relative_to(output)): file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()},
       'annotation_method': spec['annotation_method'], 'external_annotation_calls': 0})
    print(json.dumps(summary, sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--specification', type=Path, required=True); parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv); build(args.specification, args.output_dir)


if __name__ == '__main__': main()
