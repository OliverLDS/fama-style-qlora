"""Compile oracle sentence-packing edits from existing, grounded claim maps.

CPU only. Whole claim IDs are indivisible; repeated IDs require finer annotation.
This protocol is separate from the frozen operation-family pilot and its gates.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
from . import operation_expansion as base
from .concise_plan import file_sha, read_lines
from .experiment import fresh, write_json, write_lines

PROTOCOL = 'elementary_packing_edits_v1'
ROOT = Path(__file__).resolve().parents[3]
CATALOG = {
    'KEEP': 'Retain this complete source sentence and all its claims.',
    'MERGE': 'Express the listed complete source sentences in one sentence; preserve all claims and their bindings. Co-location alone licenses no new logical relation.',
    'SPLIT': 'Express separately the already distinct, independently anchored claims in this source sentence. Keep each whole claim intact.',
}


def partition(units, known):
    groups = [u['claim_refs'] for u in units]
    flat = [c for g in groups for c in g]
    if any(not g or len(g) != len(set(g)) for g in groups):
        raise ValueError('empty_or_duplicate_unit_claims')
    if set(flat) != known:
        raise ValueError('incomplete_claim_partition')
    if len(flat) != len(set(flat)):
        raise ValueError('repeated_claim_requires_subclaim_binding')
    return groups


def compile_edits(record, annotation):
    checked = base.validate_annotation(record, annotation)
    if annotation['status'] != 'already_structured':
        raise ValueError('separate_protocol_only_for_already_structured_rows')
    claims = {c['claim_id']: c for c in checked['compiled_claims']}
    source = partition(annotation['source_units'], set(claims))
    target = partition(annotation['target_units'], set(claims))
    # Connected components of source/target sentence membership, not semantic edges.
    components = []
    unseen = set(claims)
    while unseen:
        component = {min(unseen)}
        while True:
            expanded = component | {c for g in source + target if component.intersection(g) for c in g}
            if expanded == component:
                break
            component = expanded
        unseen -= component
        si = [i for i, g in enumerate(source) if component.intersection(g)]
        ti = [i for i, g in enumerate(target) if component.intersection(g)]
        if si != list(range(si[0], si[-1] + 1)) or ti != list(range(ti[0], ti[-1] + 1)):
            raise ValueError('noncontiguous_edit_not_supported')
        if len(si) == len(ti) == 1:
            code = 'KEEP'
        elif len(ti) == 1:
            code = 'MERGE'
        elif len(si) == 1:
            code = 'SPLIT'
            # Evidence overlap cannot establish independent subclaim boundaries.
            ordered = [c for j in ti for c in target[j]]
            for n, cid in enumerate(ordered):
                for other in ordered[n + 1:]:
                    if any(base.overlaps(a, b) for a in claims[cid]['source_spans'] for b in claims[other]['source_spans']):
                        raise ValueError('split_claim_evidence_overlaps')
            starts = [min(s['start'] for s in claims[c]['source_spans']) for c in ordered]
            if starts != sorted(starts):
                raise ValueError('split_would_reorder_claims')
        else:
            raise ValueError('many_to_many_regrouping_not_supported')
        components.append({'op': code, 'source_unit_ids': [annotation['source_units'][i]['unit_id'] for i in si],
                           'output_slots': [{'claim_refs': target[j]} for j in ti],
                           '_source_index': si[0], '_target_index': ti[0]})
    components.sort(key=lambda x: x['_source_index'])
    if [c['_target_index'] for c in components] != sorted(c['_target_index'] for c in components):
        raise ValueError('sentence_reordering_not_supported')
    for c in components:
        del c['_source_index']; del c['_target_index']
    if not any(c['op'] != 'KEEP' for c in components):
        raise ValueError('no_elementary_packing_change')
    claim_index = []
    for c in checked['compiled_claims']:
        claim_index.append({'claim_id': c['claim_id'], 'composite': c['composite'],
                            'epistemic_status': c['epistemic_status'], 'source_spans': c['source_spans'],
                            'protected': [{'kind': p['kind'], 'source_spans': base.resolve(record['source'], p['source_evidence'])} for p in c['protected']]})
    # Explicit allowlist: no reference, target evidence, annotator notes or free-text plans.
    payload = {'protocol': PROTOCOL, 'source': record['source'], 'source_units': record['source_units'],
               'claim_index': claim_index, 'source_relations': annotation['relations'], 'edits': components}
    # Relations contain source evidence only; their certainty remains explicit.
    return {'case_id': record['case_id'], 'pair_sha256': record['pair_sha256'],
            'generation_payload': payload, 'prompt': render(payload),
            'semantic_equivalence_verified': False, 'atomic_claim_completeness_verified': False,
            'oracle_reference_grouping': True, 'training_eligible': False}


def render(payload):
    return ('Rewrite as one formal economics paragraph, returning only the paragraph. Follow the sentence-packing edits below. '
            'Keep every whole claim, actor, attribution, quantity, condition, negation, uncertainty and scope bound to its original claim. '
            'Retain all untouched sentences. Do not invent causal, conditional or inferential connections from sentence grouping. '
            'Preserve annotated relations at their recorded certainty; interpretive edges are not established facts. '
            'Composite claims remain indivisible. Resolve pronouns using the original source without changing attribution. '
            'The plan is an oracle diagnostic of sentence organization, not a proven author-style specification.\n'
            + json.dumps({'operations': CATALOG, 'task': payload}, ensure_ascii=False, sort_keys=True))


def run(package, audit, output):
    key = base.check_package(package)
    manifest = json.loads((audit / 'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        if file_sha(audit / name) != digest:
            raise ValueError('Original audit changed')
    checked = read_lines(audit / 'annotations.checked.private.jsonl')
    infos = {r['case_id']: r for r in key['records']}
    accepted, blocked, unchanged = [], [], []
    for row in checked:
        a = row['annotation']
        if a['status'] != 'already_structured':
            continue
        info = infos[row['case_id']]
        try:
            result = compile_edits(info['record'], a)
        except ValueError as exc:
            entry = {'case_id': row['case_id'], 'paper_key': info['paper_key'], 'reason': str(exc)}
            (unchanged if str(exc) == 'no_elementary_packing_change' else blocked).append(entry)
            continue
        result.update(source_id=info['source_id'], paper_key=info['paper_key'],
                      selection_role=info['selection_role'],
                      diagnostic_probe=info['selection_role'] == 'preservation_or_no_increment_probe',
                      annotation_sha256=base.sha(json.dumps(a, sort_keys=True, ensure_ascii=False)))
        accepted.append(result)
    counts = Counter(op['op'] for r in accepted for op in r['generation_payload']['edits'])
    summary = {'protocol': PROTOCOL, 'already_structured_rows': 24, 'compiled_oracle_candidates': len(accepted),
               'nonprobe_candidates': sum(not r['diagnostic_probe'] for r in accepted),
               'diagnostic_probes': sum(r['diagnostic_probe'] for r in accepted),
               'candidate_papers': dict(Counter(r['paper_key'] for r in accepted)),
               'operation_instances': dict(counts), 'blocked': dict(Counter(r['reason'] for r in blocked)),
               'unchanged': len(unchanged), 'prerequisite_isolation_programs': 0,
               'prerequisite_note': 'No independently bound prerequisite isolation established by these maps; no condition is manufactured from composite claims.',
               'training_eligible': 0, 'original_family_gates_unchanged': True,
               'reserved_test_rows_read': 0, 'validation_rows_read': 0, 'gpu_jobs_started': 0,
               'semantic_equivalence_verified': False,
               'limitations': ['Oracle reference-derived grouping is not deployment performance.',
                              'Exact evidence and coverage checks are not independent semantic verification.',
                              'Sentence packing alone does not establish distinctive author style.',
                              'Six repeated-claim rows need finer subclaim boundaries before compilation.']}
    if len(accepted) + len(blocked) + len(unchanged) != summary['already_structured_rows']:
        raise ValueError('Unexpected annotation population')
    fresh(output)
    write_lines(output / 'oracle_candidates.private.jsonl', accepted)
    write_json(output / 'dispositions.private.json', {'blocked': blocked, 'unchanged': unchanged})
    write_json(output / 'summary.json', summary)
    write_json(output / 'manifest.json', {'protocol': PROTOCOL, 'code_sha256': file_sha(Path(__file__)),
               'package_manifest_sha256': file_sha(package / 'manifest.json'),
               'audit_manifest_sha256': file_sha(audit / 'manifest.json'),
               'files': {p.name: file_sha(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps(summary, sort_keys=True))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--audit-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    run(a.package, a.audit_dir, a.output_dir)


if __name__ == '__main__':
    main()
