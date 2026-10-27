"""Grounded edit plans preserve claim coverage and isolate oracle information."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fama_style.rhetoric import elementary_edits as e


def fixture(source_groups, target_groups):
    claims = [{'claim_id': c, 'composite': False, 'epistemic_status': 'assertion',
               'source_spans': [{'start': i * 20, 'end': i * 20 + 10, 'quote': 'source ' + c}],
               'target_spans': [{'start': i, 'end': i + 1, 'quote': 'SECRET_REFERENCE'}], 'protected': []}
              for i, c in enumerate(['c1', 'c2', 'c3'])]
    record = {'case_id': 'synthetic', 'pair_sha256': 'hash', 'source': 'Synthetic source only.',
              'reference': 'SECRET_REFERENCE', 'source_units': [{'unit_id': 'S' + str(i), 'text': 'unit'} for i in range(len(source_groups))]}
    a = {'status': 'already_structured', 'source_units': [{'unit_id': 'S' + str(i), 'claim_refs': g} for i, g in enumerate(source_groups)],
         'target_units': [{'unit_id': 'T' + str(i), 'claim_refs': g} for i, g in enumerate(target_groups)],
         'notes': 'SECRET_NOTES', 'relations': [{'relation_id': 'r1', 'kind': 'causal', 'certainty': 'interpretive',
             'from_claim_refs': ['c1'], 'to_claim_refs': ['c2'], 'source_evidence': [{'quote': 'source', 'occurrence': 1}]}]}
    return record, a, {'compiled_claims': claims}


class ElementaryEditsTests(unittest.TestCase):
    def compile_fixture(self, s, t):
        r, a, checked = fixture(s, t)
        with patch.object(e.base, 'validate_annotation', return_value=checked):
            return e.compile_edits(r, a)

    def test_merge_preserves_untouched_claims_and_relation_certainty(self):
        r = self.compile_fixture([['c1'], ['c2'], ['c3']], [['c1', 'c2'], ['c3']])
        p = r['generation_payload']
        self.assertEqual([x['op'] for x in p['edits']], ['MERGE', 'KEEP'])
        self.assertEqual(p['source_relations'][0]['certainty'], 'interpretive')
        self.assertEqual({c for op in p['edits'] for slot in op['output_slots'] for c in slot['claim_refs']}, {'c1', 'c2', 'c3'})
        self.assertNotIn('SECRET_REFERENCE', r['prompt'])
        self.assertNotIn('SECRET_NOTES', r['prompt'])
        self.assertNotIn('target_spans', r['prompt'])
        self.assertFalse(r['training_eligible'])

    def test_split_requires_independent_evidence(self):
        s, t = [['c1', 'c2'], ['c3']], [['c1'], ['c2'], ['c3']]
        self.assertEqual(self.compile_fixture(s, t)['generation_payload']['edits'][0]['op'], 'SPLIT')
        r, a, checked = fixture(s, t)
        checked['compiled_claims'][1]['source_spans'] = checked['compiled_claims'][0]['source_spans']
        with patch.object(e.base, 'validate_annotation', return_value=checked):
            with self.assertRaisesRegex(ValueError, 'evidence_overlaps'):
                e.compile_edits(r, a)

    def test_repeated_claim_and_reordering_blocked(self):
        cases = [([['c1'], ['c2'], ['c3']], [['c1'], ['c2'], ['c2'], ['c3']], 'subclaim'),
                 ([['c1'], ['c2'], ['c3']], [['c2'], ['c1'], ['c3']], 'reordering'),
                 ([['c1'], ['c2'], ['c3']], [['c1', 'c3'], ['c2']], 'noncontiguous'),
                 ([['c1', 'c2'], ['c3']], [['c1'], ['c2', 'c3']], 'many_to_many')]
        for s, t, error in cases:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                self.compile_fixture(s, t)

    def test_no_change_is_not_a_style_success(self):
        with self.assertRaisesRegex(ValueError, 'no_elementary'):
            self.compile_fixture([['c1'], ['c2'], ['c3']], [['c1'], ['c2'], ['c3']])

    def test_private_maps_recompile_without_mutating_parents(self):
        folder = e.ROOT / 'research_runs/operation_expansion_v1'
        if not (folder / 'response_audit_v1/annotations.checked.private.jsonl').exists():
            self.skipTest('Private annotations unavailable')
        key = e.base.check_package(folder / 'package_v1')
        infos = {r['case_id']: r for r in key['records']}
        counts = {'compiled': 0, 'repeated': 0, 'unchanged': 0}
        papers = set()
        for row in e.read_lines(folder / 'response_audit_v1/annotations.checked.private.jsonl'):
            a = row['annotation']
            if a['status'] != 'already_structured':
                continue
            record = infos[row['case_id']]['record']; before = copy.deepcopy((record, a))
            try:
                compiled = e.compile_edits(record, a)
            except ValueError as exc:
                if str(exc) == 'repeated_claim_requires_subclaim_binding': counts['repeated'] += 1
                elif str(exc) == 'no_elementary_packing_change': counts['unchanged'] += 1
                else: raise
            else:
                counts['compiled'] += 1; papers.add(infos[row['case_id']]['paper_key'])
                p = compiled['generation_payload']
                self.assertEqual(set(p), {'protocol', 'source', 'source_units', 'claim_index', 'source_relations', 'edits'})
                self.assertEqual(p['source_relations'], a['relations'])
                self.assertNotIn(record['reference'], compiled['prompt'])
                flat = [c for op in p['edits'] for slot in op['output_slots'] for c in slot['claim_refs']]
                self.assertEqual(sorted(flat), sorted(c['claim_id'] for c in a['claims']))
            self.assertEqual((record, a), before)
        self.assertEqual(counts, {'compiled': 10, 'repeated': 6, 'unchanged': 8})
        self.assertEqual(len(papers), 6)


if __name__ == '__main__':
    unittest.main()
