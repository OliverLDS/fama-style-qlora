"""Synthetic checks plus optional private-release reproducibility."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fama_style.research import repaired_source_bindings as binding


class BindingTests(unittest.TestCase):
    def spec(self):
        return {'claims': [{'claim_id': 'c1', 'source_quote': 'A may rise.'},
                           {'claim_id': 'c2', 'source_quote': 'B falls.'}],
                'relations': [{'relation_id': 'r1', 'kind': 'presentational', 'certainty': 'interpretive',
                               'from_claim_refs': ['c1'], 'to_claim_refs': ['c2'],
                               'from_quote': 'may rise', 'to_quote': 'B falls'}],
                'source_facets': [], 'intra_claim_constraints': []}

    def test_relation_endpoint_owned(self):
        relations, endpoints, _, _ = binding.bind_details('A may rise. B falls.', self.spec())
        self.assertEqual(relations[0]['certainty'], 'interpretive')
        self.assertEqual(endpoints[0]['to']['claim_id'], 'c2')

    def test_reject_misbound_relation_even_when_quote_exists(self):
        spec = self.spec(); spec['relations'][0]['from_quote'] = 'B falls'
        with self.assertRaisesRegex(ValueError, 'owned_evidence'):
            binding.bind_details('A may rise. B falls.', spec)

    def test_repeated_quote_resolved_within_owner(self):
        text = 'A may rise. B may rise.'
        owner = binding.anchored(text, 'B may rise.')[1]
        evidence, bounds = binding.anchored(text, 'may rise', owner)
        self.assertEqual(evidence['occurrence'], 2)
        self.assertEqual(text[bounds['start']:bounds['end']], 'may rise')
        with self.assertRaises(ValueError): binding.anchored(text, 'may rise')

    def test_constraint_cannot_escape_claim(self):
        spec = self.spec(); spec['relations'] = []
        spec['intra_claim_constraints'] = [{'constraint_id': 'q1', 'claim_id': 'c1', 'kind': 'conditional',
            'parts': [{'role': 'antecedent', 'quote': 'may rise'}, {'role': 'consequent', 'quote': 'B falls'}], 'rule': 'keep_condition'}]
        with self.assertRaisesRegex(ValueError, 'owned_evidence'):
            binding.bind_details('A may rise. B falls.', spec)

    def test_constraint_roles_and_order(self):
        spec = self.spec(); spec['relations'] = []
        constraint = {'constraint_id': 'q1', 'claim_id': 'c1', 'kind': 'scoped_modality',
                      'parts': [{'role': 'modal_clause', 'quote': 'A'}, {'role': 'asserted_clause', 'quote': 'may rise'}], 'rule': 'test'}
        spec['intra_claim_constraints'] = [constraint]
        self.assertEqual(len(binding.bind_details('A may rise. B falls.', spec)[3]), 1)
        constraint['parts'][0]['role'] = 'cause'
        with self.assertRaisesRegex(ValueError, 'constraint_roles'):
            binding.bind_details('A may rise. B falls.', spec)
        constraint['parts'] = [{'role': 'modal_clause', 'quote': 'may rise'}, {'role': 'asserted_clause', 'quote': 'A'}]
        with self.assertRaisesRegex(ValueError, 'constraint_order'):
            binding.bind_details('A may rise. B falls.', spec)

    def test_parent_hash_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); (p/'file').write_text('first')
            hashes = {'file': binding.file_sha(p/'file')}
            binding.verify_files(p, hashes)
            (p/'file').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'parent_artifact_changed'): binding.verify_files(p, hashes)

    @unittest.skipUnless((binding.ROOT/'research_runs/repaired_source_bindings_v1/release_v1/manifest.json').exists(), 'Private release unavailable')
    def test_private_release_and_no_reference_payload(self):
        release = binding.ROOT/'research_runs/repaired_source_bindings_v1/release_v1'
        binding.verify_files(release, json.loads((release/'manifest.json').read_text())['files'])
        cards = binding.read_lines(release/'paired_cards.private.jsonl')
        self.assertEqual(sum(len(c['annotation']['claims']) for c in cards), 21)
        for c in cards:
            self.assertFalse(c['training_ready'])
            self.assertEqual(set(c['evidence']), {'source_units', 'claim_index', 'source_relations'})
            for mode in ('ordinary', 'planned'):
                prompt = binding.make_prompt(c['record']['source'], mode, c['evidence'], c['plan'] if mode == 'planned' else None)
                self.assertEqual(binding.base.sha(prompt), c['prompt_validation_hashes'][mode])
                self.assertNotIn(c['record']['reference'], prompt)
        me = next(c for c in cards if c['paper_key'] == 'ME1997')
        self.assertEqual([r['relation_id'] for r in me['annotation']['relations']], ['r1', 'r3'])
        self.assertEqual(me['alignment_caveats'][0]['code'], 'recognition_versus_correction')
        self.assertTrue(next(c for c in cards if c['paper_key']=='NL2003')['context_flags'])

    @unittest.skipUnless((binding.ROOT/'research_runs/repaired_source_bindings_v1/release_v1/manifest.json').exists(), 'Private release unavailable')
    def test_private_rebuild_reproducible(self):
        root = binding.ROOT/'research_runs/repaired_source_bindings_v1'
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'release'
            binding.build(root/'specification_v1/source_bindings.private.json', output)
            manifest = json.loads((root/'release_v1/manifest.json').read_text())
            for name, digest in manifest['files'].items(): self.assertEqual(binding.file_sha(output/name), digest)


if __name__ == '__main__': unittest.main()
