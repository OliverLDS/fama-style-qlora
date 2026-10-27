"""Default policy, evidence/plan boundaries and matched historical prompt replay."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fama_style.prompting import economics as e


class EconomicsPromptingTests(unittest.TestCase):
    def fixture(self):
        source = 'Costs may rise. Investors may trade less.'
        parts = ['Costs may rise.', 'Investors may trade less.']; cursor = 0; claims = []
        for i, part in enumerate(parts, 1):
            start = source.index(part, cursor); end = start + len(part); cursor = end
            claims.append({'claim_id': 'c' + str(i), 'composite': False, 'epistemic_status': 'hypothesis',
                           'source_spans': [{'start': start, 'end': end, 'quote': part}], 'protected': []})
        evidence = {'source_units': [{'unit_id': 'S' + str(i), 'text': p} for i, p in enumerate(parts, 1)],
                    'claim_index': claims, 'source_relations': []}
        plan = [{'op': 'MERGE', 'source_unit_ids': ['S1', 'S2'], 'output_slots': [{'claim_refs': ['c1', 'c2']}]}]
        return source, evidence, plan

    def test_default_is_ordinary_and_plan_requires_explicit_opt_in(self):
        source, evidence, plan = self.fixture()
        prompt = e.make_prompt(source); self.assertIn(source, prompt); self.assertNotIn('Sentence-packing edit plan', prompt)
        self.assertEqual(e.make_prompt(source, evidence=evidence), e.make_prompt(source, 'ordinary', evidence))
        with self.assertRaises(ValueError): e.make_prompt(source, evidence=evidence, plan=plan)
        with self.assertRaises(ValueError): e.make_prompt(source, 'planned')
        with self.assertRaises(ValueError): e.make_prompt(' ')
        planned = e.make_prompt(source, 'planned', evidence, plan)
        self.assertIn('MERGE on source sentence(s) S1, S2', planned)

    def test_reference_fields_and_changed_source_evidence_rejected(self):
        source, evidence, plan = self.fixture()
        for mutate in ('reference', 'quote', 'coverage'):
            changed = copy.deepcopy(evidence)
            if mutate == 'reference': changed['reference'] = 'Private target'
            if mutate == 'quote': changed['claim_index'][0]['source_spans'][0]['quote'] = 'Invented claim'
            if mutate == 'coverage': changed['source_units'] = changed['source_units'][:1]
            with self.subTest(mutate=mutate), self.assertRaises(ValueError): e.make_prompt(source, evidence=changed)

    def test_plan_requires_full_coverage_known_operations_and_order(self):
        source, evidence, plan = self.fixture()
        for mutate in ('omit', 'unknown', 'order', 'merge_shape'):
            changed = copy.deepcopy(plan)
            if mutate == 'omit': changed[0]['output_slots'][0]['claim_refs'] = ['c1']
            if mutate == 'unknown': changed[0]['op'] = 'INVENT_CAUSE'
            if mutate == 'order': changed[0]['source_unit_ids'].reverse()
            if mutate == 'merge_shape': changed[0]['op'] = 'KEEP'
            with self.subTest(mutate=mutate), self.assertRaises(ValueError): e.make_prompt(source, 'planned', evidence, changed)

    def test_ambiguous_split_does_not_guess_subclaims(self):
        source = 'Costs rise and prices fall.'
        claims = [{'claim_id': c, 'composite': True, 'epistemic_status': 'assertion', 'protected': [],
                   'source_spans': [{'start': 0, 'end': len(source), 'quote': source}]} for c in ['c1', 'c2']]
        evidence = {'source_units': [{'unit_id': 'S1', 'text': source}], 'claim_index': claims, 'source_relations': []}
        plan = [{'op': 'SPLIT', 'source_unit_ids': ['S1'], 'output_slots': [{'claim_refs': ['c1']}, {'claim_refs': ['c2']}]}]
        with self.assertRaisesRegex(ValueError, 'distinct grounded claims'): e.make_prompt(source, 'planned', evidence, plan)

    def test_private_evidence_prompts_match_all_twelve_frozen_jobs(self):
        p = Path(__file__).resolve().parents[1] / 'research_runs/paper_plan_comparison_v1/package_v1'
        if not p.exists(): self.skipTest('Private research data absent')
        key = e.comparison.check_package(p)
        for row in key['rows']:
            payload = row['payload']; evidence = {k: payload[k] for k in ['source_units', 'claim_index', 'source_relations']}
            for mode, condition in zip(e.MODES, e.comparison.CONDITIONS):
                plan = payload['edits'] if mode == 'planned' else None
                self.assertEqual(e.make_prompt(row['source'], mode, evidence, plan), e.comparison.prompt_for(row, condition))

    def test_cli_writes_prompt_only_and_rejects_overwriting(self):
        source, evidence, plan = self.fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); f = root / 'source.txt'; f.write_text(source); output = root / 'out'
            e.main(['--source-file', str(f), '--output-dir', str(output)])
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['mode'], 'ordinary'); self.assertEqual(manifest['model_calls'], 0)
            self.assertFalse(manifest['training_called']); self.assertFalse(manifest['semantic_preservation_verified'])
            self.assertEqual((output / 'prompt.txt').read_text(), e.make_prompt(source))
            with self.assertRaises(ValueError): e.prepare(f, output)
            self.assertNotIn('torch', sys.modules)


if __name__ == '__main__': unittest.main()
