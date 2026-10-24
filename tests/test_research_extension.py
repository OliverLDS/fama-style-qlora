"""Synthetic CPU checks for the claim/style extension; no model or private text."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fama_style.research.claims import ledger, variants, verify, risk_flags, sha
from fama_style.research.signal import fit_profile, overlap
from fama_style.research.prototype import rewrite
from fama_style.research.experiments import prepare, paraphrases, plans
from fama_style.research.run_extension import synthetic, execute, read_split


class ClaimLedgerTests(unittest.TestCase):
    def test_evidence_and_protected_information(self):
        text = 'If demand falls, firms may not increase output by 5%. Costs fall because prices fall.'
        result = ledger(text)
        self.assertFalse(result['independently_semantically_verified'])
        first, second = result['claims']
        self.assertEqual(first['conditions'][0]['text'], 'demand falls')
        for key in ('quantities', 'negation', 'modality', 'condition_cues'):
            self.assertTrue(first[key])
            for span in first[key]:
                self.assertEqual(text[span['start']:span['end']], span['text'])
        self.assertEqual(second['causal_relations'][0]['direction'], 'right_explains_left')
        self.assertFalse(second['causal_relations'][0]['verified_parse'])

    def test_unparsed_condition_retained(self):
        result = ledger('Unless demand rises, firms reduce output.')
        self.assertEqual(result['claims'][0]['conditions'][0]['parse_status'], 'unresolved_scope')

    def test_semantic_mutations_and_role_reversal_never_pass(self):
        source = 'Some firms may not increase output by 5%.'
        for candidate in (source.replace('5%', '10%'), source.replace('may', 'will'),
                          source.replace('not ', ''), source.replace('Some', 'All')):
            self.assertEqual(verify(source, candidate)['status'], 'rejected')
        self.assertEqual(verify('Firms cause losses.', 'Losses cause firms.')['status'], 'unknown')
        self.assertEqual(verify('Costs fall because prices fall.', 'Prices fall because costs fall.')['status'], 'unknown')

    def test_bounded_replay_and_guard(self):
        source = 'Firms report sales. The investors use reports.'
        joined = next(item for item in variants(source) if item['operation'] == 'join_independent_sentences')
        self.assertEqual(verify(source, joined['text'])['status'], 'surface_preserved')
        self.assertFalse(verify(source, joined['text'])['semantic_proof'])
        guarded = 'Firms may report sales. Investors use reports.'
        self.assertNotIn('join_independent_sentences', [item['operation'] for item in variants(guarded)])
        conditional = 'If demand falls, firms reduce output.'
        moved = next(item for item in variants(conditional) if item['operation'] == 'condition_to_end')
        self.assertEqual(verify(conditional, moved['text'])['status'], 'surface_preserved')


class ResearchPreparationTests(unittest.TestCase):
    def setUp(self):
        self.train, self.validation = synthetic()
        self.profile = fit_profile(self.train)

    def test_train_only_fit_and_preparation(self):
        with self.assertRaises(ValueError):
            fit_profile(self.validation)
        with self.assertRaises(ValueError):
            prepare(self.validation, self.profile)
        with self.assertRaises(ValueError):
            read_split('/does/not/exist/test.jsonl', 'test')

    def test_reference_not_used_at_generation(self):
        before = rewrite(self.validation[0]['plain_english'], self.profile)
        self.validation[0]['completion'] = 'A deliberately unrelated validation reference.'
        after = rewrite(self.validation[0]['plain_english'], self.profile)
        self.assertEqual(before, after)
        self.assertEqual(before['style_plan']['fit_split'], 'train')
        self.assertFalse(before['semantic_guarantee'])

    def test_targets_unchanged_and_provenance_replayable(self):
        prepared = prepare(self.train, self.profile)
        parents = {row['id']:row for row in self.train}
        self.assertTrue(prepared['sft'])
        for item in prepared['sft']:
            parent = parents[item['parent_id']]
            self.assertEqual(item['target'], parent['completion'])
            self.assertEqual(item['target_sha256'], sha(parent['completion']))
            self.assertFalse(item['training_ready'])
            text = parent['plain_english']
            for step in item['provenance']['operations']:
                self.assertEqual(step['input_sha256'], sha(text))
                self.assertEqual(verify(text, step['text'])['status'], 'surface_preserved')
                text = step['text']
            self.assertEqual(text, item['input'])

    def test_already_clear_not_copy_negative_and_mutations_quarantined(self):
        rows = copy.deepcopy(self.train)
        rows[0]['completion'] = rows[0]['plain_english']
        rows[2]['completion'] = rows[2]['plain_english'].replace('may', 'will')
        prepared = prepare(rows, fit_profile(rows))
        all_pairs = prepared['preferences'] + prepared['quarantined_preferences']
        self.assertFalse(any(row['parent_id'] == rows[0]['id'] and row['negative_category'] == 'copying' for row in all_pairs))
        bad = [row for row in prepared['quarantined_preferences'] if row['parent_id'] == rows[2]['id']]
        self.assertTrue(bad)
        self.assertTrue(all(row['reference_alignment_flags'] for row in bad))
        self.assertTrue(all(not row['training_ready'] for row in all_pairs))

    def test_plans_do_not_run_training(self):
        for plan in plans().values():
            self.assertFalse(plan['gpu_submission_allowed'])
            self.assertFalse(plan['reserved_test_use'])
        dpo = plans()['B_contrastive_dpo']
        self.assertIsNone(dpo['trl_proposed_options']['max_length'])
        self.assertIn('frozen', dpo['reference_policy'])

    def test_reproducible_dry_run_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory)/'a', Path(directory)/'b'
            one = execute(self.train, self.validation, a, True)
            two = execute(self.train, self.validation, b, True)
            self.assertEqual(one, two)
            self.assertEqual((a/'manifest.json').read_bytes(), (b/'manifest.json').read_bytes())
            manifest = json.loads((a/'manifest.json').read_text())
            for name, digest in manifest['files'].items():
                self.assertEqual(hashlib.sha256((a/name).read_bytes()).hexdigest(), digest)
            self.assertEqual(one['test_rows_used'], 0)
            self.assertEqual(one['gpu_jobs_submitted'], 0)
            with self.assertRaises(ValueError):
                execute(self.train, self.validation, a, True)
            with self.assertRaises(ValueError):
                execute(self.validation, self.validation, Path(directory)/'bad', True)


if __name__ == '__main__':
    unittest.main()
