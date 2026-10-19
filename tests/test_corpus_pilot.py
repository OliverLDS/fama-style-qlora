import tests
import builtins
import copy
import dis
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from fama_style.training import corpus_pilot as pilot
from fama_style.evaluation import corpus_baseline as baseline

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/fama_corpus_v1/curated.jsonl"
BASELINE = ROOT / "experiments/corpus_v1_baseline/outputs/version_4"
SMOKE = ROOT / "archive/corpus_v1_smoke_version_1"


@tests.requires_local_artifacts(DATASET, BASELINE / "baseline_manifest.json", SMOKE / "smoke_status.json")
class CorpusPilotTests(unittest.TestCase):
    def setUp(self):
        self.groups, self.audit, self.original = pilot.load_inputs(DATASET, BASELINE, SMOKE)

    def reports(self):
        schedule = pilot.epoch_batches([r['id'] for r in self.groups['train']])
        status = {'complete': True, 'epochs_completed': 1, 'epochs_planned': 1, 'training_called': True,
                  'optimizer_steps_completed': 124, 'training_examples_seen': 247,
                  'adapter_weights_changed': True, 'adapter_reload_verified': True, 'test_rows_used': 0,
                  'train_order': [key for ids in schedule for key in ids],
                  'updates': [{'step': i+1, 'microbatch_ids': ids, 'loss': 2., 'gradient_norm': .1}
                              for i, ids in enumerate(schedule)]}
        losses = {'loss': 2., 'rows': [{'id': r['id'], 'loss': 2., 'supervised_tokens': 3}
                                     for r in self.groups['validation']]}
        evaluation = {'complete': True, 'original': copy.deepcopy(losses), 'fine_tuned': copy.deepcopy(losses),
                      'reloaded': copy.deepcopy(losses), 'original_validation_loss': 2.,
                      'fine_tuned_validation_loss': 2., 'reloaded_validation_loss': 2., 'test_rows_used': 0,
                      'epochs': 1, 'optimizer_steps': 124, 'train_rows': 247, 'adapter_reload_verified': True}
        comparison = {'complete': True, 'model_id': baseline.MODEL_ID, 'model_revision': pilot.smoke.MODEL_REVISION,
                      'dataset_sha256': baseline.DATASET_SHA256, 'test_rows_used': 0, 'epochs': 1,
                      'generation': self.original['generation'], 'results': [
                          {'id': r['id'], 'source': r['plain_english'], 'reference': r['completion'],
                           'prompt_sha256': baseline.digest(r['prompt']), 'baseline_output': old['output'],
                           'output': 'An adapted output.', 'output_tokens': 4, 'hit_output_limit': False}
                          for r, old in zip(self.groups['validation'], self.original['results'])]}
        return status, evaluation, comparison

    def test_epoch_covers_every_training_pair_once_without_padding_last_update(self):
        ids = [r['id'] for r in self.groups['train']]
        batches = pilot.epoch_batches(ids)
        flattened = [key for batch in batches for key in batch]
        self.assertEqual(len(batches), 124)
        self.assertEqual([len(batch) for batch in batches], [2]*123+[1])
        self.assertEqual(set(flattened), set(ids))
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(pilot.epoch_batches(ids), batches)
        with self.assertRaises(ValueError):
            pilot.epoch_batches(ids[:-1]+[ids[0]])

    def test_modified_smoke_gate_is_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder)
            for name in pilot.SMOKE_HASHES:
                (dest/name).write_bytes((SMOKE/name).read_bytes())
            path = dest/'smoke_status.json'
            path.write_bytes(path.read_bytes()+b'\n')
            with self.assertRaises(ValueError):
                pilot.load_inputs(DATASET, BASELINE, dest)

    def test_completion_rejects_extra_epoch_repeated_data_and_unverified_reload(self):
        status, evaluation, comparison = self.reports()
        pilot.validate_completion(status, evaluation, comparison, self.groups, self.original)
        for key, value in (('epochs_completed', 2), ('epochs_planned', 2), ('optimizer_steps_completed', 123),
                           ('training_examples_seen', 248), ('adapter_reload_verified', False), ('test_rows_used', 1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                pilot.validate_completion(dict(status, **{key:value}), evaluation, comparison, self.groups, self.original)
        changed = copy.deepcopy(status)
        changed['updates'][-1]['microbatch_ids'].append(changed['train_order'][0])
        with self.assertRaises(ValueError):
            pilot.validate_completion(changed, evaluation, comparison, self.groups, self.original)
        for key, value in (('loss', float('nan')), ('gradient_norm', 0.)):
            changed = copy.deepcopy(status)
            changed['updates'][0][key] = value
            with self.assertRaises(ValueError):
                pilot.validate_completion(changed, evaluation, comparison, self.groups, self.original)
        changed = copy.deepcopy(evaluation)
        changed['reloaded']['rows'][0]['loss'] = 2.2
        changed['reloaded']['loss'] = sum(r['loss'] for r in changed['reloaded']['rows'])/21
        changed['reloaded_validation_loss'] = changed['reloaded']['loss']
        with self.assertRaises(ValueError):
            pilot.validate_completion(status, changed, comparison, self.groups, self.original)

    def test_loss_requires_all_heldout_rows_and_correct_token_weighting(self):
        _, evaluation, _ = self.reports()
        loss = evaluation['original']
        for changed in (dict(loss, rows=loss['rows'][:-1]), dict(loss, loss=2.1)):
            with self.assertRaises(ValueError):
                pilot.validate_loss_report(changed, self.groups['validation'])

    def test_generation_gate_rejects_wrong_prompts_test_rows_empty_and_cutoff(self):
        _, _, comparison = self.reports()
        for key, value in (('prompt_sha256','changed'), ('id', self.groups['test'][0]['id']),
                           ('output',''), ('hit_output_limit', True), ('baseline_output','wrong base')):
            changed = copy.deepcopy(comparison)
            changed['results'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                pilot.validate_comparison(changed, self.groups['validation'], self.original)
        with self.assertRaises(ValueError):
            pilot.validate_comparison(dict(comparison, results=comparison['results'][:-1]),
                                      self.groups['validation'], self.original)

    def test_masked_order_review_keeps_claim_and_style_fields_blank_and_separate(self):
        _, _, comparison = self.reports()
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            pilot.save_review(directory, comparison)
            mapping = json.loads((directory/'comparison_key.json').read_text())
            forms = [json.loads(line) for line in (directory/'review_scores.jsonl').read_text().splitlines()]
            self.assertEqual(len(mapping),21)
            self.assertEqual(len(forms),42)
            self.assertTrue(all(set(value.values())=={'base','epoch_1'} for value in mapping.values()))
            self.assertTrue(all(f['claim_preservation_score'] is None and f['style']['concision'] is None for f in forms))
            worksheet = (directory/'claim_style_review.md').read_text()
            for row in self.groups['test']:
                self.assertNotIn(row['id'],worksheet)
            (directory/'review_scores.jsonl').write_text('review in progress\n')
            pilot.save_review(directory,comparison)
            self.assertEqual((directory/'review_scores.jsonl').read_text(),'review in progress\n')

    def test_local_preflight_does_not_access_secret_or_enter_gpu_runtime(self):
        with mock.patch.object(pilot,'run_training',side_effect=AssertionError('GPU runtime entered')):
            with mock.patch.object(baseline,'load_hf_token',side_effect=AssertionError('Secret accessed')):
                pilot.main(['--dataset',str(DATASET),'--baseline-dir',str(BASELINE),'--smoke-dir',str(SMOKE),'--check'])

    def test_runtime_has_no_unresolved_global_names_without_importing_ml(self):
        # Catch stale constants in rarely reached GPU branches during CPU development.
        def check(code):
            for instruction in dis.get_instructions(code):
                if instruction.opname=='LOAD_GLOBAL':
                    self.assertTrue(instruction.argval in vars(pilot) or hasattr(builtins,instruction.argval),
                                    instruction.argval)
            for constant in code.co_consts:
                if isinstance(constant,types.CodeType):
                    check(constant)
        for value in vars(pilot).values():
            if isinstance(value,types.FunctionType) and value.__module__==pilot.__name__:
                check(value.__code__)


if __name__=='__main__':
    unittest.main()
