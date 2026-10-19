import tests
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fama_style.training import corpus_smoke as smoke
from fama_style.evaluation import corpus_baseline as baseline
from tests.test_corpus_baseline import FakeProcessor

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/fama_corpus_v1/curated.jsonl"
BASELINE = ROOT / "experiments/corpus_v1_baseline/outputs/version_4"


@tests.requires_local_artifacts(DATASET, BASELINE / "baseline_manifest.json")
class CorpusSmokeTests(unittest.TestCase):
    def setUp(self):
        self.groups, self.audit, self.train_ids, self.validation_ids = smoke.load_inputs(DATASET, BASELINE)

    def test_selection_is_exactly_longest_training_rows_and_heldout_loss_rows(self):
        self.assertEqual(self.train_ids, self.audit["longest_train_ids"])
        records = {r["id"]: r for r in self.audit["records"]}
        self.assertEqual(len(set(self.validation_ids)), 2)
        self.assertFalse(set(self.train_ids) & set(self.validation_ids))
        self.assertTrue(all(records[key]["split"] == "train" for key in self.train_ids))
        self.assertTrue(all(records[key]["split"] == "validation" for key in self.validation_ids))
        changed = copy.deepcopy(self.audit["records"])
        next(r for r in changed if r["id"] == self.train_ids[0])["split"] = "test"
        with self.assertRaises(ValueError):
            smoke.choose_examples(self.groups, changed)

    def test_rejects_modified_or_incomplete_baseline_before_ml_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for source in BASELINE.iterdir():
                if source.is_file():
                    (directory / source.name).write_bytes(source.read_bytes())
            (directory / "baseline.json").write_text('{}')
            with self.assertRaises(ValueError):
                smoke.load_inputs(DATASET, directory)

    def test_training_batch_masks_prompt_and_retains_complete_target(self):
        row = {"id": "train-example", "paper_key": "example", "split": "train",
               "prompt": "Small test prompt", "completion": "Complete original target."}
        processor = FakeProcessor()
        expected = baseline.prepare_pair(processor, row)
        batch, audit = smoke.encode_pair(processor, row, expected)
        prefix = expected["prompt_tokens"]
        self.assertEqual(batch["labels"][:prefix], [-100]*prefix)
        self.assertEqual(batch["labels"][prefix:], batch["input_ids"][prefix:])
        self.assertEqual(processor.decode(batch["labels"][prefix:]), row["completion"])
        self.assertGreater(audit["shifted_supervised_tokens"], 0)
        with self.assertRaises(ValueError):
            smoke.encode_pair(processor, row, dict(expected, full_tokens=expected["full_tokens"]+1))

    def test_actual_training_batch_is_rechecked_after_audit(self):
        row = {"id": "train-example", "paper_key": "example", "split": "train",
               "prompt": "Small test prompt", "completion": "Complete original target."}
        processor = FakeProcessor()
        expected = baseline.prepare_pair(processor, row)
        render = processor.apply_chat_template
        calls = []
        def altered(*args, **kwargs):
            calls.append(True)
            batch = render(*args, **kwargs)
            if len(calls) == 3:
                batch["input_ids"][0][-2] += 1
            return batch
        processor.apply_chat_template = altered
        with self.assertRaises(ValueError):
            smoke.encode_pair(processor, row, expected)

    def complete_reports(self):
        status = {"complete": True, "optimizer_steps_completed": 5, "test_rows_used": 0,
                  "adapter_weights_changed": True, "adapter_reload_verified": True,
                  "trained_ids": self.train_ids, "validation_ids": self.validation_ids,
                  "updates": [{"step": step+1, "loss": 2.0, "gradient_norm": 0.1,
                               "microbatch_ids": [self.train_ids[(step*2+i)%3] for i in range(2)]}
                              for step in range(5)]}
        evaluation = {"original_validation_loss": 2.0, "fine_tuned_validation_loss": 2.1,
                      "reloaded_validation_loss": 2.1, "validation_ids": self.validation_ids,
                      "test_rows_used": 0, "train_rows": 3, "optimizer_steps": 5,
                      "adapter_reload_verified": True}
        return status, evaluation

    def test_completion_requires_five_finite_updates_and_matching_reload(self):
        status, evaluation = self.complete_reports()
        # Improvement is deliberately not a success requirement for five updates.
        smoke.validate_completion(status, evaluation)
        for key, value in (("complete", False), ("optimizer_steps_completed", 4),
                           ("adapter_weights_changed", False), ("adapter_reload_verified", False),
                           ("test_rows_used", 1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                smoke.validate_completion(dict(status, **{key: value}), evaluation)
        for key, value in (("gradient_norm", 0.0), ("gradient_norm", float('nan')),
                           ("loss", float('inf')), ("microbatch_ids", [self.validation_ids[0]]*2)):
            changed = copy.deepcopy(status)
            changed["updates"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                smoke.validate_completion(changed, evaluation)
        for key, value in (("reloaded_validation_loss", 2.2), ("original_validation_loss", float('nan')),
                           ("test_rows_used", 1), ("optimizer_steps", 6)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                smoke.validate_completion(status, dict(evaluation, **{key: value}))

    def test_local_check_never_loads_secret_or_ml_runtime(self):
        with mock.patch.object(smoke, 'run_training', side_effect=AssertionError('GPU runtime entered')):
            with mock.patch.object(baseline, 'load_hf_token', side_effect=AssertionError('Secret accessed')):
                smoke.main(['--dataset', str(DATASET), '--baseline-dir', str(BASELINE), '--check'])

    def test_adapter_targets_only_language_attention_qv(self):
        import re
        self.assertTrue(re.fullmatch(smoke.TARGET_PATTERN, 'model.language_model.layers.0.self_attn.q_proj'))
        self.assertFalse(re.fullmatch(smoke.TARGET_PATTERN, 'model.vision_tower.layers.0.self_attn.q_proj'))
        self.assertFalse(re.fullmatch(smoke.TARGET_PATTERN, 'model.language_model.layers.0.mlp.v_proj'))


if __name__ == '__main__':
    unittest.main()
