"""Core contracts using synthetic prose only; no private artifacts or ML imports."""
import tests
import copy
import unittest
from pathlib import Path

from fama_style.evaluation import corpus_baseline as baseline
from fama_style.evaluation import source_control_review as review
from fama_style.training import corpus_pilot as pilot
from fama_style.training import corpus_smoke as smoke
from tests.test_corpus_baseline import FakeProcessor
from scripts.review_public_files import allowed, audit_file


class PublicContracts(unittest.TestCase):
    def test_complete_target_is_supervised_and_prompt_is_masked(self):
        row = {"id": "synthetic", "paper_key": "SYN", "split": "train",
               "prompt": "Rewrite without changing the claim.", "completion": "Firms may reduce output when demand falls."}
        processor = FakeProcessor()
        expected = baseline.prepare_pair(processor, row)
        batch, audit = smoke.encode_pair(processor, row, expected)
        self.assertEqual(batch["labels"][:audit["prompt_tokens"]], [-100]*audit["prompt_tokens"])
        self.assertEqual(processor.decode(batch["labels"][audit["prompt_tokens"]:]), row["completion"])
        with self.assertRaises(ValueError):
            baseline.prepare_pair(FakeProcessor(shortened_target=True), row)

    def test_one_epoch_uses_all_247_examples_once_with_short_final_update(self):
        ids = [f"synthetic-{i}" for i in range(247)]
        batches = pilot.epoch_batches(ids)
        self.assertEqual(len(batches), 124)
        self.assertEqual(len(batches[-1]), 1)
        self.assertCountEqual([key for batch in batches for key in batch], ids)
        with self.assertRaises(ValueError):
            pilot.epoch_batches(ids[:-1]+[ids[0]])

    def test_source_control_mapping_and_analysis_with_synthetic_data(self):
        comparison = {"results": [{"id": f"synthetic-{i}", "source": f"Input claim {i}.",
                                   "reference": f"Reference expression {i}.", "baseline_output": f"Base expression {i}.",
                                   "output": f"Adapted expression {i}."} for i in range(21)]}
        bundle, mapping, scores, manifest = review.make_review(comparison)
        review.validate_bundle(bundle, mapping, manifest)
        with self.assertRaises(ValueError):
            review.analyze(bundle, mapping, scores)
        for row in scores["ratings"]:
            row.update(claims=4, economic_precision=2, analytical_structure=1, concision=2)
        summary = review.analyze(bundle, mapping, scores)
        self.assertEqual(summary["pairwise"]["epoch_1_vs_unchanged_input"]["claims"]["ties"], 21)
        changed = copy.deepcopy(mapping)
        changed["assignments"]["synthetic-0"]["A"] = "wrong"
        with self.assertRaises(ValueError):
            review.validate_bundle(bundle, changed, manifest)

    def test_public_file_audit_rejects_artifacts_and_token_patterns(self):
        self.assertFalse(allowed("archive/paper.pdf"))
        self.assertFalse(allowed("reviews/source_control_v1/review.html"))
        self.assertTrue(allowed("src/fama_style/training/corpus_pilot.py"))
        # Construct fake markers rather than storing a token-shaped literal.
        marker = ("hf_"+"x"*30).encode()
        self.assertIn("Hugging Face token pattern", audit_file("README.md",marker))
        self.assertTrue(audit_file("kaggle/test.ipynb",b'{"cells":[{"outputs":[{}],"execution_count":1}]}'))


if __name__ == "__main__":
    unittest.main()
