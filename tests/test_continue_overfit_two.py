import tests  # Make src/ importable without installing ML dependencies.
import json
import tempfile
import unittest
from pathlib import Path

from fama_style.diagnostics.continue_overfit_two import EXPECTED_ADAPTER, validate_prior_adapter


class ContinueOverfitTwoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.adapter = self.root / "adapter"
        self.adapter.mkdir()
        self.config = dict(EXPECTED_ADAPTER, target_modules=["q_proj", "v_proj"])
        self.report = {"model_id": "google/gemma-3-1b-it",
                       "train_ids": ["p001", "p014"], "held_out_id": "p003",
                       "snapshots": [{"step": 60}]}
        self.write_inputs()

    def write_inputs(self):
        (self.adapter / "adapter_config.json").write_text(json.dumps(self.config), encoding="utf-8")
        (self.root / "overfit_two_report.json").write_text(json.dumps(self.report), encoding="utf-8")

    def test_accepts_completed_matching_adapter(self):
        self.assertEqual(validate_prior_adapter(self.adapter, require_weights=False), self.report)

    def test_rejects_wrong_rank_or_incomplete_run(self):
        self.config["r"] = 16
        self.write_inputs()
        with self.assertRaises(ValueError):
            validate_prior_adapter(self.adapter, require_weights=False)
        self.config["r"] = 8
        self.report["snapshots"][-1]["step"] = 30
        self.write_inputs()
        with self.assertRaises(ValueError):
            validate_prior_adapter(self.adapter, require_weights=False)


if __name__ == "__main__":
    unittest.main()
