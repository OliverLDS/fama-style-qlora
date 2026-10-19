import tests  # Make src/ importable without installing ML dependencies.
import json
import tempfile
import unittest
from pathlib import Path

from fama_style.diagnostics.check_epoch_adapters import validate_run


class SavedEpochTests(unittest.TestCase):
    def test_requires_complete_run_and_correct_adapters(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = {"model_id": "google/gemma-3-1b-it", "prompt_format": "rewrite_v1",
                      "train_rows": 54, "validation_rows": 12, "test_rows_used": 0,
                      "snapshots": [{"epoch": e, "generations": [
                          {"source_id": "p008"}, {"source_id": "p050"}]} for e in range(5)]}
            report_path = root / "epoch_validation_report.json"
            report_path.write_text(json.dumps(report))
            config = {"base_model_name_or_path": "google/gemma-3-1b-it",
                      "peft_type": "LORA", "r": 8, "lora_alpha": 16,
                      "lora_dropout": 0.05, "target_modules": ["q_proj", "v_proj"]}
            for epoch in (2, 4):
                adapter = root / f"adapter_epoch_{epoch}"
                adapter.mkdir()
                (adapter / "adapter_config.json").write_text(json.dumps(config))
            self.assertEqual(validate_run(root, require_weights=False), report)
            with self.assertRaises(FileNotFoundError):
                validate_run(root)
            report["snapshots"].pop()
            report_path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                validate_run(root, require_weights=False)
