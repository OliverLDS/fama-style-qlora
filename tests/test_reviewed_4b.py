import tests  # Make src/ importable without installing ML dependencies.
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from fama_style.evaluation.compare_few_shot import INSTRUCTION, make_prompt
from scripts.data.review_fama_training import build_review, REVIEWED_SHA256
from fama_style.training.smoke_4b_reviewed import load_inputs, choose_smoke_ids, TARGET_PATTERN
import re

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "data/historical/fama_jensen_1983_pilot_instructed.jsonl"


class ReviewedDatasetTests(unittest.TestCase):
    @tests.requires_local_artifacts(ORIGINAL, ROOT / "data/historical/reviewed_training_decisions.private.json")
    def test_every_training_row_is_reviewed_and_holdouts_are_preserved(self):
        rows, audit = build_review(ORIGINAL)
        original = [json.loads(x) for x in ORIGINAL.read_text().splitlines()]
        self.assertEqual(len(audit["decisions"]), 54)
        self.assertEqual(sum(r["role"] == "training" for r in audit["decisions"]), 48)
        self.assertEqual(sum(r["role"] == "excluded" for r in audit["decisions"]), 3)
        self.assertEqual(sum(r["role"] == "reserved_demonstration" for r in audit["decisions"]), 3)
        self.assertEqual([r for r in rows if r["split"] == "test"],
                         [r for r in original if r["split"] == "test"])
        for row, old in zip([r for r in rows if r["split"] == "validation"],
                            [r for r in original if r["split"] == "validation"]):
            self.assertEqual((row["plain_english"], row["completion"]),
                             (old["plain_english"], old["completion"]))
        train_ids = {r["source_ids"][0] for r in rows if r["split"] == "train"}
        self.assertFalse(train_ids & {"p009", "p015", "p076", "p014", "p018", "p075"})
        p001 = next(r for r in rows if r["source_ids"] == ["p001"])
        self.assertIn("Adam Smith", p001["plain_english"])
        self.assertIn("Jensen and Meckling", p001["plain_english"])

    @tests.requires_local_artifacts(ORIGINAL, ROOT / "data/historical/reviewed_training_decisions.private.json")
    def test_changed_original_requires_new_review(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "changed.jsonl"
            path.write_bytes(ORIGINAL.read_bytes() + b"\n")
            with self.assertRaises(ValueError):
                build_review(path)

    @tests.requires_local_artifacts(ORIGINAL, ROOT / "data/historical/reviewed_training_decisions.private.json")
    def test_baseline_alignment_and_demonstration_exclusion_are_enforced(self):
        rows, audit = build_review(ORIGINAL)
        val = [r for r in rows if r["split"] == "validation"]
        baseline = {"model_id": "google/gemma-3-4b-it", "complete": True,
                    "training_called": False, "test_rows_used": 0,
                    "dataset_sha256": REVIEWED_SHA256,
                    "quantization": "NF4 double quantization, FP32 compute",
                    "generation": {"do_sample": False, "max_new_tokens": 512},
                    "instruction": INSTRUCTION, "demonstrations": audit["demonstrations"],
                    "results": [{"source_ids": r["source_ids"], "source": r["plain_english"],
                                 "reference": r["completion"], "output": "A baseline output.",
                                 "hit_output_limit": False,
                                 "prompt_sha256": hashlib.sha256(r["prompt"].encode()).hexdigest()}
                                for r in val]}
        with tempfile.TemporaryDirectory() as folder:
            dataset = Path(folder) / "reviewed.jsonl"
            prior = Path(folder) / "baseline.json"
            prior.write_text(json.dumps(baseline))
            dataset.write_text("\n".join(json.dumps(r) for r in rows))
            groups, _ = load_inputs(dataset, prior)
            self.assertEqual(len(groups["train"]), 48)
            rows[0]["source_ids"] = ["p014"]
            dataset.write_text("\n".join(json.dumps(r) for r in rows))
            with self.assertRaises(ValueError):
                load_inputs(dataset, prior)

    def test_memory_probe_includes_longest_training_row_and_only_three_rows(self):
        selected = choose_smoke_ids({"p031": 10, "p060": 20, "p039": 1500, "p001": 800})
        self.assertEqual(selected, ("p039", "p031", "p060"))
        collision = choose_smoke_ids({"p031": 1500, "p060": 20, "p039": 1000})
        self.assertEqual(len(set(collision)), 3)
        self.assertIn("p031", collision)

    def test_adapter_pattern_excludes_vision_and_nonattention_layers(self):
        self.assertIsNotNone(re.fullmatch(TARGET_PATTERN, "model.language_model.layers.0.self_attn.q_proj"))
        self.assertIsNone(re.fullmatch(TARGET_PATTERN, "model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj"))
        self.assertIsNone(re.fullmatch(TARGET_PATTERN, "model.language_model.layers.0.mlp.gate_proj"))
