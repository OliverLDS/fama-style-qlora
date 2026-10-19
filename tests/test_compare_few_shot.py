import tests  # Make src/ importable without installing ML dependencies.
import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from fama_style.evaluation.compare_few_shot import (DEMO_IDS, MODEL_IDS, load_experiment, make_prompt,
                              validate_completion, write_review, INSTRUCTION,
                              validate_baseline, validate_gpu_placement, validate_probe)

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "historical" / "fama_jensen_1983_pilot_instructed.jsonl"


class FewShotTests(unittest.TestCase):
    def test_rejects_cpu_offload_or_single_gpu_placement(self):
        validate_gpu_placement({"embeddings": 0, "layers": "cuda:1"})
        for mapping in ({}, {"layers": 0}, {"a": 0, "b": 1, "head": "cpu"},
                        {"a": 0, "b": "disk"}, {"a": 0, "b": 2}):
            with self.assertRaises(RuntimeError):
                validate_gpu_placement(mapping)

    def test_first_generation_must_be_nonempty_and_finish_within_budget(self):
        validate_probe({"output": "A valid paragraph.", "hit_output_limit": False})
        for result in ({"output": " ", "hit_output_limit": False},
                       {"output": "Incomplete", "hit_output_limit": True}):
            with self.assertRaises(RuntimeError):
                validate_probe(result)

    @tests.requires_local_artifacts(DATASET)
    def test_baseline_requires_identical_prompts_and_valid_precision(self):
        demos, validation = load_experiment(DATASET)
        report = {"model_id": MODEL_IDS[1], "complete": True, "training_called": False,
                  "test_rows_used": 0, "quantization": "NF4 double quantization, FP32 compute",
                  "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
                  "instruction": INSTRUCTION, "demonstrations": demos,
                  "generation": {"do_sample": False, "max_new_tokens": 512}, "results": []}
        for row in validation:
            report["results"].append({"source_ids": row["source_ids"],
                "source": row["plain_english"], "reference": row["completion"],
                "output": "A baseline paragraph.", "hit_output_limit": False,
                "prompt_sha256": hashlib.sha256(make_prompt(demos, row["plain_english"]).encode()).hexdigest()})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "baseline.json"
            path.write_text(json.dumps(report))
            validate_baseline(path, DATASET, demos, validation)
            report["results"][0]["prompt_sha256"] = "different"
            path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                validate_baseline(path, DATASET, demos, validation)
            report["results"][0]["prompt_sha256"] = hashlib.sha256(
                make_prompt(demos, validation[0]["plain_english"]).encode()).hexdigest()
            report["quantization"] = "NF4 double quantization, FP16 compute"
            path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                validate_baseline(path, DATASET, demos, validation)

    def test_rejects_empty_output_run_even_when_all_rows_were_processed(self):
        with self.assertRaises(RuntimeError):
            validate_completion({"results": [{"output": ""} for _ in range(12)]})
        with self.assertRaises(RuntimeError):
            validate_completion({"results": [{"output": "Text"} for _ in range(11)]})
        validate_completion({"results": [{"output": "Text"} for _ in range(12)]})

    @tests.requires_local_artifacts(DATASET)
    def test_demonstrations_are_training_only_and_targets_do_not_leak(self):
        demos, validation = load_experiment(DATASET)
        self.assertEqual(tuple(d["source_id"] for d in demos), DEMO_IDS)
        self.assertEqual(len(validation), 12)
        self.assertTrue(all(r["split"] == "validation" for r in validation))
        for row in validation:
            prompt = make_prompt(demos, row["plain_english"])
            self.assertIn(row["plain_english"], prompt)
            self.assertNotIn(row["completion"], prompt)
            self.assertEqual(prompt.count("Example "), 3)

    @tests.requires_local_artifacts(DATASET)
    def test_reviewed_sources_keep_original_provenance(self):
        demos, _ = load_experiment(DATASET)
        self.assertEqual(demos[0]["source"], demos[0]["original_source"])
        self.assertIn("is costly", demos[1]["source"])
        self.assertIn("costs money", demos[1]["original_source"])
        self.assertIn("imprecisely", demos[2]["source"])
        self.assertNotIn("imprecisely", demos[2]["original_source"])

    @tests.requires_local_artifacts(DATASET)
    def test_rejects_validation_demonstration_even_if_split_counts_match(self):
        rows = [json.loads(line) for line in DATASET.read_text().splitlines()]
        next(r for r in rows if r["source_ids"] == ["p014"])["split"] = "validation"
        next(r for r in rows if r["source_ids"] == ["p008"])["split"] = "train"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows))
            with self.assertRaises(ValueError):
                load_experiment(path)

    @tests.requires_local_artifacts(DATASET)
    def test_worksheet_contains_both_models_and_blank_claim_checks(self):
        _, validation = load_experiment(DATASET)
        row = validation[0]
        result = {"source_ids": row["source_ids"], "source": row["plain_english"],
                  "reference": row["completion"], "output": "A candidate rewrite."}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for model in MODEL_IDS:
                (root / (model.split("/")[-1] + ".json")).write_text(json.dumps(
                    {"model_id": model, "results": [result]}))
            write_review(root)
            worksheet = (root / "claim_style_review.md").read_text()
        self.assertTrue(all(model in worksheet for model in MODEL_IDS))
        self.assertIn("[ ] preserved [ ] changed [ ] omitted", worksheet)
        self.assertNotIn("[x]", worksheet)
