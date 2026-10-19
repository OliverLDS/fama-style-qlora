import tests
import copy
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from fama_style.evaluation import corpus_baseline as baseline
from fama_style.prompting.corpus import INSTRUCTION, make_prompt

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/fama_corpus_v1/curated.jsonl"


class FakeProcessor:
    """A reversible toy chat renderer, not a stand-in for real token lengths."""
    def __init__(self, bad_prefix=False, shortened_target=False):
        self.tokenizer = self
        self.bad_prefix = bad_prefix
        self.shortened_target = shortened_target

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["truncation"] is False
        prefix = [1] + [ord(c) + 10 for c in messages[0]["content"][0]["text"]] + [2]
        if len(messages) == 1:
            return {"input_ids": [prefix]}
        target = messages[1]["content"][0]["text"]
        if self.shortened_target:
            target = target[:-8]
        ids = prefix + [ord(c) + 10 for c in target] + [3]
        if self.bad_prefix:
            ids[0] = 7
        return {"input_ids": [ids]}

    def decode(self, ids, **kwargs):
        return "".join(chr(i - 10) for i in ids if i > 9)


@tests.requires_local_artifacts(DATASET)
class CorpusBaselineTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.groups = baseline.load_corpus(DATASET)

    def test_placement_checks_actual_tensors_without_device_map_metadata(self):
        def tensor(kind="cuda", index=0):
            return types.SimpleNamespace(device=types.SimpleNamespace(type=kind, index=index))
        model = types.SimpleNamespace(parameters=lambda: iter([tensor(), tensor()]),
                                      buffers=lambda: iter([tensor()]),
                                      get_input_embeddings=lambda: types.SimpleNamespace(weight=tensor()))
        result = baseline.validate_model_placement(model)
        self.assertFalse(result["device_map_present"])
        self.assertEqual(result["parameter_tensors_checked"], 2)
        self.assertEqual(result["buffer_tensors_checked"], 1)
        model.hf_device_map = {"": 0}
        self.assertTrue(baseline.validate_model_placement(model)["device_map_present"])
        for device in ("cpu", "disk", "cuda:1"):
            model.hf_device_map = {"": device}
            with self.assertRaises(baseline.PreflightError):
                baseline.validate_model_placement(model)
        del model.hf_device_map
        for kind, index in (("cpu", None), ("meta", None), ("cuda", 1)):
            for field in ("parameters", "buffers"):
                bad = copy.copy(model)
                setattr(bad, field, lambda: iter([tensor(kind, index)]))
                with self.assertRaises(baseline.PreflightError):
                    baseline.validate_model_placement(bad)
        model.parameters = lambda: iter([])
        with self.assertRaises(baseline.PreflightError):
            baseline.validate_model_placement(model)

    def test_cuda_allocator_is_initialized_before_resetting_peak_stats(self):
        torch = mock.Mock()
        initialized = []
        torch.cuda.init.side_effect = lambda: initialized.append(True)
        def reset(device):
            if not initialized:
                raise RuntimeError("invalid argument to reset_peak_memory_stats")
        torch.cuda.reset_peak_memory_stats.side_effect = reset
        baseline.initialize_cuda(torch)
        self.assertEqual(torch.mock_calls, [mock.call.cuda.init(), mock.call.cuda.set_device(0),
                                          mock.call.manual_seed(42), mock.call.cuda.reset_peak_memory_stats(0)])

    def test_failure_diagnostics_never_save_exception_text_or_locals(self):
        marker = "test-only-private-value"
        try:
            raise RuntimeError("invalid argument to reset_peak_memory_stats " + marker)
        except RuntimeError as exc:
            details = baseline.safe_failure_details(exc)
        self.assertEqual(details["error_code"], "cuda_allocator_not_initialized")
        self.assertNotIn(marker, json.dumps(details))
        self.assertNotIn("reset_peak_memory_stats", json.dumps(details))
        self.assertEqual(baseline.safe_failure_details(RuntimeError(marker))["error_code"],
                         "unclassified_runtime_failure")

    def complete_report(self):
        return {"complete": True, "model_id": baseline.MODEL_ID, "model_revision": "frozen-revision",
                "dataset_sha256": baseline.DATASET_SHA256, "prompt_format": baseline.FORMAT,
                "instruction": INSTRUCTION, "training_called": False,
                "test_rows_generated": 0, "test_rows_scored": 0, "demonstrations": [],
                "quantization": "NF4 double quantization, FP32 compute",
                "generation": {"do_sample": False, "max_new_tokens": 512},
                "results": [{"id": r["id"], "source": r["plain_english"], "reference": r["completion"],
                             "prompt_sha256": baseline.digest(r["prompt"]), "output": "Toy output.",
                             "output_tokens": 3, "hit_output_limit": False}
                            for r in self.groups["validation"]]}

    def test_revision_is_resolved_publicly_when_config_has_no_commit_hash(self):
        # Regression: Transformers 5.20 configurations deliberately lack this private field.
        config = types.SimpleNamespace(model_type="gemma3")
        self.assertFalse(hasattr(config, "_commit_hash"))
        sha = "a" * 40
        api = mock.Mock()
        api.model_info.return_value = types.SimpleNamespace(sha=sha)
        config_class = mock.Mock()
        config_class.from_pretrained.return_value = config
        progress = mock.Mock()
        actual, revision = baseline.load_pinned_config(api, config_class, "test-only-marker", progress)
        self.assertIs(actual, config)
        self.assertEqual(revision, sha)
        api.model_info.assert_called_once_with(baseline.MODEL_ID, revision="main", timeout=30)
        config_class.from_pretrained.assert_called_once_with(
            baseline.MODEL_ID, revision=sha, token="test-only-marker")
        progress.assert_called_once_with("loading_model_config", model_id=baseline.MODEL_ID, model_revision=sha)

    def test_unresolved_revision_does_not_fall_back_to_a_mutable_branch(self):
        for revision in (None, "main", "", "not-a-commit"):
            api = mock.Mock()
            api.model_info.return_value = types.SimpleNamespace(sha=revision)
            config_class = mock.Mock()
            with self.assertRaises(baseline.PreflightError) as error:
                baseline.load_pinned_config(api, config_class, "test-only-marker")
            self.assertEqual(error.exception.code, "hub_revision_missing_or_invalid")
            config_class.from_pretrained.assert_not_called()

    def test_frozen_hash_prompt_and_paper_splits(self):
        self.assertEqual(len(self.rows), 313)
        self.assertEqual(INSTRUCTION, (ROOT / "data/fama_corpus_v1/rewrite_instruction.txt").read_text().strip())
        for row in self.rows:
            self.assertEqual(row["prompt"], make_prompt(row["plain_english"]))
        changed = copy.deepcopy(self.rows)
        train = next(r for r in changed if r["split"] == "train")
        validation = next(r for r in changed if r["split"] == "validation")
        train["split"], validation["split"] = validation["split"], train["split"]
        with self.assertRaises(ValueError):
            baseline.validate_rows(changed)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "curated.jsonl"
            path.write_bytes(DATASET.read_bytes() + b"\n")
            with self.assertRaises(ValueError):
                baseline.load_corpus(path)

    def test_audit_rejects_bad_boundary_and_missing_target_text(self):
        row = self.groups["validation"][0]
        audit = baseline.prepare_pair(FakeProcessor(), row)
        self.assertEqual(audit["full_tokens"], audit["prompt_tokens"] + audit["target_tokens"])
        self.assertTrue(audit["decoded_target_matches"])
        self.assertFalse(audit["truncated"])
        self.assertNotIn("reference", audit)
        for processor in (FakeProcessor(bad_prefix=True), FakeProcessor(shortened_target=True)):
            with self.assertRaises(ValueError):
                baseline.prepare_pair(processor, row)

    def test_budget_audit_includes_test_lengths_but_not_their_generation_needs(self):
        records = [{"id": r["id"], "split": r["split"], "prompt_tokens": 400,
                    "target_tokens": 100, "full_tokens": 500} for r in self.rows]
        test = next(r for r in records if r["split"] == "test")
        test.update(full_tokens=2200, target_tokens=1800)
        audit = baseline.summarize_audit(records)
        self.assertEqual(audit["recommended_max_length"], 3072)
        self.assertEqual(audit["baseline_max_new_tokens"], 512)
        self.assertEqual(audit["test_rows_length_audited"], 45)
        self.assertEqual(audit["test_rows_generated"], 0)
        self.assertEqual(audit["limits"]["2048"]["test"], 1)
        self.assertTrue(audit["fits_context_budget"])
        test["full_tokens"] = 4097
        self.assertFalse(baseline.summarize_audit(records)["fits_context_budget"])

    def test_baseline_rejects_missing_swapped_test_cutoff_and_incompatible_runs(self):
        report = self.complete_report()
        baseline.validate_baseline(report, self.groups["validation"])
        for field, value in (("complete", False), ("training_called", True),
                             ("test_rows_generated", 1), ("quantization", "FP16"),
                             ("demonstrations", ["unexpected demo"])):
            altered = copy.deepcopy(report)
            altered[field] = value
            with self.assertRaises(ValueError):
                baseline.validate_baseline(altered, self.groups["validation"])
        for mutation in ("missing", "test_id", "empty", "cutoff", "prompt_hash"):
            altered = copy.deepcopy(report)
            if mutation == "missing":
                altered["results"].pop()
            elif mutation == "test_id":
                altered["results"][0]["id"] = self.groups["test"][0]["id"]
            elif mutation == "empty":
                altered["results"][0]["output"] = ""
            elif mutation == "cutoff":
                altered["results"][0]["hit_output_limit"] = True
            else:
                altered["results"][0]["prompt_sha256"] = "changed"
            with self.assertRaises(ValueError):
                baseline.validate_baseline(altered, self.groups["validation"])

    def test_worksheet_separates_claims_and_style_and_contains_no_test_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            baseline.save_outputs(output, self.complete_report(), self.groups["validation"])
            text = (output / "claim_style_review.md").read_text()
            self.assertIn("### Claim preservation", text)
            self.assertIn("### Style", text)
            self.assertTrue(all(r["id"] not in text for r in self.groups["test"]))
            forms = [json.loads(l) for l in (output / "review_scores.jsonl").read_text().splitlines()]
            self.assertEqual(len(forms), 21)
            self.assertTrue(all(r["claim_preservation_score"] is None for r in forms))
            (output / "review_scores.jsonl").write_text("preserve manual edits\n")
            baseline.save_outputs(output, self.complete_report(), self.groups["validation"])
            self.assertEqual((output / "review_scores.jsonl").read_text(), "preserve manual edits\n")

    def test_local_check_does_not_enter_runtime_or_access_secret(self):
        with mock.patch.object(baseline, "run_training", side_effect=AssertionError("runtime entered")), \
                mock.patch.object(baseline, "load_hf_token", side_effect=AssertionError("secret accessed")):
            baseline.main(["--dataset", str(DATASET), "--check"])

    def test_kaggle_secret_is_preferred_and_error_details_are_not_exposed(self):
        client = mock.Mock()
        module = types.ModuleType("kaggle_secrets")
        module.UserSecretsClient = mock.Mock(return_value=client)
        with mock.patch.dict("sys.modules", {"kaggle_secrets": module}), \
                mock.patch.object(baseline.Path, "exists", return_value=True), \
                mock.patch.dict(baseline.os.environ, {"HF_TOKEN": "environment-test-marker"}):
            client.get_secret.return_value = "kaggle-test-marker"
            self.assertEqual(baseline.load_hf_token(), "kaggle-test-marker")
            client.get_secret.assert_called_with("HF_TOKEN")
            client.get_secret.side_effect = RuntimeError("private-test-marker")
            with self.assertRaises(RuntimeError) as error:
                baseline.load_hf_token()
            self.assertNotIn("private-test-marker", str(error.exception))


if __name__ == "__main__":
    unittest.main()
