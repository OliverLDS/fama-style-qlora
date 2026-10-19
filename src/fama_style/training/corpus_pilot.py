"""One-epoch 247-pair 4B QLoRA pilot with frozen 21-row base comparison."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import threading
from pathlib import Path

from fama_style.evaluation import corpus_baseline as baseline
from fama_style.training import corpus_smoke as smoke

IMPLEMENTATION = "corpus_pilot_one_epoch_v1"
SMOKE_HASHES = {
    "smoke_status.json": "c79cddd061750a5430cdcf65bac37cd1bab64eed19168d83a327fbdac214596b",
    "evaluation.json": "a8c1daa71f68dbdd38bd13ba621cdf66a36ffcc5bf9546e7ddc3750f566e38b2",
}


def load_inputs(dataset, baseline_dir, smoke_dir):
    groups, audit, train_ids, validation_ids = smoke.load_inputs(dataset, baseline_dir)
    smoke_dir = Path(smoke_dir)
    for name, expected in SMOKE_HASHES.items():
        if hashlib.sha256((smoke_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Attach the successful corpus-v1 smoke version 1 output")
    status = json.loads((smoke_dir / "smoke_status.json").read_text())
    evaluation = json.loads((smoke_dir / "evaluation.json").read_text())
    smoke.validate_completion(status, evaluation)
    if (status["dataset_sha256"] != baseline.DATASET_SHA256
            or status["model_revision"] != smoke.MODEL_REVISION
            or status["trained_ids"] != train_ids or status["validation_ids"] != validation_ids):
        raise ValueError("Smoke provenance does not match frozen corpus/baseline")
    original = json.loads((Path(baseline_dir) / "baseline.json").read_text())
    return groups, audit, original


def epoch_batches(train_ids):
    if len(train_ids) != 247 or len(set(train_ids)) != 247:
        raise ValueError("One epoch requires 247 unique training IDs")
    order = list(train_ids)
    random.Random(42).shuffle(order)
    return [order[i:i+smoke.ACCUMULATION] for i in range(0, len(order), smoke.ACCUMULATION)]


def validate_loss_report(report, validation):
    if [r["id"] for r in report["rows"]] != [r["id"] for r in validation]:
        raise ValueError("Loss report must cover exactly all 21 validation rows")
    if any(not math.isfinite(r["loss"]) or r["loss"] < 0 or r["supervised_tokens"] <= 0 for r in report["rows"]):
        raise ValueError("Invalid validation loss or supervised-token count")
    total = sum(r["supervised_tokens"] for r in report["rows"])
    expected = sum(r["loss"]*r["supervised_tokens"] for r in report["rows"])/total
    if not math.isfinite(report["loss"]) or not math.isclose(report["loss"], expected, rel_tol=1e-10):
        raise ValueError("Invalid token-weighted aggregate validation loss")


def validate_comparison(comparison, validation, original, require_complete=True):
    if (comparison.get("complete") is not True and require_complete
            or comparison["model_id"] != baseline.MODEL_ID
            or comparison["model_revision"] != smoke.MODEL_REVISION
            or comparison["dataset_sha256"] != baseline.DATASET_SHA256
            or comparison["test_rows_used"] != 0 or comparison["epochs"] != 1
            or comparison["generation"] != original["generation"]
            or len(comparison["results"]) != 21):
        raise ValueError("Comparison must cover the frozen 21-row validation set")
    for result, row, old in zip(comparison["results"], validation, original["results"]):
        baseline.validate_result(result)
        if (result["id"] != row["id"] or result["source"] != row["plain_english"]
                or result["reference"] != row["completion"] or result["baseline_output"] != old["output"]
                or result["prompt_sha256"] != baseline.digest(row["prompt"])):
            raise ValueError("Comparison provenance or prompt changed")


def validate_completion(status, evaluation, comparison, groups, original):
    expected = epoch_batches([row["id"] for row in groups["train"]])
    if (status.get("complete") is not True or status.get("epochs_completed") != 1
            or status.get("epochs_planned") != 1 or status.get("training_called") is not True
            or status.get("optimizer_steps_completed") != len(expected)
            or status.get("training_examples_seen") != 247
            or status.get("adapter_weights_changed") is not True
            or status.get("adapter_reload_verified") is not True or status.get("test_rows_used") != 0
            or status.get("train_order") != [key for batch in expected for key in batch]
            or len(status.get("updates", [])) != len(expected)):
        raise ValueError("Pilot must complete exactly one epoch with verified adapter reload")
    for step, (update, ids) in enumerate(zip(status["updates"], expected), 1):
        if (update["step"] != step or update["microbatch_ids"] != ids
                or not math.isfinite(update["loss"]) or not math.isfinite(update["gradient_norm"])
                or update["gradient_norm"] <= 0):
            raise ValueError("Invalid pilot update, training coverage or gradients")
    for key in ("original", "fine_tuned", "reloaded"):
        validate_loss_report(evaluation[key], groups["validation"])
    for before, after in zip(evaluation["fine_tuned"]["rows"], evaluation["reloaded"]["rows"]):
        if before["supervised_tokens"] != after["supervised_tokens"] or not math.isclose(
                before["loss"], after["loss"], rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError("Reloaded per-row losses do not match")
    for label, key in (("original_validation_loss", "original"),
                       ("fine_tuned_validation_loss", "fine_tuned"), ("reloaded_validation_loss", "reloaded")):
        if evaluation[label] != evaluation[key]["loss"]:
            raise ValueError("Validation summary differs from per-row loss report")
    if (evaluation.get("complete") is not True or evaluation["test_rows_used"] != 0 or evaluation["epochs"] != 1
            or evaluation["optimizer_steps"] != 124
            or evaluation["train_rows"] != 247 or evaluation["adapter_reload_verified"] is not True):
        raise ValueError("Evaluation scope or reload status changed")
    validate_comparison(comparison, groups["validation"], original)


def save_comparison(directory, comparison):
    baseline.write_json(directory / "comparison.json", comparison)
    (directory / "validation_generations.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False)+"\n" for row in comparison["results"]), encoding="utf-8")


def save_review(directory, comparison):
    """Masked-order worksheet; mapping stays in a separate file, not independently blind."""
    rng = random.Random(20261019)
    lines = ["# One-epoch validation comparison", "",
             "Review claim preservation and style separately. Build a claim inventory from the source.",
             "A/B order varies by paragraph. Keep comparison_key.json closed until scores are saved.",
             "This masks labels only; it does not establish independent blinded evaluation.", ""]
    mapping, forms = {}, []
    for result in comparison["results"]:
        labels = ["base", "epoch_1"]
        rng.shuffle(labels)
        mapping[result["id"]] = dict(zip(("A", "B"), labels))
        outputs = {"base": result["baseline_output"], "epoch_1": result["output"]}
        lines += ["## "+result["id"], "", "### Source", "", result["source"], "",
                  "### Original reference", "", result["reference"], ""]
        for slot, label in zip(("A", "B"), labels):
            lines += ["### Candidate "+slot, "", outputs[label], "",
                      "- Claim inventory / additions / omissions / changed conditions, actors, numbers or causality:",
                      "- Claim preservation (0–4):", "- Economic precision (0–2):",
                      "- Analytical structure (0–2):", "- Concision resembling the reference (0–2):",
                      "- Style evidence; copying and verbosity observations:", ""]
            forms.append({"id": result["id"], "candidate": slot, "claim_inventory": [],
                "added_claims": [], "lost_claims": [], "changed_claims": [], "claim_preservation_score": None,
                "style": {"economic_precision": None, "analytical_structure": None, "concision": None},
                "review_status": "pending_review"})
    (directory / "claim_style_review.md").write_text("\n".join(lines), encoding="utf-8")
    baseline.write_json(directory / "comparison_key.json", mapping)
    path = directory / "review_scores.jsonl"
    if not path.exists():
        path.write_text("".join(json.dumps(form)+"\n" for form in forms), encoding="utf-8")


def run_training(groups, frozen_audit, original_baseline, output_dir):
    train_ids = [row["id"] for row in groups["train"]]
    validation_ids = [row["id"] for row in groups["validation"]]
    schedule = epoch_batches(train_ids)
    train_order = [key for ids in schedule for key in ids]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Use a fresh output directory; saved pilot results must not be overwritten")
    output_dir.mkdir(parents=True, exist_ok=True)
    status = {"implementation": IMPLEMENTATION, "complete": False, "phase": "starting",
              "purpose": "bounded one-epoch pilot; no automatic continuation",
              "model_id": baseline.MODEL_ID, "model_revision": smoke.MODEL_REVISION,
              "dataset_sha256": baseline.DATASET_SHA256, "baseline_manifest_sha256": smoke.BASELINE_MANIFEST_SHA256,
              "trained_ids": train_ids, "validation_ids": validation_ids, "test_rows_used": 0,
              "max_length": smoke.MAX_LENGTH, "optimizer_steps_planned": len(schedule), "optimizer_steps_completed": 0,
              "epochs_planned": 1, "epochs_completed": 0, "training_examples_seen": 0, "train_order": train_order,
              "gradient_accumulation_steps": smoke.ACCUMULATION, "microbatch_size": 1,
              "learning_rate": 5e-5, "lora_rank": 8, "lora_alpha": 16, "lora_dropout": 0.05,
              "target_modules_regex": smoke.TARGET_PATTERN, "seed": 42,
              "quantization": "NF4 double quantization, FP32 compute", "updates": [],
              "training_called": False, "training_gpus_used": 1, "second_t4_used": False}
    status_path = output_dir / "pilot_status.json"

    def phase(name, **values):
        status.update(phase=name, **values)
        baseline.write_json(status_path, status)
        print("Phase: " + name, flush=True)

    def heartbeat(operation, label):
        stop = threading.Event()
        def announce():
            while not stop.wait(30):
                print(label + ": still working", flush=True)
        thread = threading.Thread(target=announce, daemon=True)
        thread.start()
        try:
            return operation()
        finally:
            stop.set()
            thread.join(timeout=1)

    try:
        phase("loading_runtime")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import torch
        from transformers import (AutoConfig, AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration,
                                  LogitsProcessor, LogitsProcessorList)
        from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict, prepare_model_for_kbit_training

        status["versions"] = {name: importlib.metadata.version(name) for name in
                              ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "huggingface-hub")}
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise baseline.PreflightError("one_visible_cuda_gpu_required")
        phase("checking_secret")
        token = baseline.load_hf_token()
        phase("loading_pinned_config")
        config = AutoConfig.from_pretrained(baseline.MODEL_ID, revision=smoke.MODEL_REVISION, token=token)
        phase("loading_processor")
        processor = heartbeat(lambda: AutoProcessor.from_pretrained(
            baseline.MODEL_ID, revision=smoke.MODEL_REVISION, token=token), "Processor load")
        phase("checking_supervised_tokens")
        rows = {row["id"]: row for split in ("train", "validation") for row in groups[split]}
        expected = {row["id"]: row for row in frozen_audit["records"]}
        prepared, audits = {}, []
        for key in train_ids + validation_ids:
            prepared[key], audit = smoke.encode_pair(processor, rows[key], expected[key])
            audits.append(audit)
        baseline.write_json(output_dir / "prepared_label_audit.json", {"records": audits, "truncation": False})
        phase("initializing_cuda")
        baseline.initialize_cuda(torch)
        phase("loading_4b_model")
        model = heartbeat(lambda: Gemma3ForConditionalGeneration.from_pretrained(
            baseline.MODEL_ID, revision=smoke.MODEL_REVISION, token=token, config=config,
            dtype=torch.float32, device_map={"": 0}, attn_implementation="sdpa",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float32)), "4B model load")
        phase("checking_model_placement")
        status["base_placement"] = baseline.validate_model_placement(model)
        phase("preparing_qlora")
        model.config.use_cache = False
        model.config.text_config.use_cache = False
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
        matched = [name for name, _ in model.named_modules() if re.fullmatch(smoke.TARGET_PATTERN, name)]
        if len(matched) != 2 * config.text_config.num_hidden_layers:
            raise baseline.PreflightError("language_qv_adapter_module_count_changed")
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=smoke.TARGET_PATTERN, bias="none", task_type="CAUSAL_LM", revision=smoke.MODEL_REVISION))
        parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
        if not parameters or any("lora_" not in name or "language_model" not in name for name in parameters):
            raise baseline.PreflightError("only_language_lora_parameters_may_train")
        status.update(trainable_parameters=sum(p.numel() for p in parameters.values()),
                      adapted_language_modules=len(matched), adapted_placement=baseline.validate_model_placement(model))

        def adapter_hashes():
            return {name: hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()
                    for name, value in get_peft_model_state_dict(model, save_embedding_layers=False).items()}

        initial_hashes = adapter_hashes()

        def batch(key):
            return {name: torch.tensor([value], dtype=torch.long, device="cuda:0")
                    for name, value in prepared[key].items()}

        target_counts = {a["id"]: a["shifted_supervised_tokens"] for a in audits}

        def evaluate():
            model.eval()
            values = []
            with torch.inference_mode():
                for key in validation_ids:
                    loss = float(model(**batch(key), use_cache=False).loss.item())
                    if not math.isfinite(loss):
                        raise baseline.PreflightError("nonfinite_validation_loss")
                    values.append({"id": key, "loss": loss, "supervised_tokens": target_counts[key]})
            return {"loss": sum(v["loss"]*v["supervised_tokens"] for v in values)
                            / sum(v["supervised_tokens"] for v in values), "rows": values}

        phase("base_validation_loss")
        with model.disable_adapter():
            original = heartbeat(evaluate, "Base validation loss")
        baseline.write_json(output_dir / "base_validation_loss.json", original)
        optimizer = torch.optim.AdamW(parameters.values(), lr=5e-5, weight_decay=0.0)
        phase("training", training_called=True)
        for step, microbatch_ids in enumerate(schedule):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            used, losses = [], []
            for micro, key in enumerate(microbatch_ids):
                def forward_backward():
                    loss = model(**batch(key), use_cache=False).loss
                    if not bool(torch.isfinite(loss).all()):
                        raise baseline.PreflightError("nonfinite_training_loss")
                    value = float(loss.detach().item())
                    (loss / len(microbatch_ids)).backward()
                    return value
                losses.append(heartbeat(forward_backward, f"Update {step+1}/{len(schedule)}, microbatch {micro+1}/{len(microbatch_ids)}"))
                used.append(key)
            gradients = [p.grad for p in parameters.values() if p.grad is not None]
            if not gradients or any(not bool(torch.isfinite(g).all()) for g in gradients):
                raise baseline.PreflightError("missing_or_nonfinite_adapter_gradients")
            norm = float(torch.nn.utils.clip_grad_norm_(list(parameters.values()), 1.0, error_if_nonfinite=True))
            if not math.isfinite(norm) or norm <= 0:
                raise baseline.PreflightError("adapter_gradient_norm_not_positive")
            optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in parameters.values()):
                raise baseline.PreflightError("nonfinite_adapter_weights")
            status["updates"].append({"step": step+1, "microbatch_ids": used,
                                      "loss": sum(losses)/len(losses), "gradient_norm": norm})
            status["optimizer_steps_completed"] = step+1
            status["training_examples_seen"] += len(used)
            status["memory"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                                "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}
            phase("training")
            print(f"Completed optimizer update {step+1}/{len(schedule)}; loss={sum(losses)/len(losses):.6f}", flush=True)
        optimizer.zero_grad(set_to_none=True)
        status["epochs_completed"] = 1
        phase("adapted_validation_loss")
        adapted = heartbeat(evaluate, "Adapted validation loss")
        trained_hashes = adapter_hashes()
        if trained_hashes == initial_hashes:
            raise baseline.PreflightError("adapter_weights_unchanged")
        status["adapter_weights_changed"] = True
        phase("saving_adapter")
        adapter_dir = output_dir / "adapter"
        # Vocabulary/embeddings are unchanged and frozen. Avoid the optional
        # automatic Hub config lookup used to detect resized embeddings.
        model.save_pretrained(adapter_dir, safe_serialization=True, save_embedding_layers=False)
        processor.save_pretrained(adapter_dir)
        baseline.write_json(output_dir / "adapter_tensor_hashes.json", trained_hashes)
        # Remove trained adapter modules, then load the files onto the same frozen
        # quantized base. Do not merge or allocate a second 4B base on the T4.
        phase("reloading_adapter")
        del optimizer, parameters, gradients
        bare_model = model.unload()
        del model
        if any("lora_" in name for name, _ in bare_model.named_parameters()):
            raise baseline.PreflightError("adapter_unload_incomplete")
        model = PeftModel.from_pretrained(bare_model, adapter_dir, is_trainable=False)
        if adapter_hashes() != trained_hashes:
            raise baseline.PreflightError("reloaded_adapter_tensor_hash_mismatch")
        status["reloaded_placement"] = baseline.validate_model_placement(model)
        phase("reloaded_validation_loss")
        reloaded = heartbeat(evaluate, "Reloaded validation loss")
        for before, after in zip(adapted["rows"], reloaded["rows"]):
            if before["id"] != after["id"] or not math.isclose(before["loss"], after["loss"], rel_tol=1e-5, abs_tol=1e-5):
                raise baseline.PreflightError("reloaded_validation_loss_mismatch")
        status.update(adapter_reload_verified=True, memory={"gpu": torch.cuda.get_device_name(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)})
        evaluation = {"purpose": status["purpose"], "model_id": baseline.MODEL_ID,
            "model_revision": smoke.MODEL_REVISION, "dataset_sha256": baseline.DATASET_SHA256,
            "train_rows": 247, "optimizer_steps": len(schedule), "epochs": 1, "validation_ids": validation_ids,
            "test_rows_used": 0, "loss_aggregation": "shifted supervised-token weighted mean across all 21 validation rows",
            "original_validation_loss": original["loss"], "fine_tuned_validation_loss": adapted["loss"],
            "reloaded_validation_loss": reloaded["loss"], "original": original, "fine_tuned": adapted,
            "reloaded": reloaded, "adapter_reload_verified": True,
            "style_improvement_claimed": False, "generations": []}
        # Preserve completed loss evaluation and adapter even if generation later fails.
        evaluation["complete"] = False
        baseline.write_json(output_dir / "evaluation.json", evaluation)
        baseline.write_json(output_dir / "training_log.json", status["updates"])
        phase("validation_generation", saved_generations=0)
        model.eval()
        comparison = {"complete": False, "model_id": baseline.MODEL_ID,
            "model_revision": smoke.MODEL_REVISION, "dataset_sha256": baseline.DATASET_SHA256,
            "epochs": 1, "test_rows_used": 0, "generation": original_baseline["generation"], "results": []}
        save_comparison(output_dir, comparison)

        class FiniteScores(LogitsProcessor):
            def __call__(self, input_ids, scores):
                if bool(torch.isnan(scores).any() | torch.isposinf(scores).any()
                        | ~torch.isfinite(scores).any(dim=-1).all()):
                    raise baseline.PreflightError("invalid_generation_scores")
                return scores

        budget = original_baseline["generation"]["max_new_tokens"]
        for index, (row, old) in enumerate(zip(groups["validation"], original_baseline["results"]), 1):
            phase("validation_generation", saved_generations=index-1, current_id=row["id"])
            inputs = processor.apply_chat_template([baseline.user_message(row)], tokenize=True,
                add_generation_prompt=True, truncation=False, return_dict=True, return_tensors="pt")
            prefix = baseline.input_ids(inputs)
            if (baseline.digest(json.dumps(prefix)) != expected[row["id"]]["prefix_token_ids_sha256"]
                    or len(prefix)+budget > baseline.CONTEXT_BUDGET):
                raise baseline.PreflightError("generation_prefix_or_context_budget_changed")
            inputs = {name: value.to("cuda:0") for name, value in inputs.items()}
            with torch.inference_mode():
                tokens = heartbeat(lambda: model.generate(**inputs, do_sample=False, max_new_tokens=budget,
                    use_cache=True, pad_token_id=processor.tokenizer.pad_token_id, remove_invalid_values=False,
                    logits_processor=LogitsProcessorList([FiniteScores()])), f"Validation generation {index}/21")
            generated = tokens[0, len(prefix):]
            text = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
            eos = model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            result = {"id": row["id"], "source": row["plain_english"], "reference": row["completion"],
                "prompt_sha256": baseline.digest(row["prompt"]), "baseline_output": old["output"], "output": text,
                "output_tokens": int(generated.numel()), "input_tokens": len(prefix),
                "hit_output_limit": generated.numel() >= budget and
                    (not generated.numel() or int(generated[-1]) not in eos),
                "copied_source": " ".join(text.split()) == " ".join(row["plain_english"].split()),
                "exact_reference_match": " ".join(text.split()) == " ".join(row["completion"].split()),
                "source_words": len(row["plain_english"].split()), "output_words": len(text.split()),
                "reference_words": len(row["completion"].split()), "claim_review_status": "pending_review"}
            comparison["results"].append(result)
            save_comparison(output_dir, comparison)
            baseline.validate_result(result)
            print(f"Saved validation generation {index}/21", flush=True)
        comparison["complete"] = True
        evaluation["complete"] = True
        status.update(complete=True, saved_generations=21)
        validate_completion(status, evaluation, comparison, groups, original_baseline)
        baseline.write_json(output_dir / "evaluation.json", evaluation)
        save_comparison(output_dir, comparison)
        save_review(output_dir, comparison)
        status["memory"].update(peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                                peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        artifacts = ["evaluation.json", "comparison.json", "validation_generations.jsonl",
                     "prepared_label_audit.json", "training_log.json", "adapter_tensor_hashes.json",
                     "adapter/adapter_config.json", "adapter/adapter_model.safetensors"]
        baseline.write_json(output_dir / "pilot_manifest.json", {"implementation": IMPLEMENTATION,
            "model_revision": smoke.MODEL_REVISION, "dataset_sha256": baseline.DATASET_SHA256,
            "baseline_manifest_sha256": smoke.BASELINE_MANIFEST_SHA256, "smoke_gate_hashes": SMOKE_HASHES,
            "files": {name: hashlib.sha256((output_dir/name).read_bytes()).hexdigest() for name in artifacts}})
        phase("complete")
        print("Pilot complete: one epoch, 247 examples, 124 updates, 21 validation generations", flush=True)
    except Exception as exc:
        failed_phase = status["phase"]
        status.update(complete=False, phase="failed", failed_phase=failed_phase, error_class=type(exc).__name__)
        status.update(baseline.safe_failure_details(exc))
        tb = exc.__traceback__
        while tb is not None:
            if tb.tb_frame.f_code.co_filename == __file__:
                status["stack_locations"].append({"component": "corpus_pilot", "line": tb.tb_lineno})
            tb = tb.tb_next
        if isinstance(exc, baseline.PreflightError):
            status["error_code"] = exc.code
        baseline.write_json(status_path, status)
        raise RuntimeError(f"Pilot stopped in {failed_phase}; inspect pilot_status.json. "
                           "Remote exception messages are suppressed.") from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--smoke-dir", required=True)
    parser.add_argument("--output-dir", default="/kaggle/working/corpus_v1_pilot")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    groups, audit, original = load_inputs(args.dataset, args.baseline_dir, args.smoke_dir)
    schedule = epoch_batches([row["id"] for row in groups["train"]])
    print(f"Validated baseline v4 and smoke v1; one epoch, 247 train pairs, {len(schedule)} updates; "
          "21 validation pairs; zero test use")
    if args.check:
        print("Dry-run OK. No credentials accessed, model downloaded or training called.")
        return
    run_training(groups, audit, original, Path(args.output_dir))


if __name__ == "__main__":
    main()
