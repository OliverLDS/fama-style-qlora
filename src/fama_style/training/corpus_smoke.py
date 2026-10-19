"""Three-row, five-update 4B QLoRA pipeline test against corpus baseline v4."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import threading
from pathlib import Path

from fama_style.evaluation import corpus_baseline as baseline

IMPLEMENTATION = "corpus_smoke_v1"
BASELINE_MANIFEST_SHA256 = "929ab47278132ce47176d81b0fbff38cf2d8ceb7c1e40c90b790d0fe525492e9"
MODEL_REVISION = "093f9f388b31de276ce2de164bdc2081324b9767"
MAX_LENGTH = 1024
STEPS = 5
ACCUMULATION = 2
TARGET_PATTERN = r".*language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)$"


def load_inputs(dataset, baseline_dir):
    rows, groups = baseline.load_corpus(dataset)
    directory = Path(baseline_dir)
    manifest_bytes = (directory / "baseline_manifest.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != BASELINE_MANIFEST_SHA256:
        raise ValueError("Attach the frozen successful corpus-v1 baseline version 4 output")
    manifest = json.loads(manifest_bytes)
    for name in ("baseline.json", "baseline_generations.jsonl", "token_audit.json"):
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != manifest["files"][name]:
            raise ValueError("Baseline artifact hash mismatch")
    report = json.loads((directory / "baseline.json").read_text())
    baseline.validate_baseline(report, groups["validation"])
    audit = json.loads((directory / "token_audit.json").read_text())
    records = audit["records"]
    if (report["model_revision"] != MODEL_REVISION or audit["model_revision"] != MODEL_REVISION
            or audit["dataset_sha256"] != baseline.DATASET_SHA256
            or audit["recommended_max_length"] != MAX_LENGTH
            or [r["id"] for r in records] != [r["id"] for r in rows]):
        raise ValueError("Baseline revision, audit or row provenance changed")
    train_ids, validation_ids = choose_examples(groups, records)
    return groups, audit, train_ids, validation_ids


def choose_examples(groups, records):
    by_id = {r["id"]: r for r in records}
    def ranked(split):
        return sorted(groups[split], key=lambda row: (-by_id[row["id"]]["full_tokens"], row["id"]))
    train_ids = [r["id"] for r in ranked("train")[:3]]
    validation_ids = [r["id"] for r in ranked("validation")[:2]]
    if (len(set(train_ids)) != 3 or len(set(validation_ids)) != 2
            or set(train_ids) & set(validation_ids)
            or any(by_id[key]["split"] != split or by_id[key]["full_tokens"] > MAX_LENGTH
                   for split, ids in (("train", train_ids), ("validation", validation_ids)) for key in ids)):
        raise ValueError("Smoke needs three unique training and two disjoint validation rows within 1024 tokens")
    return train_ids, validation_ids


def encode_pair(processor, row, expected):
    audit = baseline.prepare_pair(processor, row)
    if audit != expected or audit["full_tokens"] > MAX_LENGTH:
        raise ValueError("Runtime tokenization differs from the frozen baseline audit")
    messages = [baseline.user_message(row), {"role": "assistant", "content": [
        {"type": "text", "text": row["completion"]}]}]
    batch = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
        truncation=False, return_dict=True, return_tensors="pt")
    ids = baseline.input_ids(batch)
    # Revalidate the actual training batch, not only a separately rendered audit.
    prefix = audit["prompt_tokens"]
    if (len(ids) != audit["full_tokens"]
            or baseline.digest(json.dumps(ids[:prefix])) != audit["prefix_token_ids_sha256"]
            or " ".join(processor.tokenizer.decode(ids[prefix:], skip_special_tokens=True).split())
               != " ".join(row["completion"].split())):
        raise ValueError("Actual training batch changed its prefix or supervised target")
    labels = [-100] * prefix + ids[prefix:]
    prepared = {name: baseline.input_ids({"input_ids": value}) for name, value in batch.items()
                if name in ("input_ids", "attention_mask", "token_type_ids")}
    prepared["labels"] = labels
    audit = dict(audit, masked_prompt_tokens=prefix, supervised_tokens=len(ids)-prefix,
                 shifted_supervised_tokens=sum(value != -100 for value in labels[1:]))
    return prepared, audit


def validate_completion(status, evaluation):
    if (status.get("complete") is not True or status.get("optimizer_steps_completed") != STEPS
            or status.get("test_rows_used") != 0 or status.get("adapter_weights_changed") is not True
            or status.get("adapter_reload_verified") is not True
            or len(set(status.get("trained_ids", []))) != 3
            or len(set(status.get("validation_ids", []))) != 2
            or set(status["trained_ids"]) & set(status["validation_ids"])
            or len(status.get("updates", [])) != STEPS):
        raise ValueError("Incomplete smoke test; five updates and adapter reload are required")
    for index, update in enumerate(status["updates"], 1):
        expected_ids = [status["trained_ids"][((index-1)*ACCUMULATION+i) % 3] for i in range(ACCUMULATION)]
        if (update["step"] != index or len(update["microbatch_ids"]) != ACCUMULATION
                or update["microbatch_ids"] != expected_ids
                or any(key not in status["trained_ids"] for key in update["microbatch_ids"])
                or not math.isfinite(update["loss"]) or not math.isfinite(update["gradient_norm"])
                or update["gradient_norm"] <= 0):
            raise ValueError("Invalid update, loss or gradients")
    for key in ("original_validation_loss", "fine_tuned_validation_loss", "reloaded_validation_loss"):
        if not math.isfinite(evaluation[key]):
            raise ValueError("Non-finite held-out loss")
    if (evaluation["validation_ids"] != status["validation_ids"] or evaluation["test_rows_used"] != 0
            or evaluation.get("optimizer_steps") != STEPS or evaluation.get("train_rows") != 3
            or evaluation.get("adapter_reload_verified") is not True
            or not math.isclose(evaluation["fine_tuned_validation_loss"], evaluation["reloaded_validation_loss"],
                                rel_tol=1e-5, abs_tol=1e-5)):
        raise ValueError("Reloaded evaluation must match the trained adapter on the same held-out rows")


def run_training(groups, frozen_audit, train_ids, validation_ids, output_dir):
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Use a fresh output directory; saved smoke results must not be overwritten")
    output_dir.mkdir(parents=True, exist_ok=True)
    status = {"implementation": IMPLEMENTATION, "complete": False, "phase": "starting",
              "purpose": "pipeline and memory smoke test; not a learning-quality experiment",
              "model_id": baseline.MODEL_ID, "model_revision": MODEL_REVISION,
              "dataset_sha256": baseline.DATASET_SHA256, "baseline_manifest_sha256": BASELINE_MANIFEST_SHA256,
              "trained_ids": train_ids, "validation_ids": validation_ids, "test_rows_used": 0,
              "max_length": MAX_LENGTH, "optimizer_steps_planned": STEPS, "optimizer_steps_completed": 0,
              "gradient_accumulation_steps": ACCUMULATION, "microbatch_size": 1,
              "learning_rate": 5e-5, "lora_rank": 8, "lora_alpha": 16, "lora_dropout": 0.05,
              "target_modules_regex": TARGET_PATTERN, "seed": 42,
              "quantization": "NF4 double quantization, FP32 compute", "updates": [],
              "training_called": False, "training_gpus_used": 1, "second_t4_used": False}
    status_path = output_dir / "smoke_status.json"

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
        from transformers import AutoConfig, AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration
        from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict, prepare_model_for_kbit_training

        status["versions"] = {name: importlib.metadata.version(name) for name in
                              ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "huggingface-hub")}
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise baseline.PreflightError("one_visible_cuda_gpu_required")
        phase("checking_secret")
        token = baseline.load_hf_token()
        phase("loading_pinned_config")
        config = AutoConfig.from_pretrained(baseline.MODEL_ID, revision=MODEL_REVISION, token=token)
        phase("loading_processor")
        processor = heartbeat(lambda: AutoProcessor.from_pretrained(
            baseline.MODEL_ID, revision=MODEL_REVISION, token=token), "Processor load")
        phase("checking_supervised_tokens")
        rows = {row["id"]: row for split in ("train", "validation") for row in groups[split]}
        expected = {row["id"]: row for row in frozen_audit["records"]}
        prepared, audits = {}, []
        for key in train_ids + validation_ids:
            prepared[key], audit = encode_pair(processor, rows[key], expected[key])
            audits.append(audit)
        baseline.write_json(output_dir / "prepared_label_audit.json", {"records": audits, "truncation": False})
        phase("initializing_cuda")
        baseline.initialize_cuda(torch)
        phase("loading_4b_model")
        model = heartbeat(lambda: Gemma3ForConditionalGeneration.from_pretrained(
            baseline.MODEL_ID, revision=MODEL_REVISION, token=token, config=config,
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
        matched = [name for name, _ in model.named_modules() if re.fullmatch(TARGET_PATTERN, name)]
        if len(matched) != 2 * config.text_config.num_hidden_layers:
            raise baseline.PreflightError("language_qv_adapter_module_count_changed")
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=TARGET_PATTERN, bias="none", task_type="CAUSAL_LM", revision=MODEL_REVISION))
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
        for step in range(STEPS):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            used, losses = [], []
            for micro in range(ACCUMULATION):
                key = train_ids[(step*ACCUMULATION+micro) % len(train_ids)]
                def forward_backward():
                    loss = model(**batch(key), use_cache=False).loss
                    if not bool(torch.isfinite(loss).all()):
                        raise baseline.PreflightError("nonfinite_training_loss")
                    value = float(loss.detach().item())
                    (loss / ACCUMULATION).backward()
                    return value
                losses.append(heartbeat(forward_backward, f"Step {step+1}/{STEPS}, microbatch {micro+1}/2"))
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
            phase("training")
            print(f"Completed optimizer update {step+1}/{STEPS}; loss={sum(losses)/len(losses):.6f}", flush=True)
        optimizer.zero_grad(set_to_none=True)
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
            "model_revision": MODEL_REVISION, "dataset_sha256": baseline.DATASET_SHA256,
            "train_rows": 3, "optimizer_steps": STEPS, "validation_ids": validation_ids,
            "test_rows_used": 0, "loss_aggregation": "shifted supervised-token weighted mean across two rows",
            "original_validation_loss": original["loss"], "fine_tuned_validation_loss": adapted["loss"],
            "reloaded_validation_loss": reloaded["loss"], "original": original, "fine_tuned": adapted,
            "reloaded": reloaded, "adapter_reload_verified": True,
            "style_improvement_claimed": False, "generations": []}
        status["complete"] = True
        validate_completion(status, evaluation)
        baseline.write_json(output_dir / "evaluation.json", evaluation)
        baseline.write_json(output_dir / "training_log.json", status["updates"])
        phase("complete")
        print("Smoke complete: 3 training rows, 5 updates, adapter reloaded; evaluation.json saved", flush=True)
    except Exception as exc:
        failed_phase = status["phase"]
        status.update(complete=False, phase="failed", failed_phase=failed_phase, error_class=type(exc).__name__)
        status.update(baseline.safe_failure_details(exc))
        tb = exc.__traceback__
        while tb is not None:
            if tb.tb_frame.f_code.co_filename == __file__:
                status["stack_locations"].append({"component": "corpus_smoke", "line": tb.tb_lineno})
            tb = tb.tb_next
        if isinstance(exc, baseline.PreflightError):
            status["error_code"] = exc.code
        baseline.write_json(status_path, status)
        raise RuntimeError(f"Smoke stopped in {failed_phase}; inspect smoke_status.json. "
                           "Remote exception messages are suppressed.") from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--output-dir", default="/kaggle/working/corpus_v1_smoke")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    groups, audit, train_ids, validation_ids = load_inputs(args.dataset, args.baseline_dir)
    print("Validated frozen corpus and successful baseline v4; smoke IDs:", train_ids, validation_ids)
    if args.check:
        print("Dry-run OK. No Secret accessed, model downloaded or training called.")
        return
    run_training(groups, audit, train_ids, validation_ids, Path(args.output_dir))


if __name__ == "__main__":
    main()
