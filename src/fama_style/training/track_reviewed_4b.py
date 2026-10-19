"""Controlled four-epoch 4B pilot with complete validation generations and claim review."""

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

from fama_style.evaluation.compare_few_shot import load_hf_token, normalized
from fama_style.training.smoke_4b_reviewed import MODEL_ID, FORMAT, MAX_LENGTH, TARGET_PATTERN, load_inputs

EPOCHS = 4
TOTAL_STEPS = 96  # 48 rows / accumulation 2, batch 1, one GPU, four epochs.


def validate_smoke(path, dataset_path):
    status = json.loads(Path(path).read_text(encoding="utf-8"))
    if (status.get("complete") is not True or status.get("phase") != "complete"
            or status.get("model_id") != MODEL_ID or status.get("optimizer_steps_completed") != 5
            or status.get("adapter_weights_changed") is not True
            or status.get("dataset_sha256") != hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()
            or len(status.get("gradient_checks", [])) != 5
            or any(g.get("finite") is not True or not math.isfinite(g.get("norm", 0))
                   or g.get("norm", 0) <= 0 for g in status.get("gradient_checks", []))):
        raise ValueError("Attach the successful reviewed 4B smoke output for this exact Dataset")
    return status


def write_claim_review(report, path):
    lines = ["# Reviewed 4B pilot: validation claim review", "",
             "Use each source below to build a claim inventory before judging its rewrites.",
             "Check each source's actors, qualifications, causal direction and mechanisms.",
             "Boxes are deliberately blank; copying flags do not measure claim preservation.", ""]
    for row in report["validation_examples"]:
        source_id = row["source_ids"][0]
        lines += [f"## {source_id}", "", "### Source", "", row["source"], "",
                  "### Reference", "", row["reference"], ""]
        for snap in report["snapshots"]:
            result = next((r for r in snap["generations"] if r["source_ids"] == row["source_ids"]), None)
            if result is None:
                continue
            lines += [f"### Epoch {snap['epoch']} — step {snap['optimizer_step']}", "", result["output"], "",
                      "- [ ] All source claims retained", "- [ ] No unsupported claims added",
                      "- [ ] Actors and their roles retained", "- [ ] Qualifications retained",
                      "- [ ] Causal direction and mechanisms retained", "- [ ] One paragraph; no explanation or repetition",
                      "- Added/lost/changed claims (quote source and output):", "- Style and clarity notes:", ""]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_snapshots(report):
    expected = [r["source_ids"] for r in report["validation_examples"]]
    snapshots = report["snapshots"]
    if len(expected) != 12 or [s["epoch"] for s in snapshots] != list(range(EPOCHS+1)):
        raise ValueError("Expected baseline and four complete validation snapshots")
    for snap in snapshots:
        if (snap.get("complete") is not True or snap["optimizer_step"] != snap["epoch"]*24
                or not math.isfinite(snap["validation_loss"])
                or [r["source_ids"] for r in snap["losses"]] != expected
                or any(not math.isfinite(r["loss"]) for r in snap["losses"])
                or [r["source_ids"] for r in snap["generations"]] != expected
                or any(not r["output"].strip() for r in snap["generations"])):
            raise ValueError("Incomplete validation snapshot, invalid loss, or mismatched held-out IDs")


def run_training(groups, baseline, dataset_path, output_dir):
    # Single-device training avoids unsupported inference-style multi-GPU dispatch.
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"
    import torch
    import transformers
    import peft
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration,
                              LogitsProcessor, LogitsProcessorList, Trainer, TrainerCallback, TrainingArguments)

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("The training process needs one visible CUDA GPU; select Kaggle T4 x2")
    if not load_hf_token():
        raise RuntimeError("Enable HF_TOKEN in Kaggle Secrets")
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "pilot_status.json"
    status = {"model_id": MODEL_ID, "complete": False, "purpose": "four-epoch reviewed-pair pilot",
              "trained_ids": [r["source_ids"][0] for r in groups["train"]],
              "validation_ids": [r["source_ids"][0] for r in groups["validation"]],
              "epochs_planned": EPOCHS, "optimizer_steps_planned": TOTAL_STEPS,
              "eligible_training_rows": 48, "test_rows_used": 0,
              "training_gpus_used": 1, "second_t4_used": False,
              "learning_rate": 5e-5, "lora_rank": 8, "lora_alpha": 16, "lora_dropout": 0.05,
              "target_modules_regex": TARGET_PATTERN, "max_length": MAX_LENGTH,
              "gradient_accumulation_steps": 2, "seed": 42,
              "dtype": "NF4 weights, FP32 compute and unquantized layers",
              "dataset_sha256": hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest(),
              "transformers_version": transformers.__version__, "peft_version": peft.__version__,
              "torch_version": torch.__version__, "phase": "loading", "gradient_checks": []}

    def save_status():
        free, total = torch.cuda.mem_get_info(0)
        status["memory"] = {"gpu_name": torch.cuda.get_device_name(0), "free_bytes": free,
                            "total_bytes": total, "allocated_bytes": torch.cuda.memory_allocated(0),
                            "reserved_bytes": torch.cuda.memory_reserved(0),
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0)}
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    save_status()
    torch.manual_seed(42)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tokenizer = processor.tokenizer
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.float32, device_map={"": 0}, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float32))
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                           gradient_checkpointing_kwargs={"use_reentrant": False})
    matched = [name for name, _ in model.named_modules() if re.fullmatch(TARGET_PATTERN, name)]
    if len(matched) != 2 * model.config.text_config.num_hidden_layers:
        raise RuntimeError("Expected q/v adapters only in every language layer; module layout changed")
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=TARGET_PATTERN, bias="none", task_type="CAUSAL_LM"))
    parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
    if not parameters or any("lora_" not in name or "language_model" not in name for name in parameters):
        raise RuntimeError("Only language LoRA parameters may train; base and vision weights must be frozen")
    before = {name: value.detach().cpu().clone() for name, value in parameters.items()}
    status.update(phase="model_loaded", trainable_parameters=sum(p.numel() for p in parameters.values()),
                  adapted_language_modules=len(matched), model_revision=getattr(model.config, "_commit_hash", None))
    save_status()

    def encode(row):
        user = {"role": "user", "content": [{"type": "text", "text": row["prompt"]}]}
        assistant = {"role": "assistant", "content": [{"type": "text", "text": row["completion"]}]}
        prefix = processor.apply_chat_template([user], tokenize=True, add_generation_prompt=True,
                                              return_dict=True, return_tensors="pt")["input_ids"][0].tolist()
        full = processor.apply_chat_template([user, assistant], tokenize=True,
                                            return_dict=True, return_tensors="pt")
        ids = full["input_ids"][0].tolist()
        if ids[:len(prefix)] != prefix or len(ids) <= len(prefix) or len(ids) > MAX_LENGTH:
            raise ValueError(f"Bad token boundary or overlength row {row['source_ids']}; no truncation permitted")
        labels = [-100] * len(prefix) + ids[len(prefix):]
        decoded = tokenizer.decode(ids[len(prefix):], skip_special_tokens=True)
        if normalized(decoded) != normalized(row["completion"]):
            raise ValueError(f"Supervised labels do not exactly match target for {row['source_ids']}")
        prepared = {key: value[0].tolist() for key, value in full.items()
                    if key in ("input_ids", "attention_mask", "token_type_ids")}
        prepared["labels"] = labels
        audit = {"source_ids": row["source_ids"], "split": row["split"], "full_tokens": len(ids),
                 "masked_prompt_tokens": len(prefix), "supervised_tokens": len(ids)-len(prefix),
                 "labels_match_target": True}
        return prepared, audit

    encoded, audits = {}, []
    for row in groups["train"] + groups["validation"]:
        prepared, audit = encode(row)
        encoded[tuple(row["source_ids"])] = prepared
        audits.append(audit)
    (output_dir / "prepared_label_audit.json").write_text(json.dumps(audits, indent=2) + "\n", encoding="utf-8")
    status.update(phase="labels_checked", max_full_tokens=max(a["full_tokens"] for a in audits))
    save_status()
    processor.save_pretrained(str(output_dir / "processor"))
    report = {"model_id": MODEL_ID, "complete": False, "prompt_format": FORMAT,
              "train_rows": 48, "validation_rows": 12, "test_rows_used": 0,
              "initialization": "fresh adapter; no smoke adapter loaded",
              "dataset_sha256": status["dataset_sha256"], "epochs_planned": EPOCHS,
              "optimizer_steps_planned": TOTAL_STEPS,
              "generation": {"do_sample": False, "max_new_tokens": 512},
              "validation_examples": [{"source_ids": r["source_ids"], "source": r["plain_english"],
                  "reference": r["completion"], "prompt_sha256": hashlib.sha256(r["prompt"].encode()).hexdigest()}
                  for r in groups["validation"]],
              "saved_untrained_baseline": baseline, "snapshots": []}

    def persist():
        (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
        write_claim_review(report, output_dir / "claim_review.md")

    def collate(features):
        if len(features) != 1:
            raise ValueError("This pilot uses batch size one")
        return {key: torch.tensor([value], dtype=torch.long) for key, value in features[0].items()}

    class FiniteScores(LogitsProcessor):
        def __call__(self, input_ids, scores):
            if bool(torch.isnan(scores).any() | torch.isposinf(scores).any()
                    | ~torch.isfinite(scores).any(dim=-1).all()):
                raise RuntimeError("Non-finite generation scores")
            return scores

    def snapshot(epoch, step):
        # Called at baseline and from on_epoch_end; restore mode before Trainer continues.
        was_training = model.training
        model.eval()
        snap = {"epoch": epoch, "optimizer_step": step, "complete": False,
                "losses": [], "generations": []}
        report["snapshots"].append(snap)
        status.update(phase="validation", current_epoch=epoch, validation_generations_completed=0)
        save_status()
        persist()
        try:
            for row in groups["validation"]:
                batch = {name: value.to("cuda:0") for name, value in
                         collate([encoded[tuple(row["source_ids"])]]).items()}
                with torch.inference_mode():
                    loss = float(model(**batch).loss.item())
                if not math.isfinite(loss):
                    raise RuntimeError("Non-finite validation loss")
                snap["losses"].append({"source_ids": row["source_ids"], "loss": loss})
            # Same mean-per-example aggregation as the smoke Trainer evaluation.
            snap["validation_loss"] = sum(r["loss"] for r in snap["losses"])/len(snap["losses"])
            for row in groups["validation"]:
                inputs = processor.apply_chat_template([{"role": "user", "content": [
                    {"type": "text", "text": row["prompt"]}]}], tokenize=True,
                    add_generation_prompt=True, return_dict=True, return_tensors="pt")
                inputs = {name: value.to("cuda:0") for name, value in inputs.items()}
                with torch.inference_mode():
                    tokens = model.generate(**inputs, max_new_tokens=512, do_sample=False, use_cache=True,
                        pad_token_id=tokenizer.pad_token_id, remove_invalid_values=False,
                        logits_processor=LogitsProcessorList([FiniteScores()]))
                generated = tokens[0, inputs["input_ids"].shape[-1]:]
                output = tokenizer.decode(generated, skip_special_tokens=True).strip()
                if not output:
                    raise RuntimeError("Empty validation generation")
                old = next(r for r in baseline["results"] if r["source_ids"] == row["source_ids"])
                snap["generations"].append({"source_ids": row["source_ids"], "output": output,
                    "output_tokens": int(generated.numel()), "at_output_budget": generated.numel() == 512,
                    "copied_source": normalized(output) == normalized(row["plain_english"]),
                    "matches_reference": normalized(output) == normalized(row["completion"]),
                    "matches_saved_baseline": normalized(output) == normalized(old["output"]),
                    "claim_review": {"claims_added": None, "claims_lost": None,
                        "actors_changed": None, "qualifications_changed": None,
                        "causal_direction_changed": None, "notes": None}})
                status["validation_generations_completed"] = len(snap["generations"])
                persist()
                save_status()
                print(f"Epoch {epoch}: validation generation {len(snap['generations'])}/12 ({row['source_ids'][0]})", flush=True)
            snap["complete"] = True
            persist()
            print(f"Epoch {epoch} complete: validation loss {snap['validation_loss']:.6f}", flush=True)
        finally:
            model.train(was_training)
        status["phase"] = "training" if epoch < EPOCHS else "finalizing"
        save_status()

    class PilotChecks(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            grads = [p.grad.detach() for p in parameters.values() if p.grad is not None]
            if not grads or any(not bool(torch.isfinite(g).all()) for g in grads):
                raise RuntimeError("Missing or non-finite adapter gradients")
            norm = math.sqrt(sum(float(g.float().square().sum().item()) for g in grads))
            if not math.isfinite(norm) or norm <= 0:
                raise RuntimeError("Adapter gradient norm must be finite and positive")
            status["gradient_checks"].append({"step": state.global_step+1, "finite": True, "norm": norm})
            save_status()

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs and not math.isfinite(logs["loss"]):
                raise RuntimeError("Non-finite training loss")

        def on_epoch_end(self, args, state, control, **kwargs):
            epoch = round(state.epoch)
            if not math.isclose(state.epoch, epoch) or state.global_step != epoch*24:
                raise RuntimeError("Unexpected epoch/update boundary")
            status["optimizer_steps_completed"] = state.global_step
            # Preserve the epoch adapter even if a later generation fails.
            model.save_pretrained(str(output_dir / f"epoch_{epoch}" / "adapter"))
            snapshot(epoch, state.global_step)
            (output_dir / "training_log.json").write_text(json.dumps(state.log_history, indent=2)+"\n")
            return control

    class FiniteLossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            result = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
            loss = result[0] if return_outputs else result
            if not bool(torch.isfinite(loss).all()):
                raise RuntimeError("Non-finite training forward loss")
            return result

    args = TrainingArguments(output_dir=str(output_dir / "trainer"), num_train_epochs=EPOCHS,
        per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=2,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=5e-5, lr_scheduler_type="linear", warmup_steps=0, optim="adamw_torch",
        fp16=False, bf16=False, logging_steps=1, save_strategy="no", report_to="none", seed=42,
        remove_unused_columns=False, prediction_loss_only=True, label_names=["labels"])
    trainer = FiniteLossTrainer(model=model, args=args, processing_class=processor, data_collator=collate,
        train_dataset=Dataset.from_list([encoded[tuple(r["source_ids"])] for r in groups["train"]]),
        callbacks=[PilotChecks()])
    trainer.model_accepts_loss_kwargs = False
    # Recompute an untrained baseline in this environment; retain old outputs for comparison.
    with model.disable_adapter():
        snapshot(0, 0)
    trainer.train()
    changed = any(not torch.equal(before[name], value.detach().cpu()) for name, value in parameters.items())
    validate_snapshots(report)
    if (trainer.state.global_step != TOTAL_STEPS or len(status["gradient_checks"]) != TOTAL_STEPS
            or not changed):
        raise RuntimeError("Pilot completion checks failed")
    report.update(complete=True, adapter_weights_changed=True, optimizer_steps_completed=TOTAL_STEPS,
                  original_validation_loss=report["snapshots"][0]["validation_loss"],
                  fine_tuned_validation_loss=report["snapshots"][-1]["validation_loss"])
    persist()
    (output_dir / "training_log.json").write_text(json.dumps(trainer.state.log_history, indent=2)+"\n")
    status.update(phase="complete", complete=True, adapter_weights_changed=True,
                  optimizer_steps_completed=TOTAL_STEPS)
    save_status()
    print("Pilot complete: four epochs, 96 updates, five snapshots of all 12 validation examples", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--smoke-status", required=True)
    parser.add_argument("--output-dir", default="/kaggle/working/reviewed_4b_pilot")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    groups, baseline = load_inputs(args.dataset, args.baseline)
    validate_smoke(args.smoke_status, args.dataset)
    print("Validated 48 train / 12 validation / 12 reserved test rows, frozen prompts, and successful smoke gate")
    if args.check:
        print("Dry-run OK. Four pilot epochs planned; no model downloaded or trained.")
        return
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory must be empty; use a fresh run to avoid mixing epochs")
    try:
        run_training(groups, baseline, args.dataset, output_dir)
    except Exception as exc:
        for name in ("pilot_status.json", "evaluation.json"):
            path = output_dir / name
            if path.exists():
                report = json.loads(path.read_text())
                report.update(complete=False, error_type=type(exc).__name__)
                path.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
