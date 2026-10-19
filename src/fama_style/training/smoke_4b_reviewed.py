"""Five-step 4B QLoRA pipeline/memory smoke test on semantically reviewed pairs."""

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

from fama_style.evaluation.compare_few_shot import DEMO_IDS, INSTRUCTION, load_hf_token, make_prompt, normalized, validate_completion

MODEL_ID = "google/gemma-3-4b-it"
FORMAT = "fama_style_few_shot_reviewed_v1"
PARENT_SHA = "3fabac4ddfedb947c88bdbb3956f21f7d524f1f51504fbca6067efffd272a8ed"
SMOKE_ANCHORS = ("p031", "p060")
EVAL_IDS = ("p008", "p050")
STEPS = 5
MAX_LENGTH = 2048
TARGET_PATTERN = r".*language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)$"


def choose_smoke_ids(lengths):
    if len(lengths) < 3 or any(key not in lengths for key in SMOKE_ANCHORS):
        raise ValueError("Smoke selection needs at least three training rows and both anchors")
    longest = max(lengths, key=lengths.get)
    candidates = [longest, *SMOKE_ANCHORS, *sorted(lengths, key=lengths.get, reverse=True)]
    return tuple(dict.fromkeys(candidates))[:3]


def load_inputs(dataset_path, baseline_path):
    rows = [json.loads(line) for line in Path(dataset_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    groups = {key: [r for r in rows if r.get("split") == key] for key in ("train", "validation", "test")}
    if len(rows) != 72 or [len(groups[k]) for k in groups] != [48, 12, 12]:
        raise ValueError("Expected reviewed Dataset: 48 train, 12 validation, 12 reserved test rows")
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    validate_completion(baseline)
    if (baseline.get("model_id") != MODEL_ID or baseline.get("complete") is not True
            or baseline.get("training_called") is not False or baseline.get("test_rows_used") != 0
            or baseline.get("dataset_sha256") != PARENT_SHA
            or baseline.get("quantization") != "NF4 double quantization, FP32 compute"
            or baseline.get("generation") != {"do_sample": False, "max_new_tokens": 512}
            or baseline.get("instruction") != INSTRUCTION
            or [d.get("source_id") for d in baseline.get("demonstrations", [])] != list(DEMO_IDS)):
        raise ValueError("Attach the completed FP32 4B report from compare-few-shot version 2")
    demos = baseline["demonstrations"]
    for row in groups["train"] + groups["validation"]:
        if (row.get("prompt_format") != FORMAT or row.get("parent_dataset_sha256") != PARENT_SHA
                or not isinstance(row.get("completion"), str) or not row["completion"].strip()
                or row.get("prompt") != make_prompt(demos, row["plain_english"])):
            raise ValueError("Dataset prompts must use the reviewed format and frozen baseline demonstrations")
    train_ids = [tuple(r["source_ids"]) for r in groups["train"]]
    if (len(set(train_ids)) != 48 or any((key,) in train_ids for key in DEMO_IDS)
            or any((key,) not in train_ids for key in SMOKE_ANCHORS)):
        raise ValueError("Training IDs must be unique, include smoke rows, and exclude demonstrations")
    for row, previous in zip(groups["validation"], baseline["results"]):
        if (row["source_ids"] != previous["source_ids"] or row["plain_english"] != previous["source"]
                or row["completion"] != previous["reference"] or previous.get("hit_output_limit")
                or hashlib.sha256(row["prompt"].encode()).hexdigest() != previous["prompt_sha256"]):
            raise ValueError("Validation sources, targets, and prompt hashes must match the saved baseline")
    if any(sum(r["source_ids"] == [key] for r in groups["validation"]) != 1 for key in EVAL_IDS):
        raise ValueError("Missing a fixed smoke validation row")
    return groups, baseline


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
    status_path = output_dir / "smoke_status.json"
    status = {"model_id": MODEL_ID, "complete": False, "purpose": "pipeline and memory smoke test",
              "trained_ids": [], "selection": "longest training row plus two anchors; three unique rows",
              "validation_ids": list(EVAL_IDS), "optimizer_steps_planned": STEPS,
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
    lengths = {a["source_ids"][0]: a["full_tokens"] for a in audits if a["split"] == "train"}
    smoke_ids = choose_smoke_ids(lengths)
    status.update(phase="labels_checked", max_full_tokens=max(a["full_tokens"] for a in audits),
                  trained_ids=list(smoke_ids), longest_training_tokens=max(lengths.values()))
    save_status()
    print("Verified completion-only labels and lengths for all 48 train and 12 validation rows", flush=True)

    def collate(features):
        if len(features) != 1:
            raise ValueError("This smoke test uses batch size one")
        return {key: torch.tensor([value], dtype=torch.long) for key, value in features[0].items()}

    class GradientChecks(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            grads = [p.grad.detach() for p in parameters.values() if p.grad is not None]
            if not grads or any(not bool(torch.isfinite(g).all()) for g in grads):
                raise RuntimeError("Missing or non-finite adapter gradients")
            norm_squared = sum(float(g.float().square().sum().item()) for g in grads)
            if not math.isfinite(norm_squared) or norm_squared <= 0:
                raise RuntimeError("Adapter gradient norm must be finite and positive")
            status["gradient_checks"].append({"step": state.global_step+1, "finite": True,
                                               "norm": math.sqrt(norm_squared)})
            save_status()
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs and not math.isfinite(logs["loss"]):
                raise RuntimeError("Non-finite training loss")

    args = TrainingArguments(output_dir=str(output_dir / "trainer"), max_steps=STEPS,
        per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=2,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=5e-5, optim="adamw_torch", fp16=False, bf16=False,
        logging_steps=1, save_strategy="no", report_to="none", seed=42,
        remove_unused_columns=False, prediction_loss_only=True, label_names=["labels"])
    class FiniteLossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            result = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
            loss = result[0] if return_outputs else result
            if not bool(torch.isfinite(loss).all()):
                raise RuntimeError("Non-finite loss during model forward pass")
            return result

    trainer = FiniteLossTrainer(model=model, args=args, processing_class=processor, data_collator=collate,
        train_dataset=Dataset.from_list([encoded[(key,)] for key in smoke_ids]),
        eval_dataset=Dataset.from_list([encoded[(key,)] for key in EVAL_IDS]),
        callbacks=[GradientChecks()])
    trainer.model_accepts_loss_kwargs = False
    with model.disable_adapter():
        original_loss = trainer.evaluate()["eval_loss"]
    if not math.isfinite(original_loss):
        raise RuntimeError("Base validation loss is non-finite")
    status["phase"] = "training"
    save_status()
    trainer.train()
    adapted_loss = trainer.evaluate()["eval_loss"]
    changed = any(not torch.equal(before[name], value.detach().cpu()) for name, value in parameters.items())
    if (trainer.state.global_step != STEPS or len(status["gradient_checks"]) != STEPS
            or not changed or not math.isfinite(adapted_loss)):
        raise RuntimeError("Smoke test did not complete five finite optimizer updates with changed adapter weights")
    status.update(phase="training_finished", optimizer_steps_completed=STEPS, adapter_weights_changed=True)
    save_status()
    model.save_pretrained(str(output_dir / "adapter"))
    processor.save_pretrained(str(output_dir / "adapter"))
    (output_dir / "training_log.json").write_text(json.dumps(trainer.state.log_history, indent=2) + "\n")

    class FiniteScores(LogitsProcessor):
        def __call__(self, input_ids, scores):
            if bool(torch.isnan(scores).any() | torch.isposinf(scores).any()
                    | ~torch.isfinite(scores).any(dim=-1).all()):
                raise RuntimeError("Non-finite generation scores after training")
            return scores

    model.eval()
    generations = []
    for key in EVAL_IDS:
        row = next(r for r in groups["validation"] if r["source_ids"] == [key])
        old = next(r for r in baseline["results"] if r["source_ids"] == [key])
        inputs = processor.apply_chat_template([{"role": "user", "content": [
            {"type": "text", "text": row["prompt"]}]}], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt")
        inputs = {name: value.to("cuda:0") for name, value in inputs.items()}
        with torch.inference_mode():
            tokens = model.generate(**inputs, max_new_tokens=512, do_sample=False, use_cache=True,
                pad_token_id=tokenizer.pad_token_id, remove_invalid_values=False,
                logits_processor=LogitsProcessorList([FiniteScores()]))
        generated = tokens[0, inputs["input_ids"].shape[-1]:]
        output = tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not output:
            raise RuntimeError("Adapted generation is empty")
        generations.append({"source_ids": [key], "source": row["plain_english"],
                            "reference": row["completion"], "baseline_output": old["output"],
                            "smoke_adapter_output": output, "output_tokens": int(generated.numel()),
                            "at_output_budget": generated.numel() == 512,
                            "copied_source": normalized(output) == normalized(row["plain_english"])})
    evaluation = {"model_id": MODEL_ID, "purpose": "pipeline smoke test, not a style experiment",
                  "prompt_format": FORMAT, "train_rows": 3, "validation_rows": 2, "optimizer_steps": STEPS,
                  "original_validation_loss": original_loss, "fine_tuned_validation_loss": adapted_loss,
                  "adapter_weights_changed": True, "test_rows_used": 0, "generations": generations}
    (output_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    status.update(phase="complete", complete=True)
    save_status()
    print(f"Smoke complete: 3 train rows, 5 optimizer steps; saved {output_dir / 'evaluation.json'}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output-dir", default="/kaggle/working/gemma4b_reviewed_smoke")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    groups, baseline = load_inputs(args.dataset, args.baseline)
    print("Validated reviewed Dataset, demonstration exclusion, and identical baseline validation prompts")
    if args.check:
        print("Dry-run OK. No model downloaded or trained.")
        return
    output_dir = Path(args.output_dir)
    try:
        run_training(groups, baseline, args.dataset, output_dir)
    except Exception as exc:
        status_path = output_dir / "smoke_status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            status.update(complete=False, error_type=type(exc).__name__)
            status_path.write_text(json.dumps(status, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
