"""Controlled two-pair QLoRA memorization diagnostic for the instructed pilot."""

import argparse
import difflib
import json
import os
from pathlib import Path


MODEL_ID = "google/gemma-3-1b-it"
TRAIN_IDS = ("p001", "p014")
HELD_OUT_ID = "p003"
SNAPSHOT_STEPS = (0, 10, 30, 60)
MAX_LENGTH = 1024


def load_examples(path):
    with Path(path).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    counts = {split: sum(row.get("split") == split for row in rows)
              for split in ("train", "validation", "test")}
    if len(rows) != 78 or counts != {"train": 54, "validation": 12, "test": 12}:
        raise ValueError("Expected the instructed 78-pair pilot with 54/12/12 splits")
    selected = {}
    for row in rows:
        ids = row.get("source_ids")
        if ids in (["p001"], ["p014"], ["p003"]):
            key = ids[0]
            if key in selected:
                raise ValueError(f"Duplicate diagnostic ID: {key}")
            selected[key] = row
    if set(selected) != set(TRAIN_IDS) | {HELD_OUT_ID}:
        raise ValueError("Missing one of p001, p014, or p003")
    for key, row in selected.items():
        expected_split = "test" if key == HELD_OUT_ID else "train"
        if row.get("split") != expected_split:
            raise ValueError(f"{key} must be in the {expected_split} split")
        if (row.get("prompt_format") != "rewrite_v1"
                or not isinstance(row.get("prompt"), str)
                or not row["prompt"].startswith("Rewrite the passage below")
                or not isinstance(row.get("plain_english"), str)
                or not isinstance(row.get("completion"), str)
                or not row["plain_english"].strip()
                or not row["completion"].strip()):
            raise ValueError(f"{key} needs a nonempty instructed prompt, source, and target")
    return selected


def to_chat_rows(rows):
    return [{"prompt": [{"role": "user", "content": row["prompt"]}],
             "completion": [{"role": "assistant", "content": row["completion"]}]}
            for row in rows]


def load_hf_token():
    token = os.environ.get("HF_TOKEN")
    if token:
        return True
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        token = None
    if token:
        os.environ["HF_TOKEN"] = token
    return bool(token)


def normalized(text):
    return " ".join(text.split())


def text_similarity(left, right):
    return round(difflib.SequenceMatcher(None, normalized(left), normalized(right)).ratio(), 4)


def snapshot(trainer, tokenizer, selected, step, output_path, torch,
             snapshot_steps=SNAPSHOT_STEPS, prior_steps=0):
    """Use the exact prepared labels for loss, then generate from the same prompts."""
    model = trainer.model
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    records = []
    try:
        for index, key in enumerate(TRAIN_IDS):
            row = selected[key]
            batch = trainer.data_collator([trainer.train_dataset[index]])
            batch = {name: value.to(device) for name, value in batch.items()}
            with torch.inference_mode():
                loss = float(model(**batch).loss.item())
            records.append({"source_id": key, "split": "train", "loss": loss,
                            "prompt": row["prompt"], "source": row["plain_english"],
                            "reference": row["completion"]})
        if step in (0, snapshot_steps[-1]):
            row = selected[HELD_OUT_ID]
            records.append({"source_id": HELD_OUT_ID, "split": "test",
                            "prompt": row["prompt"], "source": row["plain_english"],
                            "reference": row["completion"]})
        for record in records:
            encoded = tokenizer.apply_chat_template(
                [{"role": "user", "content": record["prompt"]}],
                tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True,
            )
            input_ids = encoded["input_ids"].to(device)
            target_tokens = len(tokenizer.encode(record["reference"], add_special_tokens=False))
            max_new_tokens = min(320, max(128, target_tokens + 32))
            with torch.inference_mode():
                generated = model.generate(
                    input_ids, max_new_tokens=max_new_tokens,
                    do_sample=False, use_cache=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            output = tokenizer.decode(
                generated[0, input_ids.shape[-1]:], skip_special_tokens=True).strip()
            record["output"] = output
            record["source_similarity"] = text_similarity(output, record["source"])
            record["reference_similarity"] = text_similarity(output, record["reference"])
            record["copied_source"] = normalized(output) == normalized(record["source"])
    finally:
        if was_training:
            model.train()
    report = json.loads(output_path.read_text(encoding="utf-8")) if step and output_path.exists() else {
        "model_id": MODEL_ID,
        "purpose": "two-example memorization diagnostic, not a style evaluation",
        "train_ids": list(TRAIN_IDS), "held_out_id": HELD_OUT_ID,
        "snapshot_steps": list(snapshot_steps), "max_length": MAX_LENGTH,
        "prior_optimizer_steps": prior_steps,
        "optimizer_state_restored": False,
        "snapshots": [],
    }
    report["snapshots"].append({"step": step,
                                "cumulative_optimizer_steps": prior_steps + step,
                                "records": records})
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = [(record["source_id"], round(record.get("loss", -1), 3),
                record["copied_source"], record["reference_similarity"])
               for record in records]
    print(f"Step {step} snapshot (id, loss, copied source, target similarity): {summary}", flush=True)


def run_diagnostic(selected, output_dir):
    # SFTTrainer uses one of Kaggle's T4 GPUs; no local GPU is needed for --check.
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("Select a Kaggle T4 GPU for this diagnostic")
    if not load_hf_token():
        raise RuntimeError("Enable the Kaggle Secret named HF_TOKEN")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "overfit_two_report.json"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
        ),
    )
    model.config.use_cache = False
    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                      target_modules=["q_proj", "v_proj"], bias="none",
                      task_type="CAUSAL_LM")
    args = SFTConfig(
        output_dir=str(output_dir / "trainer"),
        max_length=MAX_LENGTH,
        max_steps=SNAPSHOT_STEPS[-1],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=2e-4,
        optim="adamw_torch",
        fp16=False, bf16=False,
        completion_only_loss=True,
        seed=42,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )
    train_rows = [selected[key] for key in TRAIN_IDS]
    trainer = SFTTrainer(
        model=model, args=args,
        train_dataset=Dataset.from_list(to_chat_rows(train_rows)),
        processing_class=tokenizer, peft_config=lora,
    )
    if len(trainer.train_dataset) != 2:
        raise ValueError("Expected exactly two prepared training examples")

    class SnapshotCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step in SNAPSHOT_STEPS[1:]:
                snapshot(trainer, tokenizer, selected, state.global_step, report_path, torch)
            return control

    trainer.add_callback(SnapshotCallback())
    snapshot(trainer, tokenizer, selected, 0, report_path, torch)
    trainer.train()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["training_log"] = trainer.state.log_history
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    trainer.model.save_pretrained(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))
    print(f"Saved diagnostic report to {report_path}")
    print(f"Saved two-example adapter to {output_dir / 'adapter'}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="/kaggle/working/overfit_two")
    parser.add_argument("--check", action="store_true",
                        help="validate data without model weights or GPU")
    args = parser.parse_args(argv)
    selected = load_examples(args.dataset)
    print("Validated train p001,p014 and untouched test p003")
    if args.check:
        print("Dry-run OK. No model downloaded or trained.")
        return
    run_diagnostic(selected, Path(args.output_dir))


if __name__ == "__main__":
    main()
