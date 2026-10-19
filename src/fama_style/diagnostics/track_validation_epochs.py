"""Track held-out rewrite quality over four QLoRA pilot epochs."""

import argparse
import difflib
import json
import os
import re
from pathlib import Path


MODEL_ID = "google/gemma-3-1b-it"
EPOCHS = 4
MAX_LENGTH = 1024
MONITORED_IDS = ("p008", "p026", "p050", "p080")


def load_pilot(path):
    with Path(path).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    groups = {name: [row for row in rows if row.get("split") == name]
              for name in ("train", "validation", "test")}
    if len(rows) != 78 or {name: len(group) for name, group in groups.items()} != {
            "train": 54, "validation": 12, "test": 12}:
        raise ValueError("Expected 78 instructed rows with 54/12/12 splits")
    for row in rows:
        if (row.get("prompt_format") != "rewrite_v1"
                or not isinstance(row.get("prompt"), str)
                or not row["prompt"].startswith("Rewrite the passage below")
                or not isinstance(row.get("plain_english"), str)
                or not isinstance(row.get("completion"), str)
                or not row["plain_english"].strip()
                or not row["completion"].strip()):
            raise ValueError("All pilot rows need the instructed prompt and target")
    selected = {}
    for row in groups["validation"]:
        ids = row.get("source_ids")
        if ids and len(ids) == 1 and ids[0] in MONITORED_IDS:
            if ids[0] in selected:
                raise ValueError(f"Duplicate monitored ID: {ids[0]}")
            selected[ids[0]] = row
    if set(selected) != set(MONITORED_IDS):
        raise ValueError("Monitored IDs must all be unique validation rows")
    return groups, selected


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


def similarity(left, right):
    return round(difflib.SequenceMatcher(None, normalized(left), normalized(right)).ratio(), 4)


def source_sentences(text):
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
            if part.strip()]


def write_claim_review(report, path):
    """Create a human review worksheet; surface metrics do not judge claims."""
    lines = [
        "# Held-out claim preservation by epoch", "",
        "For each source claim, mark preserved, changed, or omitted in each output.",
        "Record unsupported additions separately. This worksheet contains no automatic",
        "semantic judgment. Leave boxes blank until a human reviews the passage.", "",
    ]
    for source_id in MONITORED_IDS:
        initial = next(row for row in report["snapshots"][0]["generations"]
                       if row["source_id"] == source_id)
        claims = source_sentences(initial["source"])
        lines.extend([f"## {source_id}", "", "### Source claims", ""])
        for index, claim in enumerate(claims, 1):
            lines.append(f"{index}. {claim}")
        lines.extend(["", "### Reference", "", initial["reference"], ""])
        for snap in report["snapshots"]:
            row = next(item for item in snap["generations"]
                       if item["source_id"] == source_id)
            lines.extend([f"### Epoch {snap['epoch']} (step {snap['optimizer_step']})", "",
                          row["output"], "", "Claim checks:", ""])
            for index in range(1, len(claims) + 1):
                lines.append(f"- Claim {index}: [ ] preserved  [ ] changed  [ ] omitted")
            lines.extend(["- Unsupported additions: ", "- Style or clarity notes: ", ""])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def snapshot(trainer, tokenizer, validation_rows, monitored, epoch,
             optimizer_step, report_path, review_path, torch):
    model = trainer.model
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    losses = []
    generations = []
    try:
        for index, row in enumerate(validation_rows):
            batch = trainer.data_collator([trainer.eval_dataset[index]])
            batch = {name: value.to(device) for name, value in batch.items()}
            supervised = int((batch["labels"] != -100).sum().item())
            if supervised == 0:
                raise ValueError(f"Validation row {row['source_ids']} has no supervised tokens")
            with torch.inference_mode():
                loss = float(model(**batch).loss.item())
            losses.append({"source_ids": row["source_ids"], "loss": loss,
                           "supervised_tokens": supervised})
        for source_id in MONITORED_IDS:
            row = monitored[source_id]
            encoded = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True,
            )
            input_ids = encoded["input_ids"].to(device)
            target_tokens = len(tokenizer.encode(row["completion"], add_special_tokens=False))
            max_new_tokens = min(512, max(128, target_tokens + 64))
            with torch.inference_mode():
                generated = model.generate(
                    input_ids, max_new_tokens=max_new_tokens,
                    do_sample=False, use_cache=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            output = tokenizer.decode(
                generated[0, input_ids.shape[-1]:], skip_special_tokens=True).strip()
            generations.append({
                "source_id": source_id,
                "source": row["plain_english"],
                "reference": row["completion"],
                "output": output,
                "copied_source": normalized(output) == normalized(row["plain_english"]),
                "source_similarity": similarity(output, row["plain_english"]),
                "reference_similarity": similarity(output, row["completion"]),
                "claim_review_status": "pending_manual_review",
            })
    finally:
        if was_training:
            model.train()
    total_tokens = sum(item["supervised_tokens"] for item in losses)
    weighted_loss = sum(item["loss"] * item["supervised_tokens"]
                        for item in losses) / total_tokens
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() and epoch else {
        "model_id": MODEL_ID,
        "purpose": "four-epoch pilot with fixed validation generations and manual claim review",
        "prompt_format": "rewrite_v1", "train_rows": 54,
        "validation_rows": 12, "test_rows_used": 0,
        "monitored_validation_ids": list(MONITORED_IDS),
        "max_length": MAX_LENGTH, "planned_epochs": EPOCHS,
        "snapshots": [],
    }
    report["snapshots"].append({
        "epoch": epoch, "optimizer_step": optimizer_step,
        "validation_loss": weighted_loss,
        "validation_supervised_tokens": total_tokens,
        "validation_row_losses": losses,
        "generations": generations,
    })
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_claim_review(report, review_path)
    print(f"Epoch {epoch}, step {optimizer_step}: validation loss={weighted_loss:.4f}; "
          f"exact input copies={sum(row['copied_source'] for row in generations)}/4", flush=True)


def run_pilot(groups, monitored, output_dir, learning_rate=2e-4):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("Select a Kaggle T4 GPU for the pilot")
    if not load_hf_token():
        raise RuntimeError("Enable the Kaggle Secret named HF_TOKEN")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "epoch_validation_report.json"
    review_path = output_dir / "epoch_claim_review.md"
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
        max_steps=-1,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=learning_rate,
        optim="adamw_torch",
        fp16=False, bf16=False,
        completion_only_loss=True,
        seed=42,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model, args=args,
        train_dataset=Dataset.from_list(to_chat_rows(groups["train"])),
        eval_dataset=Dataset.from_list(to_chat_rows(groups["validation"])),
        processing_class=tokenizer, peft_config=lora,
    )
    if len(trainer.train_dataset) != 54 or len(trainer.eval_dataset) != 12:
        raise ValueError("Prepared training/validation sizes differ from the pilot")

    class EpochSnapshotCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            epoch = round(state.epoch or 0)
            if epoch not in range(1, EPOCHS + 1) or abs((state.epoch or 0) - epoch) > 0.01:
                raise ValueError(f"Unexpected epoch boundary: {state.epoch}")
            snapshot(trainer, tokenizer, groups["validation"], monitored,
                     epoch, state.global_step, report_path, review_path, torch)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["initial_learning_rate"] = learning_rate
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            trainer.model.save_pretrained(str(output_dir / f"adapter_epoch_{epoch}"))
            tokenizer.save_pretrained(str(output_dir / f"adapter_epoch_{epoch}"))
            return control

    trainer.add_callback(EpochSnapshotCallback())
    snapshot(trainer, tokenizer, groups["validation"], monitored,
             0, 0, report_path, review_path, torch)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["initial_learning_rate"] = learning_rate
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    trainer.train()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if [item["epoch"] for item in report["snapshots"]] != list(range(EPOCHS + 1)):
        raise RuntimeError("Missing one or more epoch snapshots")
    report["training_log"] = trainer.state.log_history
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved epoch report to {report_path}")
    print(f"Saved manual claim worksheet to {review_path}")
    print("Saved adapters for epochs 1 through 4; test rows were not used")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="/kaggle/working/epoch_validation")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--check", action="store_true",
                        help="validate data without model weights or GPU")
    args = parser.parse_args(argv)
    if not 0 < args.learning_rate < float("inf"):
        raise ValueError("learning-rate must be finite and positive")
    groups, monitored = load_pilot(args.dataset)
    print(f"Validated 54 train, 12 validation, 12 untouched test rows; monitored {','.join(MONITORED_IDS)}")
    if args.check:
        print("Dry-run OK. No model downloaded or trained.")
        return
    run_pilot(groups, monitored, Path(args.output_dir), args.learning_rate)


if __name__ == "__main__":
    main()
