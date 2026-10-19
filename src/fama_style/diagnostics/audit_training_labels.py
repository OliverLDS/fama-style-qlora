"""Inspect the batches prepared by the pilot SFTTrainer without training."""

import argparse
import difflib
import json
import os
from pathlib import Path


MODEL_ID = "google/gemma-3-1b-it"
MAX_LENGTH = 1024


def load_pilot(path):
    with Path(path).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    counts = {split: sum(row.get("split") == split for row in rows)
              for split in ("train", "validation", "test")}
    if len(rows) != 78 or counts != {"train": 54, "validation": 12, "test": 12}:
        raise ValueError("Expected the instructed 78-pair pilot with 54/12/12 splits")
    if any(row.get("prompt_format") != "rewrite_v1" or
           not isinstance(row.get("prompt"), str) or
           not row["prompt"].startswith("Rewrite the passage below") or
           not isinstance(row.get("plain_english"), str) or
           not isinstance(row.get("completion"), str) or
           not row["completion"].strip() for row in rows):
        raise ValueError("Expected nonempty rewrite_v1 prompt/completion pairs")
    if not any(row.get("source_ids") == ["p014"] and row["split"] == "train"
               for row in rows):
        raise ValueError("Training row p014 is missing")
    return rows


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


def compare_prepared_row(row, prepared, collator, tokenizer):
    batch = collator([prepared])
    input_ids = batch["input_ids"][0].tolist()
    labels = batch["labels"][0].tolist()
    attention = batch["attention_mask"][0].tolist()
    supervised_positions = [index for index, (label, mask) in
                            enumerate(zip(labels, attention)) if label != -100 and mask == 1]
    supervised_ids = [labels[index] for index in supervised_positions]
    first = supervised_positions[0] if supervised_positions else len(labels)
    supervised_text = normalized(tokenizer.decode(supervised_ids, skip_special_tokens=True))
    masked_prefix = normalized(tokenizer.decode(input_ids[:first], skip_special_tokens=True))
    reference = normalized(row["completion"])
    source = normalized(row["plain_english"])
    return {
        "source_ids": row["source_ids"],
        "split": row["split"],
        "prepared_tokens": sum(attention),
        "supervised_tokens": len(supervised_positions),
        "first_supervised_position": first if supervised_positions else None,
        "labels_match_input_ids": all(labels[index] == input_ids[index]
                                      for index in supervised_positions),
        "source_in_masked_prefix": source in masked_prefix,
        "reference_exact_after_whitespace_normalization": supervised_text == reference,
        "reference_similarity": round(difflib.SequenceMatcher(None, supervised_text, reference).ratio(), 4),
        "source_exact_after_whitespace_normalization": supervised_text == source,
        "source_similarity": round(difflib.SequenceMatcher(None, supervised_text, source).ratio(), 4),
    }


def run_audit(rows, output_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"
    import torch
    import transformers
    import trl
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("Select a Kaggle T4 GPU to construct the version 9 trainer")
    if not load_hf_token():
        raise RuntimeError("Enable the Kaggle Secret named HF_TOKEN")

    groups = {split: [row for row in rows if row["split"] == split]
              for split in ("train", "validation", "test")}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    token_lengths = []
    for row in rows:
        chat_row = to_chat_rows([row])[0]
        conversation = chat_row["prompt"] + chat_row["completion"]
        encoded = tokenizer.apply_chat_template(
            conversation, tokenize=True, return_dict=True, return_tensors="pt")
        token_lengths.append({"source_ids": row["source_ids"], "split": row["split"],
                              "full_tokens": int(encoded["input_ids"].shape[-1]),
                              "over_limit": encoded["input_ids"].shape[-1] > MAX_LENGTH})
    print("Tokenized all 78 prompt/target pairs", flush=True)

    print("Loading 4-bit Gemma for trainer construction", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map={"": 0},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        ),
    )
    model.config.use_cache = False
    print("Gemma loaded; constructing SFTTrainer without training", flush=True)
    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                      target_modules=["q_proj", "v_proj"], bias="none",
                      task_type="CAUSAL_LM")
    args = SFTConfig(
        output_dir="/kaggle/working/label_audit_tmp",
        max_length=MAX_LENGTH,
        max_steps=-1,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=2e-4,
        optim="adamw_torch",
        fp16=False,
        bf16=False,
        completion_only_loss=True,
        seed=42,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(to_chat_rows(groups["train"])),
        eval_dataset=Dataset.from_list(to_chat_rows(groups["validation"])),
        processing_class=tokenizer,
        peft_config=lora,
    )
    if len(trainer.train_dataset) != 54 or len(trainer.eval_dataset) != 12:
        raise ValueError("Prepared dataset sizes differ from the version 9 split")
    print("SFTTrainer prepared 54 training and 12 validation rows", flush=True)
    prepared_rows = []
    for split, prepared_dataset in (("train", trainer.train_dataset),
                                    ("validation", trainer.eval_dataset)):
        for source_row, prepared in zip(groups[split], prepared_dataset):
            prepared_rows.append(compare_prepared_row(
                source_row, prepared, trainer.data_collator, tokenizer))
        print(f"Checked {len(groups[split])} {split} label batches", flush=True)
    report = {
        "model_id": MODEL_ID,
        "training_called": False,
        "transformers_version": transformers.__version__,
        "trl_version": trl.__version__,
        "max_length": MAX_LENGTH,
        "token_lengths": token_lengths,
        "prepared_rows": prepared_rows,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved label audit to {output_path}")
    print(f"Full sequences over {MAX_LENGTH} tokens: {sum(item['over_limit'] for item in token_lengths)}/78")
    print(f"Prepared rows with exact reference labels: "
          f"{sum(item['reference_exact_after_whitespace_normalization'] for item in prepared_rows)}/66")
    print(f"Prepared rows with zero supervised tokens: "
          f"{sum(item['supervised_tokens'] == 0 for item in prepared_rows)}/66")
    print("p014 summary:", next(item for item in prepared_rows
                                if item["source_ids"] == ["p014"]))
    print("No optimizer step or training was run")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="/kaggle/working/label_audit.json")
    parser.add_argument("--check", action="store_true",
                        help="validate dataset without model, GPU, or downloaded weights")
    args = parser.parse_args(argv)
    rows = load_pilot(args.dataset)
    print(f"Validated {len(rows)} instructed rows")
    if args.check:
        print("Dry-run OK. No model downloaded or trained.")
        return
    run_audit(rows, args.output)


if __name__ == "__main__":
    main()
