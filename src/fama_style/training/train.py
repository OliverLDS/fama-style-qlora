"""Gemma 3 1B SFT with LoRA or QLoRA.

Run `python train.py --check` locally with only Python installed.
Run `python train.py` on a GPU machine after installing requirements-kaggle.txt.
The notebook builder embeds this module for manual Kaggle UI upload.
"""

import argparse
import json
import os
import random
from pathlib import Path

MODEL_ID = "google/gemma-3-1b-it"

SAMPLE_ROWS = [
    {"prompt": "Explain overfitting in two sentences.", "completion": "Overfitting occurs when a model learns features specific to its training set. It then performs worse on unseen examples."},
    {"prompt": "What is endogeneity in econometrics?", "completion": "Endogeneity occurs when an explanatory variable is correlated with the regression error term, potentially biasing coefficient estimates."},
    {"prompt": "Define LoRA concisely.", "completion": "LoRA freezes pretrained weights and learns small low-rank update matrices for selected neural-network layers."},
]


def load_rows(path=None):
    """Read a local JSONL dataset or use 3 samples for a smoke test."""
    if path:
        with open(path, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    else:
        rows = SAMPLE_ROWS
    if not rows:
        raise ValueError("Training data is empty")
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("prompt"), str) or not isinstance(row.get("completion"), str):
            raise ValueError(f"Row {i} must contain string fields 'prompt' and 'completion'")
        if not row["prompt"].strip() or not row["completion"].strip():
            raise ValueError(f"Row {i} has an empty prompt or completion")
    return rows


def to_chat_rows(rows):
    return [
        {
            "prompt": [{"role": "user", "content": r["prompt"]}],
            "completion": [{"role": "assistant", "content": r["completion"]}],
        }
        for r in rows
    ]


def split_rows(rows, validation_fraction=0.2, seed=42):
    """Make a reproducible, disjoint train/validation split."""
    if len(rows) < 2:
        raise ValueError("At least 2 rows are required for a train/validation split")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    validation_count = min(len(rows) - 1, max(1, round(len(rows) * validation_fraction)))
    validation_indices = set(indices[:validation_count])
    return (
        [row for i, row in enumerate(rows) if i not in validation_indices],
        [row for i, row in enumerate(rows) if i in validation_indices],
    )


def use_labeled_splits(rows):
    """Use supplied train/validation/test labels without exposing test rows to training."""
    if not any("split" in row for row in rows):
        return None
    if any(row.get("split") not in {"train", "validation", "test"} for row in rows):
        raise ValueError("Every row must have a train, validation, or test split")
    groups = {name: [row for row in rows if row["split"] == name]
              for name in ("train", "validation", "test")}
    if any(not group for group in groups.values()):
        raise ValueError("Labeled data needs nonempty train, validation, and test splits")
    return groups


def validate_config(args):
    if args.steps is not None and args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.lora_rank <= 0:
        raise ValueError("lora_rank must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if args.max_length <= 0:
        raise ValueError("max_length must be positive")
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")


def load_hf_token():
    """Use HF_TOKEN environment variable; Kaggle Secrets as a fallback."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
    except (ImportError, Exception):
        token = None
    if token:
        os.environ["HF_TOKEN"] = token
    return token


def run_training(train_rows, validation_rows, method, steps, epochs, lora_rank,
                 learning_rate, output_dir, seed=42, max_length=1024):
    # SFTTrainer does not support DataParallel; use one of Kaggle's two T4 GPUs.
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices.split(",")[0] or "0"
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available. Use --check locally; train on a Kaggle GPU.")
    if not load_hf_token():
        raise RuntimeError("Set HF_TOKEN or add a Kaggle Secret named HF_TOKEN after accepting the Gemma license.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    load_kwargs = {"dtype": torch.float16, "device_map": {"": 0}}
    if method == "qlora":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
    model.config.use_cache = False

    lora = LoraConfig(
        r=lora_rank,
        lora_alpha=2 * lora_rank,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    sft_args = SFTConfig(
        output_dir=str(output_dir),
        max_length=max_length,
        max_steps=steps if steps is not None else -1,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=learning_rate,
        optim="adamw_torch",
        # Avoid FP16 GradScaler on T4: Gemma can produce BF16 adapter gradients.
        fp16=False,
        bf16=False,
        completion_only_loss=True,
        seed=seed,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=Dataset.from_list(to_chat_rows(train_rows)),
        eval_dataset=Dataset.from_list(to_chat_rows(validation_rows)),
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.train()
    adapted_loss = trainer.evaluate(metric_key_prefix="fine_tuned")["fine_tuned_loss"]
    base_model = trainer.accelerator.unwrap_model(trainer.model)
    with base_model.disable_adapter():
        original_loss = trainer.evaluate(metric_key_prefix="original")["original_loss"]
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = {
        "model_id": MODEL_ID,
        "method": method,
        "prompt_format": "rewrite_v1" if all(
            row.get("prompt_format") == "rewrite_v1"
            for row in train_rows + validation_rows
        ) else "bare",
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "original_validation_loss": original_loss,
        "fine_tuned_validation_loss": adapted_loss,
    }
    (output_dir / "evaluation.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(f"Validation loss: original={original_loss:.4f}; fine-tuned={adapted_loss:.4f}")
    print(f"Saved {method} adapter to: {output_dir}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate data and exit; no GPU or ML libraries required")
    parser.add_argument("--method", choices=["lora", "qlora"], default=os.getenv("METHOD", "qlora"))
    parser.add_argument("--data", default=os.getenv("TRAIN_JSONL", ""))
    parser.add_argument("--steps", type=int, default=int(os.environ["MAX_STEPS"]) if "MAX_STEPS" in os.environ else None,
                        help="optimizer steps; overrides epochs (defaults to 5 for sample data)")
    parser.add_argument("--epochs", type=float, default=float(os.getenv("EPOCHS", "1")))
    parser.add_argument("--lora-rank", type=int, default=int(os.getenv("LORA_RANK", "8")))
    parser.add_argument("--learning-rate", type=float, default=float(os.getenv("LEARNING_RATE", "2e-4")))
    parser.add_argument("--max-length", type=int, default=int(os.getenv("MAX_LENGTH", "1024")),
                        help="maximum tokens for the full prompt and completion")
    parser.add_argument("--validation-fraction", type=float, default=float(os.getenv("VALIDATION_FRACTION", "0.2")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("SPLIT_SEED", "42")))
    parser.add_argument("--output", default=os.getenv("OUTPUT_DIR", "/kaggle/working/gemma3_adapter"))
    args = parser.parse_args(argv)
    validate_config(args)
    rows = load_rows(args.data or None)
    labeled = use_labeled_splits(rows)
    if labeled is not None:
        train_rows, validation_rows = labeled["train"], labeled["validation"]
        split_summary = f"train={len(train_rows)}, validation={len(validation_rows)}, untouched_test={len(labeled['test'])}"
    else:
        train_rows, validation_rows = split_rows(rows, args.validation_fraction, args.seed)
        split_summary = f"train={len(train_rows)}, validation={len(validation_rows)}"
    steps = args.steps if args.steps is not None else (5 if not args.data else None)
    schedule = f"steps={steps}" if steps is not None else f"epochs={args.epochs}"
    print(f"Validated {len(rows)} examples; {split_summary}; method={args.method}; {schedule}; max_length={args.max_length}")
    if args.check:
        print("Dry-run OK. No model downloaded or trained.")
        return
    run_training(train_rows, validation_rows, args.method, steps, args.epochs,
                 args.lora_rank, args.learning_rate, Path(args.output), args.seed, args.max_length)


if __name__ == "__main__":
    main()
