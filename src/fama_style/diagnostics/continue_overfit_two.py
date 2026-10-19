"""Continue the saved two-example QLoRA adapter for 120 more optimizer steps."""

import argparse
import json
import os
from pathlib import Path

from fama_style.diagnostics.overfit_two import (HELD_OUT_ID, MAX_LENGTH, MODEL_ID, TRAIN_IDS,
                         load_examples, load_hf_token, snapshot, to_chat_rows)


PRIOR_STEPS = 60
SNAPSHOT_STEPS = (0, 30, 60, 120)
EXPECTED_ADAPTER = {
    "base_model_name_or_path": MODEL_ID,
    "peft_type": "LORA",
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}


def validate_prior_adapter(adapter_dir, require_weights=True):
    adapter_dir = Path(adapter_dir)
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    mismatches = [name for name, value in EXPECTED_ADAPTER.items()
                  if config.get(name) != value]
    if mismatches or set(config.get("target_modules", [])) != {"q_proj", "v_proj"}:
        raise ValueError(f"Prior adapter does not match the two-example QLoRA setup: {mismatches}")
    report_path = adapter_dir.parent / "overfit_two_report.json"
    prior = json.loads(report_path.read_text(encoding="utf-8"))
    if (prior.get("model_id") != MODEL_ID
            or prior.get("train_ids") != list(TRAIN_IDS)
            or prior.get("held_out_id") != HELD_OUT_ID
            or not prior.get("snapshots")
            or prior["snapshots"][-1].get("step") != PRIOR_STEPS):
        raise ValueError("Attached adapter is not the completed 60-step two-example diagnostic")
    if require_weights and not (adapter_dir / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("Missing adapter_model.safetensors in the attached run output")
    return prior


def run_continuation(selected, adapter_dir, output_dir):
    # One Kaggle T4 is sufficient; local --check imports no ML packages.
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"
    import torch
    from datasets import Dataset
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("Select a Kaggle T4 GPU for the continuation")
    if not load_hf_token():
        raise RuntimeError("Enable the Kaggle Secret named HF_TOKEN")
    validate_prior_adapter(adapter_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "overfit_two_continuation_report.json"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
        ),
    )
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(
        base, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=True)
    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    if trainable == 0:
        raise RuntimeError("Loaded adapter has no trainable parameters")
    print(f"Loaded prior adapter with {trainable} trainable parameters", flush=True)

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
        processing_class=tokenizer,
    )
    if len(trainer.train_dataset) != 2:
        raise ValueError("Expected exactly two prepared training examples")

    class SnapshotCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step in SNAPSHOT_STEPS[1:]:
                snapshot(trainer, tokenizer, selected, state.global_step,
                         report_path, torch, SNAPSHOT_STEPS, PRIOR_STEPS)
            return control

    trainer.add_callback(SnapshotCallback())
    snapshot(trainer, tokenizer, selected, 0, report_path, torch,
             SNAPSHOT_STEPS, PRIOR_STEPS)
    trainer.train()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["training_log"] = trainer.state.log_history
    report["learning_rate"] = 2e-4
    report["prior_adapter_training_steps"] = PRIOR_STEPS
    report["additional_training_steps"] = SNAPSHOT_STEPS[-1]
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    trainer.model.save_pretrained(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))
    print(f"Saved continuation report to {report_path}")
    print(f"Saved continued adapter to {output_dir / 'adapter'}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", default="/kaggle/working/overfit_two_continued")
    parser.add_argument("--check", action="store_true",
                        help="validate inputs without model weights or GPU")
    args = parser.parse_args(argv)
    selected = load_examples(args.dataset)
    validate_prior_adapter(args.adapter, require_weights=not args.check)
    print("Validated train p001,p014 and the completed 60-step adapter")
    if args.check:
        print("Dry-run OK. No model downloaded or trained.")
        return
    run_continuation(selected, Path(args.adapter), Path(args.output_dir))


if __name__ == "__main__":
    main()
