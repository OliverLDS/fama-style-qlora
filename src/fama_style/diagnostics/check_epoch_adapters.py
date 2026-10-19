"""Reload saved epoch adapters and reproduce two validation generations."""

import argparse
import json
import os
from pathlib import Path

from fama_style.diagnostics.track_validation_epochs import MODEL_ID, load_pilot, load_hf_token, normalized

CHECK_IDS = ("p008", "p050")
CHECK_EPOCHS = (2, 4)


def validate_run(run_dir, require_weights=True):
    run_dir = Path(run_dir)
    report = json.loads((run_dir / "epoch_validation_report.json").read_text(encoding="utf-8"))
    if (report.get("model_id") != MODEL_ID or report.get("prompt_format") != "rewrite_v1"
            or report.get("train_rows") != 54 or report.get("validation_rows") != 12
            or report.get("test_rows_used") != 0
            or [item.get("epoch") for item in report.get("snapshots", [])] != [0, 1, 2, 3, 4]):
        raise ValueError("Attach the completed four-epoch validation pilot output")
    for epoch in CHECK_EPOCHS:
        folder = run_dir / f"adapter_epoch_{epoch}"
        config = json.loads((folder / "adapter_config.json").read_text(encoding="utf-8"))
        if (config.get("base_model_name_or_path") != MODEL_ID
                or config.get("peft_type") != "LORA" or config.get("r") != 8
                or config.get("lora_alpha") != 16 or config.get("lora_dropout") != 0.05
                or set(config.get("target_modules", [])) != {"q_proj", "v_proj"}):
            raise ValueError(f"Epoch {epoch} adapter has unexpected model or LoRA settings")
        if require_weights and not (folder / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"Missing epoch {epoch} adapter weights")
        snap = next(item for item in report["snapshots"] if item["epoch"] == epoch)
        if any(sum(row.get("source_id") == key for row in snap["generations"]) != 1
               for key in CHECK_IDS):
            raise ValueError(f"Epoch {epoch} is missing a monitored validation output")
    return report


def run_check(monitored, run_dir, prior, output_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("Select a Kaggle T4 GPU for inference")
    if not load_hf_token():
        raise RuntimeError("Enable the Kaggle Secret named HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
        ),
    )
    model = PeftModel.from_pretrained(base, str(run_dir / "adapter_epoch_2"),
                                    adapter_name="epoch2", is_trainable=False)
    model.load_adapter(str(run_dir / "adapter_epoch_4"), adapter_name="epoch4",
                       is_trainable=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for epoch in CHECK_EPOCHS:
        model.set_adapter(f"epoch{epoch}")
        model.eval()
        for key in CHECK_IDS:
            row = monitored[key]
            encoded = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to("cuda:0")
            target_tokens = len(tokenizer.encode(row["completion"], add_special_tokens=False))
            with torch.inference_mode():
                tokens = model.generate(
                    input_ids, max_new_tokens=min(512, max(128, target_tokens + 64)),
                    do_sample=False, use_cache=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            output = tokenizer.decode(tokens[0, input_ids.shape[-1]:], skip_special_tokens=True).strip()
            snap = next(item for item in prior["snapshots"] if item["epoch"] == epoch)
            previous = next(item["output"] for item in snap["generations"]
                            if item["source_id"] == key)
            result = {
                "epoch": epoch, "source_id": key, "source": row["plain_english"],
                "reference": row["completion"], "output": output,
                "during_training_output": previous,
                "matches_during_training": normalized(output) == normalized(previous),
                "copied_source": normalized(output) == normalized(row["plain_english"]),
            }
            results.append(result)
            output_path.write_text(json.dumps({"model_id": MODEL_ID, "training_called": False,
                                               "test_rows_used": 0, "results": results},
                                              indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"Epoch {epoch}, {key}: matches earlier output={result['matches_during_training']}; "
                  f"copied source={result['copied_source']}", flush=True)
    print(f"Saved standalone inference check to {output_path}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default="/kaggle/working/epoch_adapter_check.json")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    _, monitored = load_pilot(args.dataset)
    prior = validate_run(args.run_dir, require_weights=not args.check)
    for epoch in CHECK_EPOCHS:
        snap = next(item for item in prior["snapshots"] if item["epoch"] == epoch)
        for key in CHECK_IDS:
            record = next(item for item in snap["generations"] if item["source_id"] == key)
            if (record["source"] != monitored[key]["plain_english"]
                    or record["reference"] != monitored[key]["completion"]):
                raise ValueError("Attached dataset differs from the saved run's validation examples")
    print("Validated epoch 2/4 adapters and validation rows p008,p050")
    if args.check:
        print("Dry-run OK. No model downloaded or trained.")
        return
    run_check(monitored, Path(args.run_dir), prior, Path(args.output))


if __name__ == "__main__":
    main()
