"""Generate a side-by-side base/QLoRA comparison on untouched pilot test rows."""

import argparse
import json
import os
import re
from pathlib import Path

from fama_style.prompting.rewrite import REWRITE_INSTRUCTION, generation_prompt


MODEL_ID = "google/gemma-3-1b-it"




def validate_inputs(dataset_path, adapter_dir):
    with Path(dataset_path).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != 78 or {name: sum(row.get("split") == name for row in rows)
                            for name in ("train", "validation", "test")} != {
                                "train": 54, "validation": 12, "test": 12}:
        raise ValueError("Expected the cleaned 78-pair pilot dataset with 54/12/12 splits")
    test_rows = [row for row in rows if row["split"] == "test"]
    if any(not isinstance(row.get("prompt"), str) or not isinstance(row.get("completion"), str)
           or not row["prompt"].strip() or not row["completion"].strip()
           for row in test_rows):
        raise ValueError("Test rows need nonempty prompt and completion fields")
    adapter_dir = Path(adapter_dir)
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    if config.get("base_model_name_or_path") != MODEL_ID:
        raise ValueError(f"Adapter must be trained from {MODEL_ID}")
    if not (adapter_dir / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("Missing adapter_model.safetensors")
    if any(row.get("prompt_format") == "rewrite_v1" for row in rows):
        evaluation_path = adapter_dir / "evaluation.json"
        if not evaluation_path.is_file() or json.loads(evaluation_path.read_text(encoding="utf-8")).get("prompt_format") != "rewrite_v1":
            raise ValueError("Attach the adapter from the instructed pilot, not the version 7 bare-prompt pilot")
    return test_rows


def write_claim_review_template(results, path):
    """Provide a sentence-level worksheet; factual judgments remain manual."""
    lines = [
        "# Meaning and style review",
        "",
        "For each source sentence, mark whether each output preserves, changes, or omits its claim."
        " Then record any added claims. Sentences are review aids, not automatic claim labels.",
        "",
    ]
    for result in results:
        source = result.get("plain_english", result["prompt"])
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", source.strip())
        lines.extend([f"## {', '.join(result['source_ids'])}", "", "### Source claims", ""])
        for index, sentence in enumerate(sentences, 1):
            lines.extend([
                f"{index}. {sentence}",
                "   - Base: [ ] preserved  [ ] changed  [ ] omitted",
                "   - Fine-tuned: [ ] preserved  [ ] changed  [ ] omitted",
            ])
        lines.extend([
            "", "### Unsupported additions", "", "- Base: ", "- Fine-tuned: ",
            "", "### Style and clarity notes", "", "- Base: ", "- Fine-tuned: ",
            "", "### Outputs", "", "**Base**", "", result["base_output"],
            "", "**Fine-tuned**", "", result["fine_tuned_output"], "",
            "**Reference**", "", result["reference"], "",
        ])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def load_hf_token():
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        token = None
    if token:
        os.environ["HF_TOKEN"] = token
    return token


def run_comparison(test_rows, adapter_dir, output_path, max_new_tokens=512,
                   instruction=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("A Kaggle GPU is required for model generation")
    if not load_hf_token():
        raise RuntimeError("Enable the Kaggle Secret named HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base = AutoModelForCausalLM.from_pretrained(
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
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def generate(input_ids):
        with torch.inference_mode():
            tokens = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        return tokenizer.decode(tokens[0, input_ids.shape[-1]:], skip_special_tokens=True).strip()

    results = []
    with output_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(test_rows, 1):
            # Both models get identical tokens; instructed mode probes task ambiguity.
            prompt = generation_prompt(row.get("plain_english", row["prompt"]), instruction) if instruction else row["prompt"]
            encoded = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            input_ids = encoded["input_ids"].to("cuda:0")
            with model.disable_adapter():
                base_output = generate(input_ids)
            fine_tuned_output = generate(input_ids)
            result = {
                "source_ids": row["source_ids"],
                "prompt": row["prompt"],
                "plain_english": row.get("plain_english", row["prompt"]),
                "generation_prompt": prompt,
                "reference": row["completion"],
                "base_output": base_output,
                "fine_tuned_output": fine_tuned_output,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            results.append(result)
            print(f"Generated comparison {index}/{len(test_rows)}")
    print(f"Saved comparisons to {output_path}")
    review_path = output_path.with_name(output_path.stem + "_claim_review.md")
    write_claim_review_template(results, review_path)
    print(f"Saved claim-review worksheet to {review_path}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", default="/kaggle/working/fama_style_comparison.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--source-ids", help="comma-separated IDs of untouched test examples")
    parser.add_argument("--instruction", action="store_true",
                        help="prepend an explicit rewrite instruction for both models")
    parser.add_argument("--require-rewrite", action="store_true",
                        help="require that every dataset prompt already has the rewrite instruction")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    test_rows = validate_inputs(args.dataset, args.adapter)
    if args.require_rewrite and any(
        row.get("prompt_format") != "rewrite_v1" or
        row["prompt"] != generation_prompt(row.get("plain_english", ""), instruction=True)
        for row in test_rows
    ):
        raise ValueError("Test prompts must use the instructed rewrite_v1 format")
    if args.source_ids:
        selected = {part.strip() for part in args.source_ids.split(",") if part.strip()}
        test_rows = [row for row in test_rows if selected.intersection(row["source_ids"])]
        if {row_id for row in test_rows for row_id in row["source_ids"]} != selected:
            raise ValueError("source_ids must identify untouched test examples")
    print(f"Validated {len(test_rows)} untouched test pairs and Gemma adapter")
    if args.check:
        print("Dry-run OK. No model downloaded or generated.")
        return
    run_comparison(test_rows, args.adapter, args.output, args.max_new_tokens,
                   args.instruction)


if __name__ == "__main__":
    main()
