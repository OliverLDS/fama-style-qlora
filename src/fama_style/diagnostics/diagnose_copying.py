"""Small, inference-only check of copying in the instructed Gemma pilot."""

import argparse
import difflib
import json
import os
from pathlib import Path

from fama_style.evaluation.compare_style import MODEL_ID, generation_prompt, load_hf_token


EXAMPLE_IDS = ("p001", "p014", "p003", "p045")
EXPECTED_SPLITS = ("train", "train", "test", "test")
MAX_LENGTH = 1024


def select_examples(dataset_path, adapter_dir):
    with Path(dataset_path).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    counts = {split: sum(row.get("split") == split for row in rows)
              for split in ("train", "validation", "test")}
    if len(rows) != 78 or counts != {"train": 54, "validation": 12, "test": 12}:
        raise ValueError("Expected the instructed 78-pair dataset with 54/12/12 splits")
    if any(row.get("prompt_format") != "rewrite_v1" or
           row.get("prompt") != generation_prompt(row.get("plain_english", ""), instruction=True) or
           not isinstance(row.get("completion"), str) or not row["completion"].strip()
           for row in rows):
        raise ValueError("Dataset prompts and completions must use rewrite_v1")
    selected = []
    for source_id, split in zip(EXAMPLE_IDS, EXPECTED_SPLITS):
        matches = [row for row in rows if row.get("source_ids") == [source_id]
                   and row.get("split") == split]
        if len(matches) != 1:
            raise ValueError(f"Expected one {split} row with source ID {source_id}")
        selected.append(matches[0])
    adapter_dir = Path(adapter_dir)
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    evaluation = json.loads((adapter_dir / "evaluation.json").read_text(encoding="utf-8"))
    if config.get("base_model_name_or_path") != MODEL_ID or evaluation.get("model_id") != MODEL_ID:
        raise ValueError(f"Adapter must be trained from {MODEL_ID}")
    if evaluation.get("prompt_format") != "rewrite_v1":
        raise ValueError("Attach the version 9 instructed adapter")
    if not (adapter_dir / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("Missing adapter_model.safetensors")
    return rows, selected


def token_audit(rows, tokenizer):
    audit = []
    for row in rows:
        user = [{"role": "user", "content": row["prompt"]}]
        conversation = user + [{"role": "assistant", "content": row["completion"]}]
        prompt_ids = tokenizer.apply_chat_template(
            user, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )["input_ids"][0].tolist()
        full_ids = tokenizer.apply_chat_template(
            conversation, tokenize=True, return_dict=True, return_tensors="pt",
        )["input_ids"][0].tolist()
        prompt_rendered = tokenizer.apply_chat_template(user, tokenize=False,
                                                        add_generation_prompt=True)
        rendered = tokenizer.apply_chat_template(conversation, tokenize=False)
        prompt_is_prefix = full_ids[:len(prompt_ids)] == prompt_ids
        audit.append({
            "source_ids": row["source_ids"],
            "split": row["split"],
            "prompt_tokens": len(prompt_ids),
            "full_tokens": len(full_ids),
            "prompt_text_is_prefix": rendered.startswith(prompt_rendered),
            "prompt_is_prefix": prompt_is_prefix,
            "completion_tokens": len(full_ids) - len(prompt_ids) if prompt_is_prefix else None,
            "over_1024_tokens": len(full_ids) > MAX_LENGTH,
            "target_in_rendered_assistant": row["completion"] in rendered,
        })
    return audit


def copy_stats(output, source):
    return {
        "exact_copy": output == source,
        "similarity_to_input": round(difflib.SequenceMatcher(None, output, source).ratio(), 4),
    }


def run_diagnostic(rows, selected, adapter_dir, output_dir):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("A Kaggle GPU is required for inference")
    if not load_hf_token():
        raise RuntimeError("Enable the Kaggle Secret named HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    audit = token_audit(rows, tokenizer)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "copy_token_audit.json").write_text(
        json.dumps({"max_length": MAX_LENGTH, "rows": audit}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Token audit saved: {len(audit)} rows; "
          f"{sum(item['over_1024_tokens'] for item in audit)} over {MAX_LENGTH} tokens; "
          f"{sum(not item['prompt_text_is_prefix'] for item in audit)} text-prefix mismatches; "
          f"{sum(not item['prompt_is_prefix'] for item in audit)} token-prefix mismatches; "
          f"{sum(not item['target_in_rendered_assistant'] for item in audit)} target mismatches")
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

    def generate(encoded, *, sample=False):
        options = {"max_new_tokens": 256, "do_sample": sample,
                   "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id}
        if sample:
            options.update(temperature=0.7, top_p=0.9)
        with torch.inference_mode():
            tokens = model.generate(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                **options,
            )
        return tokenizer.decode(tokens[0, encoded["input_ids"].shape[-1]:],
                                skip_special_tokens=True).strip()

    results = []
    for index, row in enumerate(selected, 1):
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
        with model.disable_adapter():
            base_greedy = generate(encoded)
        adapted_greedy = generate(encoded)
        torch.manual_seed(42 + index)
        adapted_sampled = generate(encoded, sample=True)
        results.append({
            "source_ids": row["source_ids"],
            "split": row["split"],
            "plain_english": row["plain_english"],
            "reference": row["completion"],
            "base_greedy": base_greedy,
            "adapted_greedy": adapted_greedy,
            "adapted_sampled": adapted_sampled,
            "base_greedy_copy": copy_stats(base_greedy, row["plain_english"]),
            "adapted_greedy_copy": copy_stats(adapted_greedy, row["plain_english"]),
            "adapted_sampled_copy": copy_stats(adapted_sampled, row["plain_english"]),
        })
        print(f"Generated diagnostic {index}/{len(selected)}: {row['split']} {row['source_ids'][0]}")

    with (output_dir / "copy_diagnostic.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} comparisons and {len(audit)} token audits to {output_dir}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", default="/kaggle/working")
    parser.add_argument("--check", action="store_true", help="validate inputs without a GPU or model download")
    args = parser.parse_args(argv)
    rows, selected = select_examples(args.dataset, args.adapter)
    print(f"Validated {len(rows)} pilot rows and {len(selected)} diagnostic examples")
    if args.check:
        print("Dry-run OK. No model downloaded or generated.")
        return
    run_diagnostic(rows, selected, args.adapter, args.output_dir)


if __name__ == "__main__":
    main()
