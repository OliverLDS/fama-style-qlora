"""Audit Gemma chat-template labels without loading model weights."""

import argparse
import difflib
import json
import os
from pathlib import Path


MODEL_ID = "google/gemma-3-1b-it"
MAX_LENGTH = 1024


def load_rows(path):
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
    return rows


def load_hf_token():
    if os.environ.get("HF_TOKEN"):
        return True
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        token = None
    if token:
        os.environ["HF_TOKEN"] = token
    return bool(token)


def token_ids(encoded):
    ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if not isinstance(ids, list) or (ids and not isinstance(ids[0], int)):
        raise TypeError("Unexpected tokenizer output; expected a flat input_ids list")
    return ids


def common_prefix_length(left, right):
    count = 0
    for first, second in zip(left, right):
        if first != second:
            break
        count += 1
    return count


def normalized(text):
    return " ".join(text.split())


def audit_row(row, tokenizer):
    user = [{"role": "user", "content": row["prompt"]}]
    full = user + [{"role": "assistant", "content": row["completion"]}]
    prompt_ids = token_ids(tokenizer.apply_chat_template(
        user, tokenize=True, return_dict=True, add_generation_prompt=True))
    full_ids = token_ids(tokenizer.apply_chat_template(
        full, tokenize=True, return_dict=True))
    prefix = common_prefix_length(prompt_ids, full_ids)
    labels = [-100] * prefix + full_ids[prefix:]
    labels = labels[:MAX_LENGTH]
    supervised_ids = [value for value in labels if value != -100]
    supervised_text = normalized(tokenizer.decode(supervised_ids, skip_special_tokens=True))
    masked_prefix = normalized(tokenizer.decode(full_ids[:prefix], skip_special_tokens=True))
    reference = normalized(row["completion"])
    source = normalized(row["plain_english"])
    return {
        "source_ids": row["source_ids"],
        "split": row["split"],
        "prompt_tokens": len(prompt_ids),
        "full_tokens": len(full_ids),
        "truncated_at_1024": len(full_ids) > MAX_LENGTH,
        "token_prefix_mismatch": prefix < len(prompt_ids),
        "prefix_tokens_missing_from_full": len(prompt_ids) - prefix,
        "supervised_tokens": len(supervised_ids),
        "source_in_masked_prefix": source in masked_prefix,
        "reference_exact_after_whitespace_normalization": supervised_text == reference,
        "reference_similarity": round(difflib.SequenceMatcher(None, supervised_text, reference).ratio(), 4),
        "source_exact_after_whitespace_normalization": supervised_text == source,
    }


def run_audit(rows, output_path):
    import transformers
    from transformers import AutoTokenizer

    if not load_hf_token():
        raise RuntimeError("Enable the Kaggle Secret named HF_TOKEN")
    print("Loading Gemma tokenizer only; model weights are not downloaded", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    audited = [audit_row(row, tokenizer) for row in rows]
    summary = {
        "model_id": MODEL_ID,
        "transformers_version": transformers.__version__,
        "model_weights_loaded": False,
        "training_called": False,
        "max_length": MAX_LENGTH,
        "rows": audited,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved preprocessing audit to {output_path}")
    print(f"Truncated rows: {sum(item['truncated_at_1024'] for item in audited)}/78")
    print(f"Token-prefix mismatches: {sum(item['token_prefix_mismatch'] for item in audited)}/78")
    evaluated = [item for item in audited if item["split"] != "test"]
    print(f"Reference exact in supervised span: "
          f"{sum(item['reference_exact_after_whitespace_normalization'] for item in evaluated)}/66")
    print("p014 summary:", next(item for item in audited if item["source_ids"] == ["p014"]))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="/kaggle/working/label_preprocessing_audit.json")
    parser.add_argument("--check", action="store_true",
                        help="validate data without ML libraries, model weights, or a GPU")
    args = parser.parse_args(argv)
    rows = load_rows(args.dataset)
    print(f"Validated {len(rows)} instructed rows")
    if args.check:
        print("Dry-run OK. No model downloaded or trained.")
        return
    run_audit(rows, args.output)


if __name__ == "__main__":
    main()
