"""Reproduce the assistant's recorded semantic review of the 54 training pairs."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fama_style.evaluation.compare_few_shot import DEMO_IDS, INSTRUCTION, load_experiment, make_prompt, normalized

REVIEWED_SHA256 = "3fabac4ddfedb947c88bdbb3956f21f7d524f1f51504fbca6067efffd272a8ed"
FORMAT = "fama_style_few_shot_reviewed_v1"
# Each entry was compared with its original target. None means retain the source.
# Exclusions are explicit; this dictionary is a review record, not an automatic judge.
# Detailed paragraph rewrites are retained in a local-only review artifact.
REVIEW_DECISIONS_PATH = Path(__file__).resolve().parents[2] / "data/historical/reviewed_training_decisions.private.json"

HEADINGS = {
 "p030": ["1. Specific Knowledge and Diffusion of Decision Functions."],
 "p034": ["2. Diffuse Residual Claims and Delegation of Decision Control."],
 "p037": ["3. Decision Control in Nonprofits and Financial Mutuals."],
 "p038": ["4. Common General Features of Decision Control Systems.", "Decision hierarchies."],
 "p039": ["Mutual monitoring systems."], "p040": ["Boards of directors."],
 "p046": ["The stock market."], "p047": ["The market for takeovers."],
 "p048": ["Expert boards."], "p055": ["1. Mutual Monitoring, Specific Knowledge, and Restricted Residual Claims."],
 "p057": ["2. Large Professional Partnerships."], "p062": ["1. The Control Function of Redeemable Claims."],
 "p064": ["2. The Board of Directors."], "p067": ["1. Nonprofit Boards."],
 "p072": ["3. The Private University and Decision Systems in Complex Nonprofits."],
}


def build_review(input_path):
    reviews = json.loads(REVIEW_DECISIONS_PATH.read_text(encoding="utf-8"))
    raw = Path(input_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != REVIEWED_SHA256:
        raise ValueError("Dataset changed since this semantic review; review it again before rebuilding")
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    demos, _ = load_experiment(input_path)
    train_keys = {r["source_ids"][0] for r in rows if r["split"] == "train"}
    if train_keys != set(reviews):
        raise ValueError("Every training row must have exactly one explicit review decision")
    audited, output = [], []
    for original in rows:
        row = dict(original)
        if row["split"] == "test":
            output.append(row)  # Preserve the reserved test records exactly.
            continue
        if row["split"] == "validation":
            row.update(prompt=make_prompt(demos, row["plain_english"]), prompt_format=FORMAT,
                       parent_dataset_sha256=REVIEWED_SHA256)
            output.append(row)
            continue
        key = row["source_ids"][0]
        source, reason = reviews[key]
        target = row["completion"]
        for heading in HEADINGS.get(key, []):
            if target.count(heading) != 1:
                raise ValueError(f"Target heading changed for {key}")
            target = target.replace(heading, "")
        if key in ("p016", "p017"):
            target = target.removeprefix("1. ").removeprefix("2. ")
        if key == "p033":
            target = target.replace("as Well as", "as well as")
        target = normalized(target)
        source = row["plain_english"] if source is None else source
        role = ("excluded" if source == "EXCLUDE" else
                "reserved_demonstration" if key in DEMO_IDS else "training")
        audited.append({"source_ids": row["source_ids"], "role": role,
                        "reviewer": "assistant_semantic_review", "reason": reason,
                        "original_source": row["plain_english"],
                        "reviewed_source": None if role == "excluded" else source,
                        "original_target": row["completion"], "reviewed_target": target,
                        "source_changed": source != row["plain_english"],
                        "target_format_changed": target != row["completion"],
                        "target_content_policy": "original paragraph body; headings/numbers/whitespace and one visible capitalization artifact only"})
        if role != "training":
            continue
        row.update(plain_english=source, completion=target, prompt=make_prompt(demos, source),
                   prompt_format=FORMAT, parent_dataset_sha256=REVIEWED_SHA256)
        output.append(row)
    report = {"input_sha256": REVIEWED_SHA256,
              "review_type": "assistant judgment, not independent human verification or factual verification against the original article",
              "original_train_rows": 54, "training_rows": 48,
              "reserved_demonstration_ids": list(DEMO_IDS), "demonstrations": demos,
              "excluded_ids": [r["source_ids"] for r in audited if r["role"] == "excluded"],
              "validation_rows": 12, "test_rows": 12, "instruction": INSTRUCTION,
              "validation_policy": "source and target unchanged; prompt matches frozen few-shot baseline",
              "test_policy": "records preserved exactly and never used by smoke test",
              "decisions": audited}
    if len(audited) != 54 or len(output) != 72:
        raise ValueError("Unexpected reviewed split sizes")
    return output, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/historical/fama_jensen_1983_pilot_instructed.jsonl")
    parser.add_argument("--output", default="data/historical/fama_jensen_1983_pilot_reviewed.jsonl")
    parser.add_argument("--audit", default="data/historical/fama_training_alignment_audit.json")
    args = parser.parse_args()
    rows, report = build_review(args.input)
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    Path(args.output).write_text(text, encoding="utf-8")
    report["output_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    Path(args.audit).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Training-pair alignment audit", "",
             report["review_type"], "",
             "54 training rows reviewed: 48 fit rows, 3 reserved demonstrations, 3 exclusions.",
             "Validation source/target fields and all test records are unchanged.", "",
             "| IDs | Role | Review decision |", "| --- | --- | --- |"]
    lines.extend(f"| {','.join(r['source_ids'])} | {r['role']} | {r['reason']} |" for r in report["decisions"])
    lines.extend(["", "Full before/after sources and targets are in `data/historical/fama_training_alignment_audit.json`.",
                  "Targets retain the original prose body; section headings, list numbers and whitespace are normalized.",
                  "Input changes restore the target's meaning; they are not independent claims that the paper is factually correct.",
                  "Frozen demonstration examples remain exactly as used in the prior inference comparison."])
    (Path(__file__).resolve().parents[2] / "experiments/reviewed_4b/training_alignment_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Reviewed 54 training rows; wrote 48 train, 12 validation, 12 untouched test rows")


if __name__ == "__main__":
    main()
