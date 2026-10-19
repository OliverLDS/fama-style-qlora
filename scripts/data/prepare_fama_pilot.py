"""Build clean prompt/completion pairs from the annotated Fama-Jensen JSONL."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from fama_style.evaluation.compare_style import generation_prompt


MERGE_IDS = ("p009", "p010", "p011", "p012", "p013")
DUPLICATE_IDS = {"p077", "p078"}  # Repeats of p016 and p017 in the summary.
TRANSCRIPTION_FIXES = {
    "univer- sities": "universities",
    "hy- pothesis": "hypothesis",
    "Cathol- icism": "Catholicism",
    "committees- and": "committees—and",
    "detemine": "determine",
}
TEST_IDS = {f"p{i:03d}" for start in (3, 21, 43, 69) for i in range(start, start + 3)}
VALIDATION_IDS = {f"p{i:03d}" for start in (6, 26, 50, 80) for i in range(start, start + 3)}


def build_pairs(rows):
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows) or not set(MERGE_IDS + tuple(DUPLICATE_IDS)).issubset(by_id):
        raise ValueError("Source IDs are missing or duplicated")
    for duplicate, original in (("p077", "p016"), ("p078", "p017")):
        if by_id[duplicate]["original_paragraph"].strip() != by_id[original]["original_paragraph"].strip():
            raise ValueError(f"{duplicate} is no longer a duplicate of {original}")

    merged = {
        "source_ids": list(MERGE_IDS),
        "section": by_id[MERGE_IDS[0]]["section"],
        "prompt": "\n".join(by_id[row_id]["plain_english"].strip() for row_id in MERGE_IDS),
        "completion": "\n".join(by_id[row_id]["original_paragraph"].strip() for row_id in MERGE_IDS),
    }
    pairs = []
    for row in rows:
        row_id = row["id"]
        if row_id == MERGE_IDS[0]:
            pairs.append(merged)
        elif row_id in MERGE_IDS or row_id in DUPLICATE_IDS:
            continue
        else:
            pairs.append({
                "source_ids": [row_id],
                "section": row["section"],
                "prompt": row["plain_english"].strip(),
                "completion": row["original_paragraph"].strip(),
            })

    for pair in pairs:
        for before, after in TRANSCRIPTION_FIXES.items():
            pair["completion"] = pair["completion"].replace(before, after)
        pair["plain_english"] = pair["prompt"]
        pair["prompt"] = generation_prompt(pair["plain_english"], instruction=True)
        pair["prompt_format"] = "rewrite_v1"
        source_ids = set(pair["source_ids"])
        if source_ids & TEST_IDS:
            pair["split"] = "test"
        elif source_ids & VALIDATION_IDS:
            pair["split"] = "validation"
        else:
            pair["split"] = "train"
    if [sum(pair["split"] == name for pair in pairs) for name in ("train", "validation", "test")] != [54, 12, 12]:
        raise ValueError("Unexpected pilot split sizes")
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="original paragraph_taxonomies.jsonl")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[2] / "data/historical/fama_jensen_1983_pilot_instructed.jsonl")
    args = parser.parse_args()
    with args.source.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    pairs = build_pairs(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"Wrote {len(pairs)} pairs to {args.output}")


if __name__ == "__main__":
    main()
