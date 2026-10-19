"""Build an offline source/base/pilot review and analyze completed human ratings."""

import argparse
import hashlib
import itertools
import json
import random
import statistics
from pathlib import Path

from fama_style.evaluation import corpus_baseline as baseline
from fama_style.training import corpus_pilot as pilot
from fama_style.training import corpus_smoke as smoke

ROOT = Path(__file__).resolve().parents[3]
COMPARISON_SHA = "a0704f8ec39897d46323d23ec58db5ed1ba6ee20ad7cc22270bbd7f10e1b5846"
LABELS = ("unchanged_input", "base", "epoch_1")
CRITERIA = {"claims": 4, "economic_precision": 2, "analytical_structure": 2, "concision": 2}
RUBRIC = {
    "claims": "Against the input: 4 no observed claim change; 3 minor precision/qualification change; "
              "2 substantive omission/addition/change; 1 multiple major errors; 0 contradiction or off-task. "
              "Check actors, attribution, quantities, conditions, uncertainty and causal direction.",
    "economic_precision": "Against the reference: 2 precise economic terms/distinctions; 1 broadly suitable "
                          "but generic or imprecise; 0 misleading or unclear. Do not reward ornate vocabulary.",
    "analytical_structure": "Against the reference: 2 clear hypotheses, contrasts, conditions and inference "
                            "structure resembling its exposition; 1 coherent but weak alignment; 0 confusing/unsupported structure.",
    "concision": "Against the reference: 2 direct, economical exposition; 1 noticeable redundancy or awkwardness; "
                 "0 substantial padding, fragmentation or obscured relationships. Shorter alone is not better.",
}


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def normalized(text):
    return " ".join(text.split())


def load_comparison(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != COMPARISON_SHA:
        raise ValueError("Use the frozen corpus-v1 pilot version 1 comparison.json")
    groups, _, _, _ = smoke.load_inputs(ROOT / "data/fama_corpus_v1/curated.jsonl",
                                      ROOT / "experiments/corpus_v1_baseline/outputs/version_4")
    original = json.loads((ROOT / "experiments/corpus_v1_baseline/outputs/version_4/baseline.json").read_text())
    comparison = json.loads(raw)
    pilot.validate_comparison(comparison, groups["validation"], original)
    return comparison


def make_review(comparison):
    if len(comparison["results"]) != 21:
        raise ValueError("Expected exactly the frozen 21 validation paragraphs")
    rng = random.Random(20261022)
    # Each condition appears seven times in each slot, including the last 3 rows.
    assignments = list(itertools.permutations(LABELS))*3 + [LABELS, LABELS[1:]+LABELS[:1], LABELS[2:]+LABELS[:2]]
    rng.shuffle(assignments)
    records = list(comparison["results"])
    rng.shuffle(records)
    examples, key = [], {}
    for row, labels in zip(records, assignments):
        texts = {"unchanged_input": row["source"], "base": row["baseline_output"], "epoch_1": row["output"]}
        candidates = [{"slot": slot, "text": texts[label]} for slot, label in zip("ABC", labels)]
        examples.append({"id": row["id"], "source": row["source"], "reference": row["reference"],
                         "candidates": candidates})
        key[row["id"]] = dict(zip("ABC", labels))
    payload = {"schema": "source_control_review_v1", "criteria": CRITERIA, "rubric": RUBRIC, "examples": examples}
    review_id = digest_json({"payload": payload, "assignments": key})
    bundle = dict(payload, review_id=review_id)
    mapping = {"review_id": review_id, "assignments": key}
    scores = {"schema": "source_control_scores_v1", "review_id": review_id, "reviewer": "",
              "claim_inventories": {example["id"]: "" for example in examples},
              "ratings": [{"id": example["id"], "slot": candidate["slot"],
                           **{criterion: None for criterion in CRITERIA}, "claim_notes": "", "style_notes": ""}
                          for example in examples for candidate in example["candidates"]]}
    manifest = {"schema": "source_control_manifest_v1", "review_id": review_id,
                "comparison_sha256": COMPARISON_SHA, "bundle_sha256": digest_json(bundle),
                "assignment_key_sha256": digest_json(mapping), "paragraphs": 21, "candidates": 63,
                "test_rows_used": 0, "ratings_prefilled": False}
    return bundle, mapping, scores, manifest


def validate_bundle(bundle, mapping, manifest):
    if (manifest["bundle_sha256"] != digest_json(bundle)
            or manifest["assignment_key_sha256"] != digest_json(mapping)
            or not bundle["review_id"] == mapping["review_id"] == manifest["review_id"]):
        raise ValueError("Review bundle or candidate mapping changed")


def analyze(bundle, mapping, scores):
    if scores.get("review_id") != bundle["review_id"] or scores.get("schema") != "source_control_scores_v1":
        raise ValueError("Scores belong to a different review")
    expected = {(example["id"], candidate["slot"]) for example in bundle["examples"] for candidate in example["candidates"]}
    ratings = scores.get("ratings", [])
    if len(ratings) != 63 or {(r["id"], r["slot"]) for r in ratings} != expected:
        raise ValueError("Scores must contain each of the 63 candidates exactly once")
    by_slot = {(r["id"], r["slot"]): r for r in ratings}
    by_model = {label: {} for label in LABELS}
    for row in ratings:
        for criterion, maximum in CRITERIA.items():
            if type(row.get(criterion)) is not int or not 0 <= row[criterion] <= maximum:
                raise ValueError("Finish all claim and style ratings before unmasking: " + row["id"])
        if any(not isinstance(row.get(field), str) for field in ("claim_notes", "style_notes")):
            raise ValueError("Review notes must be text")
        if row["claims"] < 4 and not row["claim_notes"].strip():
            raise ValueError("Explain the claim change for " + row["id"] + " " + row["slot"])
        label = mapping["assignments"][row["id"]][row["slot"]]
        by_model[label][row["id"]] = row
    identical_pairs = 0
    for example in bundle["examples"]:
        for left, right in itertools.combinations(example["candidates"], 2):
            if normalized(left["text"]) == normalized(right["text"]):
                identical_pairs += 1
                a, b = by_slot[(example["id"], left["slot"])], by_slot[(example["id"], right["slot"])]
                if any(a[criterion] != b[criterion] for criterion in CRITERIA):
                    raise ValueError("Identical candidate texts must have tied ratings: " + example["id"])
    means = {label: {criterion: statistics.mean(row[criterion] for row in rows.values()) for criterion in CRITERIA}
             for label, rows in by_model.items()}
    pairwise = {}
    for control in ("unchanged_input", "base"):
        pairwise["epoch_1_vs_"+control] = {}
        for criterion in CRITERIA:
            counts = {"wins": 0, "ties": 0, "losses": 0}
            for paragraph_id, row in by_model["epoch_1"].items():
                delta = row[criterion]-by_model[control][paragraph_id][criterion]
                counts["wins" if delta > 0 else "losses" if delta < 0 else "ties"] += 1
            pairwise["epoch_1_vs_"+control][criterion] = counts
    return {"review_id": bundle["review_id"], "reviewer": scores.get("reviewer", ""),
            "paragraphs": 21, "ratings": 63, "identical_candidate_pairs": identical_pairs,
            "means": means, "pairwise": pairwise, "test_rows_used": 0,
            "limits": "Development validation from one paper; label masking is not independent blinding. "
                      "Claim and style scores are separate; no combined quality score or statistical significance is inferred."}


def write_summary(directory, summary):
    baseline.write_json(directory / "summary.json", summary)
    lines = ["# Source-control review results", "", "All 63 candidate ratings are complete.", "",
             "| Condition | Claims (0–4) | Economic precision (0–2) | Analytical structure (0–2) | Concision (0–2) |",
             "|---|---:|---:|---:|---:|"]
    for label in LABELS:
        lines.append("| "+label+" | "+" | ".join(f"{summary['means'][label][criterion]:.2f}" for criterion in CRITERIA)+" |")
    lines += ["", "## Paired epoch comparisons", "", "| Comparison | Criterion | Wins | Ties | Losses |",
              "|---|---|---:|---:|---:|"]
    for comparison, criteria in summary["pairwise"].items():
        for criterion, counts in criteria.items():
            lines.append(f"| {comparison} | {criterion} | {counts['wins']} | {counts['ties']} | {counts['losses']} |")
    lines += ["", summary["limits"], "", "Do not continue training based on loss alone. Inspect claim changes and "
              "whether the epoch output improves style relative to unchanged input, with no fidelity loss."]
    (directory / "summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def build(comparison_path, directory):
    comparison = load_comparison(comparison_path)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Refusing to overwrite a review; choose an empty output directory")
    directory.mkdir(parents=True, exist_ok=True)
    bundle, mapping, scores, manifest = make_review(comparison)
    for name, value in (("review_bundle.json", bundle), ("assignment_key.json", mapping),
                        ("blank_scores.json", scores), ("manifest.json", manifest)):
        baseline.write_json(directory / name, value)
    template = Path(__file__).with_name("templates") / "source_control_review.html"
    payload = json.dumps({"bundle": bundle, "blankScores": scores}, ensure_ascii=False).replace("<", "\\u003c")
    (directory / "review.html").write_text(template.read_text().replace("__REVIEW_PAYLOAD__", payload), encoding="utf-8")
    lines = ["# Three-way source-control worksheet", "", "Score the input, not model identity, for claim preservation.",
             "Score style against the original reference. Exact copying is not automatically an error or style success.",
             "These labels are masked, not independently blinded; source copies and earlier familiarity can identify them.", ""]
    for criterion, rubric in RUBRIC.items():
        lines += ["- **"+criterion+"**: "+rubric]
    for example in bundle["examples"]:
        lines += ["", "## "+example["id"], "", "### Source", "", example["source"], "",
                  "### Reference", "", example["reference"], "", "Claim inventory:", ""]
        for candidate in example["candidates"]:
            lines += ["### Candidate "+candidate["slot"], "", candidate["text"], "",
                      "Claims (0–4): ___; economic precision (0–2): ___; analytical structure (0–2): ___; concision (0–2): ___",
                      "", "Claim evidence: ___", "", "Style evidence: ___", ""]
    (directory / "worksheet.md").write_text("\n".join(lines), encoding="utf-8")
    print("Built 21 paragraphs / 63 candidates. Ratings are blank. Open", directory / "review.html")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("build")
    create.add_argument("--comparison", type=Path, default=ROOT / "archive/corpus_v1_pilot_version_1/comparison.json")
    create.add_argument("--output-dir", type=Path, default=ROOT / "reviews/source_control_v1")
    evaluate = sub.add_parser("analyze")
    evaluate.add_argument("--review-dir", type=Path, default=ROOT / "reviews/source_control_v1")
    evaluate.add_argument("--scores", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        build(args.comparison, args.output_dir)
        return
    bundle = json.loads((args.review_dir / "review_bundle.json").read_text())
    mapping = json.loads((args.review_dir / "assignment_key.json").read_text())
    manifest = json.loads((args.review_dir / "manifest.json").read_text())
    validate_bundle(bundle, mapping, manifest)
    summary = analyze(bundle, mapping, json.loads(args.scores.read_text()))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Refusing to overwrite completed review results")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_summary(args.output_dir, summary)
    baseline.write_json(args.output_dir / "scores_used.json", json.loads(args.scores.read_text()))
    print("Analyzed complete review; separate claim/style results saved to", args.output_dir)


if __name__ == "__main__":
    main()
