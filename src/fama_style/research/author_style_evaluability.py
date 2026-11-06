"""Metadata-only paper coverage audit. Never opens prose, PDFs or reserved rows."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


SOURCES = {
    "inventory": "docs/author_inventory_v2/inventory.json",
    "fama_splits": "data/fama_corpus_v1/split_manifest.json",
    "human_splits": "research_runs/human_prose_annotation_v1/package_v2/SPLIT_MANIFEST.json",
    "extraction": "data/author_style_corpus_v1/summary.json",
}


def audit(inventory, fama_splits, human_splits, extraction):
    """Only declared release roles establish exposure; absence is not freshness."""
    papers = {p["paper_id"]: p for p in inventory["papers"]}
    assignments = fama_splits["paper_assignments"]
    sealed = {p for p, a in assignments.items() if a["split"] == "test"}
    sealed.update(p["paper_id"] for p in human_splits["papers"]
                  if p["split"] == "reserved_test")
    annotated = {p["paper_id"]: p for p in human_splits["papers"]}
    hashes = defaultdict(list)
    titles = defaultdict(list)
    for p in papers.values():
        if p.get("pdf_sha256"):
            hashes[p["pdf_sha256"]].append(p["paper_id"])
        key = (" ".join(p["title"].lower().split()), tuple(p["authors"]))
        titles[key].append(p["paper_id"])
    groups = []
    for roster in inventory["roster"]:
        ids = roster["paper_ids"]
        if len(ids) != len(set(ids)) or any(p not in papers for p in ids):
            raise ValueError("Invalid roster IDs")
        counts = Counter()
        rows = []
        for pid in ids:
            p = papers[pid]
            a = assignments.get(pid)
            h = annotated.get(pid)
            train_pairs = a["selected_rows"] if a and a["split"] == "train" else 0
            # Selected human paragraph counts are annotation selections, not admitted pairs.
            annotated_rows = len(h["paragraph_ids"]) if h and pid not in sealed else 0
            counts["fama_training_pairs"] += train_pairs
            counts["fama_training_papers"] += bool(train_pairs)
            counts["human_annotation_selected_papers"] += bool(annotated_rows)
            counts["human_annotation_selected_paragraphs"] += annotated_rows
            counts["sealed_papers"] += pid in sealed
            counts["unassigned_papers"] += p.get("existing_split") == "unassigned_new_acquisition"
            rows.append({"paper_id": pid, "title": p["title"],
                         "authors": p["authors"], "version_year": p.get("version_year"),
                         "role": p.get("existing_split"), "sealed": pid in sealed,
                         "fama_training_pairs": train_pairs,
                         "human_annotation_selected_paragraphs": annotated_rows})
        groups.append({"style_group": roster["style_group"], "label": roster["label"],
                       "papers": len(ids), "counts": dict(counts), "paper_evidence": rows,
                       "additional_to_20": max(0, 20-len(ids)),
                       "additional_to_30": max(0, 30-len(ids))})
    return {"protocol": "author_style_evaluability_v1", "groups": groups,
            "sealed_paper_ids": sorted(sealed), "reserved_text_bodies_read": 0,
            "extraction_candidates": extraction["annotation_candidates"],
            "extraction_candidates_by_group": extraction["annotation_candidates_by_style_group"],
            "duplicate_pdf_hash_groups": [sorted(v) for v in hashes.values() if len(v)>1],
            "duplicate_title_author_groups": [sorted(v) for v in titles.values() if len(v)>1],
            "limits": ["Metadata roles are not independent semantic verification.",
                       "Annotation selections are not admitted training pairs.",
                       "No recorded role does not certify absence of earlier exposure.",
                       "Exact duplicate checks do not detect every draft/version alias.",
                       "Counts across releases must not be summed as unique training examples."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs, receipt = {}, {}
    for name, relative in SOURCES.items():
        raw = (args.root / relative).read_bytes()
        inputs[name] = json.loads(raw)
        receipt[relative] = hashlib.sha256(raw).hexdigest()
    result = audit(**inputs)
    result["metadata_input_sha256"] = receipt
    payload = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False)+"\n"
    if args.output.exists() and args.output.read_text() != payload:
        raise SystemExit("Output differs: use a new versioned path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
    print("Metadata audit complete; reserved paragraph bodies read: 0")


if __name__ == "__main__":
    main()
