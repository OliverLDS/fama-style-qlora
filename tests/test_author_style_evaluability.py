import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fama_style.research.author_style_evaluability import SOURCES, audit


class CoverageTests(unittest.TestCase):
    def fixtures(self):
        return dict(
            inventory={"papers": [{"paper_id": "P", "title": "A", "authors": ["X"],
                                    "pdf_sha256": "abc", "existing_split": "train"}],
                       "roster": [{"paper_ids": ["P"], "style_group": "X", "label": "X"}]},
            fama_splits={"paper_assignments": {"P": {"split": "train", "selected_rows": 3}}},
            human_splits={"papers": [{"paper_id": "SEALED", "split": "reserved_test",
                                      "paragraph_ids": ["secret-id"]}]},
            extraction={"annotation_candidates": 9, "annotation_candidates_by_style_group": {"X": 9}})

    def test_roles_are_not_conflated(self):
        result = audit(**self.fixtures())
        counts = result["groups"][0]["counts"]
        self.assertEqual(counts["fama_training_pairs"], 3)
        self.assertEqual(counts["human_annotation_selected_paragraphs"], 0)
        self.assertEqual(result["sealed_paper_ids"], ["SEALED"])

    def test_reject_duplicate_roster(self):
        inputs = self.fixtures()
        inputs["inventory"]["roster"][0]["paper_ids"] *= 2
        with self.assertRaises(ValueError):
            audit(**inputs)

    def test_cli_metadata_only_and_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for key, relative in SOURCES.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(self.fixtures()[key]))
            # Invalid reserved prose cannot be parsed even accidentally by the audit.
            reserved = root / "data/fama_corpus_v1/test.jsonl"
            reserved.write_text("NOT JSON: RESERVED BODY MUST NOT BE OPENED")
            output = root / "packet/audit.json"
            cmd = [sys.executable, str(ROOT / "src/fama_style/research/author_style_evaluability.py"),
                   "--root", str(root), "--output", str(output)]
            subprocess.run(cmd, check=True, capture_output=True)
            first = output.read_bytes()
            subprocess.run(cmd, check=True, capture_output=True)
            self.assertEqual(first, output.read_bytes())
            self.assertEqual(json.loads(first)["reserved_text_bodies_read"], 0)


if __name__ == "__main__":
    unittest.main()
