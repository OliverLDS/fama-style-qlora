import tests  # Make src/ importable without installing ML dependencies.
import json
import tempfile
import unittest
from pathlib import Path

from fama_style.diagnostics.overfit_two import HELD_OUT_ID, TRAIN_IDS, load_examples


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "historical" / "fama_jensen_1983_pilot_instructed.jsonl"


class OverfitTwoTests(unittest.TestCase):
    @tests.requires_local_artifacts(DATASET)
    def test_selected_rows_are_disjoint_and_in_expected_splits(self):
        selected = load_examples(DATASET)
        self.assertEqual(set(selected), set(TRAIN_IDS) | {HELD_OUT_ID})
        self.assertTrue(all(selected[key]["split"] == "train" for key in TRAIN_IDS))
        self.assertEqual(selected[HELD_OUT_ID]["split"], "test")

    @tests.requires_local_artifacts(DATASET)
    def test_rejects_test_row_mislabeled_as_training(self):
        rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines()]
        next(row for row in rows if row["source_ids"] == [HELD_OUT_ID])["split"] = "train"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_examples(path)


if __name__ == "__main__":
    unittest.main()
