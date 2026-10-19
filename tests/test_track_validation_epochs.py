import tests  # Make src/ importable without installing ML dependencies.
import tempfile
import unittest
from pathlib import Path

from fama_style.diagnostics.track_validation_epochs import MONITORED_IDS, load_pilot, write_claim_review


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "historical" / "fama_jensen_1983_pilot_instructed.jsonl"


class TrackValidationEpochsTests(unittest.TestCase):
    @tests.requires_local_artifacts(DATASET)
    def test_monitor_is_fixed_validation_subset(self):
        groups, monitored = load_pilot(DATASET)
        self.assertEqual([len(groups[name]) for name in ("train", "validation", "test")],
                         [54, 12, 12])
        self.assertEqual(set(monitored), set(MONITORED_IDS))
        self.assertTrue(all(row["split"] == "validation" for row in monitored.values()))
        self.assertTrue(all(row not in groups["train"] + groups["test"]
                            for row in monitored.values()))

    @tests.requires_local_artifacts(DATASET)
    def test_claim_worksheet_leaves_judgments_for_human(self):
        _, monitored = load_pilot(DATASET)
        generations = [{"source_id": key, "source": row["plain_english"],
                        "reference": row["completion"], "output": "A diagnostic output."}
                       for key, row in monitored.items()]
        report = {"snapshots": [{"epoch": 0, "optimizer_step": 0,
                                 "generations": generations}]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "review.md"
            write_claim_review(report, path)
            review = path.read_text(encoding="utf-8")
        self.assertTrue(all(f"## {key}" in review for key in MONITORED_IDS))
        self.assertIn("[ ] preserved  [ ] changed  [ ] omitted", review)
        self.assertIn("Unsupported additions", review)
        self.assertNotIn("[x]", review)


if __name__ == "__main__":
    unittest.main()
