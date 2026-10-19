import tests  # Make src/ importable without installing ML dependencies.
import json
import tempfile
import unittest
from pathlib import Path
from argparse import Namespace
from fama_style.training.train import load_rows, split_rows, to_chat_rows, use_labeled_splits, validate_config


class DataTests(unittest.TestCase):
    def test_example_data(self):
        rows = load_rows()
        self.assertEqual(len(rows), 3)
        chat = to_chat_rows(rows)
        self.assertEqual(chat[0]["prompt"][0]["role"], "user")
        self.assertEqual(chat[0]["completion"][0]["role"], "assistant")

    def test_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text(json.dumps({"prompt": "P", "completion": "C"}) + "\n", encoding="utf-8")
            self.assertEqual(load_rows(path)[0]["prompt"], "P")

    def test_invalid_row(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.jsonl"
            path.write_text(json.dumps({"prompt": "P"}) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_rows(path)

    def test_split_is_reproducible_and_disjoint(self):
        rows = [{"prompt": str(i), "completion": str(i)} for i in range(10)]
        train, validation = split_rows(rows, 0.3, seed=7)
        self.assertEqual((len(train), len(validation)), (7, 3))
        self.assertEqual(set(map(id, train)) & set(map(id, validation)), set())
        self.assertEqual(split_rows(rows, 0.3, seed=7), (train, validation))

    def test_small_split_and_invalid_inputs(self):
        train, validation = split_rows(load_rows())
        self.assertEqual((len(train), len(validation)), (2, 1))
        with self.assertRaises(ValueError):
            split_rows(load_rows()[:1])
        with self.assertRaises(ValueError):
            split_rows(load_rows(), 1.0)

    def test_config_validation(self):
        args = Namespace(steps=None, epochs=1, lora_rank=8, learning_rate=2e-4,
                         max_length=1024,
                         validation_fraction=0.2)
        validate_config(args)
        for field, value in (("steps", 0), ("epochs", 0), ("lora_rank", 0),
                             ("learning_rate", 0), ("max_length", 0),
                             ("validation_fraction", 1)):
            bad = Namespace(**vars(args))
            setattr(bad, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_config(bad)

    def test_labeled_splits_leave_test_untouched(self):
        rows = [{"prompt": str(i), "completion": str(i), "split": name}
                for i, name in enumerate(("train", "validation", "test"))]
        groups = use_labeled_splits(rows)
        self.assertEqual([len(groups[name]) for name in ("train", "validation", "test")], [1, 1, 1])
        self.assertEqual(groups["test"][0]["prompt"], "2")
        self.assertIsNone(use_labeled_splits(load_rows()))
        with self.assertRaises(ValueError):
            use_labeled_splits(rows[:-1])

if __name__ == "__main__":
    unittest.main()
