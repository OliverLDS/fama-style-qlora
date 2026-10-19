import tests  # Make src/ importable without installing ML dependencies.
import unittest
from collections import UserDict

from fama_style.diagnostics.audit_label_preprocessing import audit_row


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, return_dict,
                            add_generation_prompt=False):
        text = "USER:" + messages[0]["content"] + "\nASSISTANT:"
        if len(messages) == 2:
            text += messages[1]["content"] + "<END>"
        return UserDict({"input_ids": [ord(character) for character in text]})

    def decode(self, ids, *, skip_special_tokens):
        return "".join(chr(token_id) for token_id in ids).replace("<END>", "")


class LabelPreprocessingTests(unittest.TestCase):
    def test_audit_uses_token_ids_and_matches_target(self):
        row = {
            "source_ids": ["example"],
            "split": "train",
            "plain_english": "plain input",
            "prompt": "Rewrite: plain input",
            "completion": "formal target",
        }
        result = audit_row(row, FakeTokenizer())
        self.assertGreater(result["full_tokens"], result["prompt_tokens"])
        self.assertFalse(result["token_prefix_mismatch"])
        self.assertTrue(result["source_in_masked_prefix"])
        self.assertTrue(result["reference_exact_after_whitespace_normalization"])
        self.assertFalse(result["source_exact_after_whitespace_normalization"])


if __name__ == "__main__":
    unittest.main()
