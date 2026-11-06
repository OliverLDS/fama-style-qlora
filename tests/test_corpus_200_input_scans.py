"""The source controls must be the exact inputs of all four output conditions."""
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('input_scans', ROOT / 'scripts/data/prepare_corpus_200_input_scans.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class InputControlTest(unittest.TestCase):
    def fixture(self):
        rows = [dict(example_id=str(i), source=f'Exact input {i}.', reference='Never scan this reference.',
                     output='Never scan this output.', paper_id=str(i // 2), style_group=str(i // 2)) for i in range(8)]
        original = dict(config={'detector_case_ids': [r['example_id'] for r in rows]},
                        snapshots=[{'generations': copy.deepcopy(rows)} for _ in range(2)])
        simple = dict(previous_detector_case_ids=original['config']['detector_case_ids'],
                      snapshots=[{'generations': copy.deepcopy(rows)} for _ in range(2)])
        return original, simple

    def test_inputs_retained_verbatim_without_reference_or_output(self):
        original, simple = self.fixture()
        items = module.select(original, simple)
        self.assertEqual(len(items), 8)
        for item, row in zip(items, original['snapshots'][0]['generations']):
            self.assertEqual(item['text'], row['source'])
            self.assertEqual(item['text_sha256'], module.runtime.sha(row['source']))
            self.assertNotIn('reference', item)
            self.assertNotIn('output', item)
            self.assertFalse(item['provider_minimum_met'])

    def test_changed_condition_or_reselected_cases_rejected(self):
        original, simple = self.fixture()
        simple['snapshots'][1]['generations'][0]['source'] = 'A different source'
        with self.assertRaisesRegex(ValueError, 'input_changed'):
            module.select(original, simple)
        original, simple = self.fixture()
        simple['previous_detector_case_ids'] = list(reversed(simple['previous_detector_case_ids']))
        with self.assertRaisesRegex(ValueError, 'same_eight'):
            module.select(original, simple)


if __name__ == '__main__':
    unittest.main()
