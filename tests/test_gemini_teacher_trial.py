"""CPU contracts for a separate eight-row, free-only data-generation trial."""
from collections import Counter
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('gemini_trial', ROOT / 'scripts/data/gemini_teacher_trial.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class GeminiTeacherTrialTest(unittest.TestCase):
    def fixture(self):
        return [dict(id=f'{g}-{i}', style_group=g, paper_id=f'{g}-paper-{i%10}',
                     split='training', training_ready=True, original_paragraph=f'Human target {g} {i}.',
                     plain_english=f'Previous source {g} {i}.')
                for g in ['Fama', 'Shiller', 'Holmstrom', 'Baker'] for i in range(50)]

    def test_selection_stable_across_order_and_no_previous_teacher_wording_in_prompt(self):
        rows = self.fixture(); before = repr(rows)
        selected = module.select(rows)
        self.assertEqual(selected, module.select(list(reversed(rows))))
        self.assertEqual(repr(rows), before)
        self.assertEqual(sorted(Counter(r['style_group'] for r in selected).values()), [2,2,2,2])
        self.assertEqual(len({r['paper_id'] for r in selected}), 8)
        for row in selected:
            self.assertEqual(row['prompt'], module.INSTRUCTION + row['original_paragraph'])
            self.assertNotIn('Previous source', row['prompt'])

    def test_nontraining_or_unapproved_rows_rejected(self):
        for field, value in [('split', 'reserved_test'), ('training_ready', False)]:
            rows = self.fixture(); rows[0][field] = value
            with self.assertRaises(ValueError): module.select(rows)


if __name__ == '__main__': unittest.main()
