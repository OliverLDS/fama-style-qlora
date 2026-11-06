"""Parse mixed Pangram outcomes without converting displayed shares to probabilities."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('corpus_pangram', ROOT / 'scripts/data/import_corpus_200_pangram.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class PangramPromptComparisonTest(unittest.TestCase):
    def page(self, summary, shares=('0.1', '51', '49')):
        return ('AI Detection Report\nNov 4, 2026 99 Words Pangram 4.1\n' + summary +
                '\n' + '\n'.join(x + ' %\nof the document' for x in shares) +
                '\nAI Generated\nAI Assisted\nHuman Written\n')

    def test_mixed_outcome_is_retained_and_display_floor_not_rescaled(self):
        parsed = module.parse(self.page('We believe that this text is a mix of AI-assisted and human-written content.'))
        self.assertFalse(parsed['entirely_human'])
        self.assertEqual(parsed['provider_label'], 'Mix: AI-assisted / human')
        self.assertEqual(parsed['human_written_document_percent'], 49)
        self.assertAlmostEqual(parsed['displayed_percentage_sum'], 100.1)
        self.assertFalse(parsed['percentages_rescaled'])
        self.assertEqual(parsed['provider_version'], '4.1')
        self.assertEqual(parsed['report_date'], '2026-11-04')

    def test_only_native_human_summary_counts_as_entirely_human(self):
        for summary in module.SUMMARIES:
            parsed = module.parse(self.page(summary))
            self.assertEqual(parsed['entirely_human'], summary == 'We believe that this entire text is human-written.')

    def test_unknown_layout_or_invalid_percentage_fails_without_guessing(self):
        with self.assertRaises(ValueError):
            module.parse(self.page('We believe something else.'))
        with self.assertRaises(ValueError):
            module.parse(self.page('We believe that this entire text is AI.', ('101', '0', '0')))


if __name__ == '__main__':
    unittest.main()
