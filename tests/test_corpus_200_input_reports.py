"""An already human-labeled source must never count as AI-to-human conversion."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('input_report_import', ROOT / 'scripts/data/import_corpus_200_input_reports.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class InputOutputTest(unittest.TestCase):
    def fixture(self, source_human=True):
        inputs = [dict(example_id=str(i), entirely_human=source_human,
                       human_written_document_percent=100.0 if source_human else .1) for i in range(8)]
        outputs = [dict(example_id=str(i), prompt=p, condition=c, entirely_human=i < 6,
                        human_written_document_percent=100.0 if i < 6 else .1)
                   for p in ['strict', 'simple'] for c in ['base', 'epoch_1'] for i in range(8)]
        return inputs, outputs

    def test_human_retention_is_not_conversion(self):
        result = module.aggregate(*self.fixture())['conditions']['strict_epoch_1']
        self.assertEqual(result['human_input_stays_human'], 6)
        self.assertEqual(result['human_input_becomes_AI_related'], 2)
        self.assertEqual(result['AI_related_inputs_available'], 0)
        self.assertEqual(result['AI_related_input_becomes_human'], 0)

    def test_actual_conversion_requires_nonhuman_input_and_complete_pairs(self):
        inputs, outputs = self.fixture(False)
        result = module.aggregate(inputs, outputs)['conditions']['strict_epoch_1']
        self.assertEqual(result['AI_related_input_becomes_human'], 6)
        self.assertEqual(result['AI_related_inputs_available'], 8)
        with self.assertRaises(ValueError): module.aggregate(inputs, outputs[:-1])


if __name__ == '__main__': unittest.main()
