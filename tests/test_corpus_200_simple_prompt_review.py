"""Keep detector packets paired and exact, including unfavorable outcomes."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('simple_prompt_review', ROOT / 'scripts/data/review_corpus_200_simple_prompt.py')
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


class SimplePromptReviewTest(unittest.TestCase):
    def fixture(self):
        ids = [f'{g}{i}' for g in ['F', 'S', 'H', 'B'] for i in [0, 1]]
        rows = []
        for rid in ids:
            for snap in ['base', 'epoch_1']:
                text = rid + ' ' + snap + ' ' + ('Exact paragraph words. ' * 20)
                rows.append(dict(example_id=rid, snapshot=snap, style_group=rid[0],
                                 paper_id=rid[0] + 'paper', output=text,
                                 output_sha256=review.runtime.sha(text),
                                 assessment={'material_claim_change': True, 'notes': ['Adverse result retained.']}))
        return rows, ids

    def test_exact_text_blind_pairing_and_adverse_results_retained(self):
        rows, ids = self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'scans'
            summary = review.scans(rows, ids, out)
            self.assertEqual(summary['texts'], 16)
            self.assertEqual(summary['new_detector_calls'], 0)
            items = [json.loads(line) for line in (out / 'PRIVATE_KEY.jsonl').read_text().splitlines()]
            expected = {(r['example_id'], r['snapshot']): r for r in rows}
            for item in items:
                original = expected[item['example_id'], item['condition']]
                self.assertEqual(item['text'], original['output'])
                self.assertTrue(item['assessment']['material_claim_change'])
                self.assertEqual((out / 'texts' / (item['scan_id'] + '.txt')).read_text(), original['output'])
            for day in [1, 2]:
                selected = [r for r in items if r['round'] == day]
                self.assertEqual({r['style_group'] for r in selected}, set('FSHB'))
                self.assertTrue(all(sum(r['example_id'] == rid for r in selected) == 2
                                    for rid in {r['example_id'] for r in selected}))

    def test_mutated_text_rejected_and_short_text_never_padded(self):
        rows, ids = self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            rows[0]['output'] += ' mutation'
            with self.assertRaisesRegex(ValueError, 'scan_text_changed'):
                review.scans(rows, ids, Path(folder) / 'bad')
            rows[0]['output'] = 'Too short.'
            rows[0]['output_sha256'] = review.runtime.sha(rows[0]['output'])
            out = Path(folder) / 'short'
            summary = review.scans(rows, ids, out)
            self.assertEqual(summary['below_minimum'], 1)
            items = [json.loads(line) for line in (out / 'PRIVATE_KEY.jsonl').read_text().splitlines()]
            item = next(r for r in items if r['example_id'] == ids[0] and r['condition'] == 'base')
            self.assertEqual(item['text'], 'Too short.')
            self.assertFalse(item['provider_minimum_met'])


if __name__ == '__main__':
    unittest.main()
