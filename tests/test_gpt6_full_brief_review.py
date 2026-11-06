import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('full_brief_review', Path(__file__).resolve().parents[1] / 'scripts/data/review_gpt6_full_briefs.py')
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


class FullBriefReviewTests(unittest.TestCase):
    def test_payload_allowlist_drops_original_and_old_rewrite(self):
        effective = dict(id='a', target_sha256='h', brief=dict(task='Explain', key_points=['content'], anchors=[]),
                         status='ready', notes=[], original_paragraph='PRIVATE', old_source='OLD')
        payload = r.payload_for(effective)
        self.assertNotIn('original_paragraph', payload)
        self.assertNotIn('old_source', payload)
        r.validate_payload(payload, effective)
        with self.assertRaises(ValueError):
            r.validate_payload(dict(payload, original_paragraph='PRIVATE'), effective)
        with self.assertRaises(ValueError):
            r.validate_payload(dict(payload, target_sha256='other'), effective)

    def test_subtraction_repair_is_exact_and_replayable(self):
        before = 'Cumulative wealth ratios can replace compounded benchmark-minus-event differences to lessen interpretive distortion'
        row = dict(id='P257', brief=dict(key_points=[before]), status='ready', notes=[])
        edits = r.edits_for(row)
        final = r.b.apply_edits(row, edits)
        self.assertNotIn('benchmark-minus-event', final['brief']['key_points'][0])
        self.assertEqual(row['brief']['key_points'][0], before)
        with self.assertRaises(ValueError):
            r.b.apply_edits(final, edits)

    def test_year_count_is_scoped_without_inventing_correction(self):
        note = 'Source labels 1984–1990 as a six-year span; inclusive calendar years would number seven, so exact span is ambiguous.'
        row = dict(id='P495', brief=dict(key_points=['Retirements']), status='source_problem', notes=[note])
        final = r.b.apply_edits(row, r.edits_for(row))
        self.assertEqual(final['status'], 'ready')
        self.assertEqual(final['brief'], row['brief'])
        self.assertIn('do not add a count', final['notes'][0])


if __name__ == '__main__':
    unittest.main()
