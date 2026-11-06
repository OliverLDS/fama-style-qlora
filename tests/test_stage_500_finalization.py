"""The final export must replay small corrections without unbound text changes."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('final500', ROOT / 'scripts/data/finalize_stage_500.py')
final = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(final)
SPEC2 = importlib.util.spec_from_file_location('minimal500', ROOT / 'scripts/data/review_stage_500_minimal.py')
minimal = importlib.util.module_from_spec(SPEC2)
SPEC2.loader.exec_module(minimal)


class Final500Tests(unittest.TestCase):
    def decision(self):
        draft = {'plain_english': 'The median stayed at 5%.'}
        row = {'target': 'The mean was unchanged at 5%.'}
        source = 'The mean stayed at 5%.'
        decision = dict(draft_sha256=final.m.sha(draft['plain_english']),
                        target_sha256=final.m.sha(row['target']),
                        final_source=source, source_sha256=final.m.sha(source),
                        applied_patches=[dict(before='median', after='mean', occurrences=1)],
                        status='approved_automated_pilot', numeric_difference_explanation='')
        return decision, row, draft

    def test_replay_preserves_neutral_wording_outside_correction(self):
        d, r, source = self.decision()
        self.assertEqual(final.validate_decision(d, r, source), 'The mean stayed at 5%.')
        self.assertNotEqual(d['final_source'], r['target'])

    def test_unlogged_final_text_change_is_rejected(self):
        d, r, source = self.decision()
        d['final_source'] = r['target']
        d['source_sha256'] = final.m.sha(r['target'])
        with self.assertRaisesRegex(ValueError, 'replay_changed'):
            final.validate_decision(d, r, source)

    def test_changed_target_or_draft_requires_new_decision(self):
        for which in ['target', 'draft']:
            d, r, source = self.decision()
            if which == 'target':
                r['target'] += ' Additional condition.'
            else:
                source['plain_english'] += ' Added claim.'
            with self.assertRaisesRegex(ValueError, 'text_binding'):
                final.validate_decision(d, r, source)

    def test_number_difference_needs_explicit_adjudication(self):
        d, r, source = self.decision()
        r['target'] = 'The mean was unchanged at five percent.'
        d['target_sha256'] = final.m.sha(r['target'])
        with self.assertRaisesRegex(ValueError, 'numeric_difference'):
            final.validate_decision(d, r, source)
        d['numeric_difference_explanation'] = 'five percent and 5% are equivalent'
        self.assertEqual(final.validate_decision(d, r, source), d['final_source'])

    def test_reviewer_cannot_replace_entire_paragraph_or_guess_anchor(self):
        source = ' '.join(['neutral'] * 50)
        with self.assertRaisesRegex(ValueError, 'too_much_wording'):
            minimal.apply_patches(source, [dict(before=source, after='Restored target.')])
        with self.assertRaisesRegex(ValueError, 'anchor_not_unique'):
            minimal.apply_patches('x and x', [dict(before='x', after='y')])


if __name__ == '__main__':
    unittest.main()
