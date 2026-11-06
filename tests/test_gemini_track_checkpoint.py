import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('gemini_checkpoint', ROOT / 'scripts/data/audit_gemini_track_checkpoint.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class CheckpointTests(unittest.TestCase):
    def test_patches_cannot_replay_against_a_different_draft(self):
        patch = dict(before='debt amount', after='leverage', occurrences=1)
        self.assertEqual(m.replay('Measure the debt amount net of cash.', [patch]), 'Measure the leverage net of cash.')
        with self.assertRaises(ValueError):
            m.replay('A different draft.', [patch])
        with self.assertRaises(ValueError):
            m.replay('debt amount and debt amount', [patch])

    @unittest.skipUnless((m.g.RUN / 'resume_20261105_v2/manifest.json').exists(), 'private checkpoint not distributed')
    def test_partial_checkpoint_preserves_previous_work_and_all_targets(self):
        result = m.audit()
        self.assertEqual(result['qualified_reviewed_inputs'], 240)
        self.assertEqual(result['accepted_reuses'], 214)
        self.assertEqual(result['reviews_pending'], 46)
        self.assertEqual(result['staged_input_qualified_total'], 454)
        self.assertEqual(result['repaired_reviewed_inputs'], 23)
        self.assertEqual(result['shared_target_followups'], 5)
        self.assertEqual(result['training_calls'], 0)


if __name__ == '__main__':
    unittest.main()
