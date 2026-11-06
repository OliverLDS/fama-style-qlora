"""Check response bindings and local repairs without credentials, ML or GPUs."""
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('gpt6_quality', ROOT / 'scripts/data/audit_gpt6_track_500.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class ResponseQualityTests(unittest.TestCase):
    def setUp(self):
        self.expected = dict(id='P001', target_sha256='a'*64)
        self.row = dict(self.expected, plain_english='A complete neutral paragraph.', rewrite_status='simplified', review_notes=[])

    def test_rejects_changed_identity_hash_and_extra_fields(self):
        m.validate_response(self.row, self.expected)
        for change in [dict(id='P002'), dict(target_sha256='b'*64), dict(original_paragraph='Extra target')]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                m.validate_response(dict(self.row, **change), self.expected)

    def test_rejects_empty_sources_and_unexplained_uncertainty(self):
        for change in [dict(plain_english='  '), dict(review_notes='not an array'), dict(rewrite_status='pending_review')]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                m.validate_response(dict(self.row, **change), self.expected)
        m.validate_response(dict(self.row, rewrite_status='pending_review', review_notes=['Source has a contradictory statistic.']), self.expected)

    def test_local_patches_replay_and_reject_broad_or_nonunique_rewrites(self):
        source = 'The policy may reduce costs. The remaining claims stay in place.'
        patch = dict(before='may reduce', after='will reduce', reason='Restore the stated modality.')
        self.assertEqual(m.apply_patches(source, [patch]), source.replace('may reduce', 'will reduce'))
        for bad in [dict(patch, before='The'), dict(patch, before=source), dict(patch, before='missing'), dict(patch, reason='')]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                m.apply_patches(source, [bad])

    def test_lexical_quantity_and_length_flags_do_not_make_rejection_decisions(self):
        result = m.metrics('More than half of firms participate.', 'More than 50% of firms participate.')
        self.assertIn('numeric_surface_difference', result['flags'])
        self.assertNotIn('decision', result)
        self.assertTrue(m.metrics('Exactly the same words.', 'Exactly the same words.')['exact_normalized_copy'])


@unittest.skipUnless((m.OUT / 'quality_manifest.json').exists(), 'private response audit is not distributed')
class PrivateResponseQualityTests(unittest.TestCase):
    def test_candidate_targets_reuses_and_repairs(self):
        responses = m.check_import()
        raw = m.p.lines(m.p.ROOT / m.p.MASTER)
        candidates = m.p.lines(m.OUT / 'candidates_0500.private.jsonl')
        decisions = {d['id']:d for d in m.p.lines(m.OUT / 'adjudications.private.jsonl')}
        self.assertEqual(len(candidates), 500)
        self.assertEqual(len({r['parent_target_id'] for r in candidates}), 500)
        repairs = 0
        for i, (old, new) in enumerate(zip(raw, candidates), 1):
            rid = f'P{i:03d}'
            self.assertEqual(new['target'], old['target'])
            self.assertEqual(new['target_sha256'], old['target_sha256'])
            self.assertFalse(new['training_ready'])
            if rid not in responses:
                self.assertEqual(new['source'], old['source'])
            else:
                expected = m.apply_patches(responses[rid]['plain_english'], decisions[rid]['patches'])
                self.assertEqual(new['source'], expected)
                repairs += bool(decisions[rid]['patches'])
            self.assertEqual(m.p.sha(new['source']), new['source_sha256'])
        self.assertEqual(repairs, 10)
        self.assertEqual({d['id'] for d in decisions.values() if d['decision']=='target_review_pending'}, {'P250','P264','P291','P458','P484'})

    def test_raw_responses_and_quality_evidence_remain_bound(self):
        m.check()
        summary = m.p.read(m.OUT / 'summary.json')
        self.assertEqual(summary['new_responses'], 227)
        self.assertEqual(summary['unique_normalized_new_sources'], 227)
        self.assertEqual(summary['exact_normalized_raw_copies'], 0)
        self.assertFalse(summary['training_ready_release'])

    def test_source_evidence_supplement_is_bound_to_pdfs_and_quality_audit(self):
        manifest = m.p.read(m.OUT / 'source_evidence_manifest.json')
        for name, expected in manifest['files'].items():
            self.assertEqual(m.p.digest(ROOT / name), expected)
        self.assertEqual(manifest['pages'], 7)
        self.assertFalse(manifest['visual_independent_verification'])


if __name__ == '__main__':
    unittest.main()
