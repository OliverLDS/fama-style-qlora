import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('full_output_review', Path(__file__).resolve().parents[1] / 'scripts/data/review_gpt6_full_outputs.py')
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


class FullOutputReviewTests(unittest.TestCase):
    def test_repairs_bind_source_id_and_target(self):
        row = dict(id='P001', target_sha256='target', plain_english='Alpha beta gamma delta epsilon zeta.')
        d = dict(id='P001', target_sha256='target', raw_source_sha256=r.m.sha(row['plain_english']),
                 decision='repair', reason='Correct scope.', patches=[dict(before='beta', after='theta', reason='Correct scope.')])
        self.assertEqual(r.repair(row, d), 'Alpha theta gamma delta epsilon zeta.')
        for key, value in [('id', 'other'), ('target_sha256', 'other'), ('raw_source_sha256', 'other')]:
            with self.assertRaises(ValueError):
                r.repair(row, dict(d, **{key: value}))
        with self.assertRaises(ValueError):
            r.repair(dict(row, plain_english='Different text.'), d)

    def test_candidate_merge_preserves_target_and_parent(self):
        parent, targets = [], {}
        for i in range(1, 501):
            rid = f'P{i:03d}'
            target = f'Human target {i}'
            targets[rid] = dict(target=target)
            parent.append(dict(id=rid, source='prior', source_sha256=r.m.sha('prior'), target=target, target_sha256=r.m.sha(target),
                               training_ready=False, provenance={}, quality={}))
        before = copy.deepcopy(parent)
        final = dict(P001=dict(final_source='fresh', source_sha256=r.m.sha('fresh'), response_file='private.jsonl',
                               adjudication=dict(patches=[])))
        with patch.object(r.m, 'digest', return_value='hash'):
            merged = r.merged_candidates(parent, final, targets)
            self.assertEqual(parent, before)
            self.assertEqual(merged[0]['source'], 'fresh')
            self.assertEqual(merged[1]['source'], 'prior')
            self.assertEqual([x['target'] for x in merged], [x['target'] for x in parent])
            targets['P001']['target'] = 'Changed human'
            with self.assertRaises(ValueError):
                r.merged_candidates(parent, final, targets)

    def test_false_accept_with_patch_is_rejected(self):
        row = dict(id='a', target_sha256='h', plain_english='Fresh prose.')
        d = dict(id='a', target_sha256='h', raw_source_sha256=r.m.sha(row['plain_english']), decision='accept',
                 reason='retain', patches=[dict(before='Fresh', after='Other', reason='test')])
        with self.assertRaises(ValueError):
            r.repair(row, d)


if __name__ == '__main__':
    unittest.main()
