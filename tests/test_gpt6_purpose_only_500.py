import copy
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('collection', Path(__file__).resolve().parents[1] / 'scripts/data/prepare_gpt6_purpose_only_500.py')
c = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c)


class PurposeOnlyCollectionTests(unittest.TestCase):
    def brief(self, rid='P001', target='Target 1'):
        return dict(id=rid, target_sha256=c.m.sha(target), status='ready', notes=[], brief=dict(
            topic='Financing and investment', purpose=dict(primary='report_result', secondary=['explain_mechanism']),
            key_points=['Investment: @A1', 'Financing: collateral requirement'], anchors=dict(A1='10%; investment growth'),
            relationships=[dict(from_point='K2', to_point='K1', type='necessary_condition', qualification='Conditional')]))

    def targets(self):
        return [dict(id=f'P{i:03d}', target=f'Target {i}', target_sha256=c.m.sha(f'Target {i}'), split='training')
                for i in range(1, 501)]

    def test_projection_preserves_purpose_and_excludes_relations(self):
        row = self.brief()
        before = copy.deepcopy(row)
        payload = c.purpose_only_payload(row)
        self.assertEqual(set(payload['brief']), {'topic', 'purpose', 'key_points', 'anchors'})
        self.assertEqual(payload['brief']['purpose'], row['brief']['purpose'])
        self.assertEqual(payload['brief']['key_points'], row['brief']['key_points'])
        self.assertEqual(row, before)
        payload['brief']['purpose']['secondary'].clear()
        self.assertEqual(row['brief']['purpose']['secondary'], ['explain_mechanism'])

    def test_five_reused_and_495_pending_are_complete_partition(self):
        targets = self.targets()
        reused = [self.brief(f'P{i:03d}', f'Target {i}') for i in range(1, 6)]
        queued = c.partition(targets, reused)
        self.assertEqual(len(queued), 495)
        self.assertEqual(queued[0]['id'], 'P006')
        self.assertEqual({r['id'] for r in queued} | {r['id'] for r in reused}, {r['id'] for r in targets})
        self.assertFalse({r['id'] for r in queued} & {r['id'] for r in reused})

    def test_incomplete_cohort_wrong_binding_and_test_split_rejected(self):
        targets = self.targets()
        reused = [self.brief(f'P{i:03d}', f'Target {i}') for i in range(1, 6)]
        with self.assertRaises(ValueError):
            c.partition(targets[:-1], reused)
        bad = copy.deepcopy(reused)
        bad[0]['target_sha256'] = 'a'*64
        with self.assertRaises(ValueError):
            c.partition(targets, bad)
        targets[0]['split'] = 'test'
        with self.assertRaises(ValueError):
            c.partition(targets, reused)


if __name__ == '__main__':
    unittest.main()
