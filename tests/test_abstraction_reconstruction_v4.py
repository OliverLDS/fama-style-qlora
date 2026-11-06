import copy
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('reconstruction', Path(__file__).resolve().parents[1] / 'scripts/data/prepare_abstraction_reconstruction_v4.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class UnchangedReconstructionTests(unittest.TestCase):
    def example(self):
        return dict(id='example', target_sha256='a'*64, status='ready', notes=[], brief=dict(
            topic='Investment', purpose=dict(primary='report_result', secondary=['explain_mechanism']),
            key_points=['Z result: @A1', 'A constraint: funding needed'], anchors=dict(A1='10%; observed change'),
            relationships=[dict(from_point='K2', to_point='K1', type='necessary_condition', qualification='Condition only')]))

    def test_guided_brief_is_exactly_the_supplied_brief(self):
        row = self.example()
        before = copy.deepcopy(row)
        projected = m.project_unchanged(row)
        self.assertEqual(projected['purpose_and_relations']['brief'], row['brief'])
        self.assertEqual(projected['purpose_and_relations']['brief']['key_points'][0], 'Z result: @A1')
        self.assertEqual(row, before)
        self.assertIn('explain_mechanism', projected['purpose_and_relations']['brief']['purpose']['secondary'])

    def test_common_values_identical_and_independent(self):
        row = self.example()
        p = m.project_unchanged(row)
        a, b = p['content_only'], p['purpose_and_relations']
        self.assertEqual(set(a['brief']), {'topic', 'key_points', 'anchors'})
        for key, value in a['brief'].items():
            self.assertEqual(value, b['brief'][key])
        a['brief']['key_points'][0] = 'Changed'
        self.assertEqual(b['brief']['key_points'], row['brief']['key_points'])


if __name__ == '__main__':
    unittest.main()
