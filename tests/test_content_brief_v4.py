import copy
import json
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fama_style.research.content_brief_v4 import validate_brief_v4, canonicalize_note_order, reconstruction_conditions
from fama_style.research.content_brief_v3 import RELATION_SPEC


class ContentBriefV4Tests(unittest.TestCase):
    def example(self):
        return dict(id='example', target_sha256='a'*64, status='ready', notes=[], brief=dict(
            topic='Financing and investment',
            purpose=dict(primary='explain_mechanism', secondary=[]),
            key_points=['Project investment: feasible only if @A1 available', 'External finance: @A1 required'],
            anchors=dict(A1='Collateral; borrowing prerequisite'),
            relationships=[dict(from_point='K2', to_point='K1', type='necessary_condition', qualification='Not sufficient')]))

    def test_compact_record_preserves_fields(self):
        row = self.example()
        before = copy.deepcopy(row)
        self.assertEqual(validate_brief_v4(row), before)
        self.assertEqual(row, before)

    def test_missing_and_nested_anchors_rejected(self):
        for modification in ('missing', 'nested'):
            row = self.example()
            if modification == 'missing':
                row['brief']['key_points'][0] += ' @A9'
            else:
                row['brief']['anchors']['A1'] = '@A2'
            with self.assertRaises(ValueError):
                validate_brief_v4(row)

    def test_redundant_legacy_fields_not_silently_accepted(self):
        row = self.example()
        row['brief']['task'] = 'Duplicate orientation'
        with self.assertRaises(ValueError):
            validate_brief_v4(row)
        row = self.example()
        row['brief']['purpose']['reader_takeaway'] = 'Repeated account'
        with self.assertRaises(ValueError):
            validate_brief_v4(row)

    def test_note_sort_keeps_direction_attached_to_content(self):
        row = self.example()
        before = copy.deepcopy(row)
        normalized = canonicalize_note_order(row)
        self.assertEqual(normalized['brief']['key_points'], list(reversed(row['brief']['key_points'])))
        edge = normalized['brief']['relationships'][0]
        self.assertEqual((edge['from_point'], edge['to_point'], edge['type']), ('K1', 'K2', 'necessary_condition'))
        self.assertEqual(normalized['brief']['anchors'], row['brief']['anchors'])
        self.assertEqual(canonicalize_note_order(normalized), normalized)
        self.assertEqual(row, before)

    def test_conditions_share_facts_and_independent_copies(self):
        row = self.example()
        variants = reconstruction_conditions(row)
        a, b = variants['content_only'], variants['purpose_and_relations']
        self.assertEqual(set(a['brief']), {'topic', 'key_points', 'anchors'})
        for key in a['brief']:
            self.assertEqual(a['brief'][key], b['brief'][key])
        a['brief']['anchors']['A1'] = 'Changed'
        self.assertNotEqual(a['brief']['anchors'], b['brief']['anchors'])
        self.assertEqual(row['brief']['anchors']['A1'], 'Collateral; borrowing prerequisite')

    def test_schema_and_prompt_match_contract(self):
        docs = ROOT / 'docs/data_scaling_v1'
        schema = json.loads((docs / 'brief_v4.schema.json').read_text())
        props = schema['properties']['brief']['properties']
        self.assertEqual(set(props), {'topic', 'purpose', 'key_points', 'anchors', 'relationships'})
        self.assertEqual(set(props['purpose']['properties']), {'primary', 'secondary'})
        self.assertEqual(props['anchors']['type'], 'object')
        self.assertEqual(set(props['relationships']['items']['properties']['type']['enum']), set(RELATION_SPEC))
        prompt = (docs / 'BRIEF_ABSTRACTION_V4.md').read_text()
        self.assertNotIn('hope', prompt.lower())
        for label in RELATION_SPEC:
            self.assertIn('`' + label + '`', prompt)


if __name__ == '__main__':
    unittest.main()
