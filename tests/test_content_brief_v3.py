import copy
import json
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fama_style.research.content_brief_v3 import RELATION_SPEC, validate_brief_v3, reconstruction_conditions


class ContentBriefV3Tests(unittest.TestCase):
    def example(self):
        return dict(id='example', target_sha256='a'*64, status='ready', notes=[], brief=dict(
            task='Explain the financing prerequisite.',
            purpose=dict(primary='explain_mechanism', secondary=[], reader_takeaway='Some investment requires finance.'),
            key_points=['Finance is required for this project.', 'The project can proceed only with financing.'],
            anchors=[], relationships=[dict(from_point='K1', to_point='K2', type='necessary_condition', qualification='Necessary, not sufficient.')]))

    def test_relation_sign_and_strength_are_not_rewritten(self):
        for label in ('necessary_condition', 'sufficient_condition', 'evidence_against', 'association', 'predicts'):
            row = self.example()
            row['brief']['relationships'][0]['type'] = label
            before = copy.deepcopy(row)
            self.assertEqual(validate_brief_v3(row), before)
            self.assertEqual(row, before)

    def test_unresolved_references_and_unsupported_labels_rejected(self):
        for field, value in [('to_point', 'K3'), ('from_point', 'K2'), ('type', 'condition'), ('type', 'certainly_causes')]:
            row = self.example()
            row['brief']['relationships'][0][field] = value
            with self.assertRaises(ValueError):
                validate_brief_v3(row)

    def test_ablation_holds_common_content_constant(self):
        row = self.example()
        before = copy.deepcopy(row)
        variants = reconstruction_conditions(row)
        a, b = variants['content_only'], variants['purpose_and_relations']
        for key in ('task', 'key_points', 'anchors'):
            self.assertEqual(a['brief'][key], b['brief'][key])
        self.assertNotIn('purpose', a['brief'])
        self.assertNotIn('relationships', a['brief'])
        self.assertEqual(b['brief']['relationships'], row['brief']['relationships'])
        self.assertEqual(set(a), {'id', 'target_sha256', 'brief', 'brief_status', 'brief_notes'})
        a['brief']['key_points'][0] = 'mutation'
        self.assertEqual(row, before)
        self.assertNotEqual(a['brief']['key_points'], b['brief']['key_points'])

    def test_prompt_schema_and_catalog_agree(self):
        docs = ROOT / 'docs/data_scaling_v1'
        schema = json.loads((docs / 'brief_v3.schema.json').read_text())
        labels = schema['properties']['brief']['properties']['relationships']['items']['properties']['type']['enum']
        catalog = json.loads((docs / 'brief_relations_v3.json').read_text())
        self.assertEqual(set(labels), set(RELATION_SPEC))
        self.assertEqual(set(labels), {r['label'] for r in catalog['relations']})
        prompt = (docs / 'BRIEF_EXTRACTION_V3.md').read_text()
        self.assertNotIn('hope', prompt.lower())
        for label in labels:
            self.assertIn('`' + label + '`', prompt)


if __name__ == '__main__':
    unittest.main()
