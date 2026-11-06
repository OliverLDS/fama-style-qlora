import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fama_style.research.content_brief import validate_brief_v2


class ContentBriefTests(unittest.TestCase):
    def example(self):
        return dict(id='example', target_sha256='a'*64, status='ready', notes=[], brief=dict(
            task='Explain how a borrowing constraint affects investment.',
            purpose=dict(primary='explain_mechanism', secondary=['draw_implication'], reader_takeaway='A financing limit can block a profitable project.'),
            key_points=['Borrowing capacity is limited.', 'Some profitable projects cannot obtain funding.'],
            anchors=[], relationships=[dict(from_point='K1', to_point='K2', type='cause', qualification='Only when internal funds are insufficient.')]))

    def test_purpose_and_resolving_relations(self):
        row = self.example()
        self.assertEqual(validate_brief_v2(row), row)

    def test_dangling_self_and_unknown_relationships(self):
        for field, value in [('to_point', 'K3'), ('to_point', 'K1'), ('type', 'adjacency_means_causation')]:
            row = self.example()
            row['brief']['relationships'][0][field] = value
            with self.assertRaises(ValueError):
                validate_brief_v2(row)

    def test_legacy_notes_do_not_silently_become_v2(self):
        row = self.example()
        del row['brief']['purpose']
        with self.assertRaises(ValueError):
            validate_brief_v2(row)

    def test_duplicate_role_and_unexplained_concern(self):
        row = self.example()
        row['brief']['purpose']['secondary'] = ['explain_mechanism']
        with self.assertRaises(ValueError):
            validate_brief_v2(row)
        row = self.example()
        row['status'] = 'source_problem'
        with self.assertRaises(ValueError):
            validate_brief_v2(row)


if __name__ == '__main__':
    unittest.main()
