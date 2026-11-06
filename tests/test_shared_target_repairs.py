import copy
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('target_repairs', Path(__file__).resolve().parents[1] / 'scripts/data/repair_shared_targets_v2.py')
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)


class SharedTargetRepairTests(unittest.TestCase):
    def fixtures(self):
        parent, decisions = [], []
        for i in range(1, 501):
            rid = f'P{i:03d}'
            text = f'Target text {i}.'
            parent.append(dict(id=rid, target=text, target_sha256=v.m.sha(text), paper_id='training_paper',
                               original_paragraph=text, parent_target_id=f'original_{i}', parent_id=f'original_{i}',
                               training_ready=False, provenance=dict(matched_mixed_id=f'old_{i}'), quality={}))
            if rid in v.PENDING | {'P134', 'P139'}:
                after = text if rid == 'P139' else f'Corrected target {i}.'
                edits = [] if rid == 'P139' else [dict(before=text, after=after, reason='User decision')]
                d = dict(id=rid, paper_id='training_paper', before_target_sha256=v.m.sha(text),
                         after_target_sha256=v.m.sha(after), target_edits=edits, target_kind='test')
                if rid == 'P291':
                    d['replacement_paragraph_id'] = 'replacement_author_paragraph'
                decisions.append(d)
        return parent, decisions

    def test_six_target_changes_and_replacement_lineage(self):
        parent, decisions = self.fixtures()
        before = copy.deepcopy(parent)
        rows = v.target_versions(parent, decisions)
        self.assertEqual(parent, before)
        self.assertEqual(sum(a['target_sha256'] != b['target_sha256'] for a, b in zip(parent, rows)), 6)
        self.assertEqual(rows[290]['parent_target_id'], 'replacement_author_paragraph')
        self.assertEqual(rows[290]['provenance']['replaced_parent_target_id'], 'original_291')
        self.assertTrue(rows[263]['quality']['source_alignment_pending'])

    def test_wrong_binding_or_duplicate_target_rejected(self):
        parent, decisions = self.fixtures()
        decisions[0]['before_target_sha256'] = 'wrong'
        with self.assertRaises(ValueError):
            v.target_versions(parent, decisions)
        parent, decisions = self.fixtures()
        d = next(x for x in decisions if x['id'] == 'P250')
        d['target_edits'][0]['after'] = parent[0]['target']
        d['after_target_sha256'] = parent[0]['target_sha256']
        with self.assertRaises(ValueError):
            v.target_versions(parent, decisions)

    def test_span_edits_cannot_match_multiple_occurrences(self):
        with self.assertRaises(ValueError):
            v.replace_once('word word', [dict(before='word', after='other', reason='ambiguous')])


if __name__ == '__main__':
    unittest.main()
