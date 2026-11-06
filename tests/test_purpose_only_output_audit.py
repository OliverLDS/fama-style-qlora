import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('output_audit', Path(__file__).resolve().parents[1] / 'scripts/data/audit_gpt6_purpose_only_outputs.py')
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)


class PurposeOnlyOutputAuditTests(unittest.TestCase):
    def fixture(self, rid='P056'):
        target = dict(id=rid, target='Human reference.', target_sha256=a.m.sha('Human reference.'),
            paper_id='training_paper', parent_target_id='target_id', authors=['Author'], style_group='Group',
            canonical_pdf_sha256='pdf', target_kind='prior_target_unchanged')
        raw = dict(id=rid, target_sha256=target['target_sha256'], plain_english='Observed AI paragraph.',
                   rewrite_status='source_problem', review_notes=['A model warning.'])
        brief = dict(target_sha256=target['target_sha256'], brief=dict(topic='Topic', relationships=[]))
        return target, raw, brief

    def test_model_source_problem_does_not_filter_or_edit_pair(self):
        target, raw, brief = self.fixture()
        before = copy.deepcopy((target, raw, brief))
        with patch.object(a.m, 'digest', return_value='manifest'):
            row = a.candidate(raw, target, brief, 'response.jsonl')
        self.assertEqual(row['source'], raw['plain_english'])
        self.assertEqual(row['target'], target['target'])
        self.assertEqual(row['raw_response'], raw)
        self.assertEqual(row['quality']['raw_model_status'], 'source_problem')
        self.assertEqual(row['quality']['followup']['level'], 'model_flag_nonblocking')
        self.assertFalse(row['training_ready'])
        self.assertEqual((target, raw, brief), before)

    def test_conflict_is_retained_but_flagged_and_bound(self):
        target, raw, brief = self.fixture('P139')
        with patch.object(a.m, 'digest', return_value='manifest'):
            row = a.candidate(raw, target, brief, 'response.jsonl')
            self.assertEqual(row['quality']['text_pair_status'], 'Candidate · targeted regeneration pending')
            self.assertEqual(row['source_sha256'], a.m.sha(raw['plain_english']))
            raw['target_sha256'] = 'wrong'
            with self.assertRaises(ValueError):
                a.candidate(raw, target, brief, 'response.jsonl')


if __name__ == '__main__':
    unittest.main()
