"""Synthetic provenance and immutability checks for semantic repairs."""
import copy
import json
import unittest
from test_neutral_reconstruction import brief
from fama_style.research.neutral_reconstruction import sha
from fama_style.research.neutral_reconstruction_binding_repair import apply_repair


class BindingRepair(unittest.TestCase):
    def fixture(self):
        record={'record_id':'SYN','original_paragraph':'Firms are not confident.','brief':brief()}
        patch={'path':['claims',0,'negation'],'before':[],
               'after':['Firms lack confidence.'],'source_evidence':'are not confident',
               'rationale':'Preserve categorical confidence scope.'}
        decision={'record_id':'SYN','reference_sha256':sha(record['original_paragraph']),
                  'before_brief_sha256':sha(json.dumps(record['brief'],sort_keys=True)), 'patches':[patch]}
        return record,decision

    def test_repair_preserves_inputs_ids_and_edge_bindings(self):
        record,decision=self.fixture();before=copy.deepcopy(record)
        result=apply_repair(record,decision)
        self.assertEqual(record,before)
        self.assertFalse(result['training_ready'])
        self.assertEqual(result['reference_sha256'],sha(record['original_paragraph']))
        self.assertEqual([c['claim_id'] for c in result['brief']['claims']],['C1','C2'])
        self.assertEqual(result['brief']['supported_relationships'],record['brief']['supported_relationships'])
        self.assertNotEqual(result['before_brief_sha256'],result['after_brief_sha256'])

    def test_source_and_before_value_drift_are_rejected(self):
        for field in ('reference_sha256','before_brief_sha256'):
            record,decision=self.fixture();decision[field]='changed'
            with self.assertRaises(ValueError):apply_repair(record,decision)
        record,decision=self.fixture();decision['patches'][0]['before']=['wrong']
        with self.assertRaises(ValueError):apply_repair(record,decision)

    def test_unanchored_or_out_of_scope_patch_rejected(self):
        for change in ({'source_evidence':'absent phrase'}, {'rationale':''},
                       {'path':['needs_review']}):
            record,decision=self.fixture();decision['patches'][0].update(change)
            with self.assertRaises(ValueError):apply_repair(record,decision)


if __name__=='__main__':unittest.main()
