"""Synthetic binding checks: identity and evidence validity do not certify meaning."""
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('resolution_audit',ROOT/'scripts/data/audit_taxonomy_resolutions.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def fixtures():
    case={'case_id':'case1','inventory_id':'I1','context_id':'ctx1','candidate_canonical_ids':['S001','S002']}
    context={'text':'Firms report revenue.','semantic_content':{'claims':[{'claim_id':'c1'}],'relations':[]}}
    row={'case_id':'case1','inventory_id':'I1','context_id':'ctx1','remaining_candidate_ids':[],
        'assignments':[{'canonical_id':'S001','evidence':[{'quote':'Firms report revenue.','occurrence':1}],
                        'claim_refs':['c1'],'semantic_relation_refs':[]}]}
    return row,case,context


class ResolutionAuditTests(unittest.TestCase):
    def test_valid_binding_without_mutation(self):
        args=fixtures();before=copy.deepcopy(args)
        self.assertEqual(audit.check_response(*args),[]);self.assertEqual(args,before)

    def test_wrong_case_unknown_id_and_missing_quote_rejected(self):
        for change,expected in [('context','case_binding_mismatch'),('category','category_outside_case_candidates'),
                                ('claim','unknown_claim_ref'),('relation','unknown_relation_ref'),('quote','unresolved_evidence_anchor')]:
            row,case,context=fixtures()
            if change=='context':row['context_id']='wrong'
            elif change=='category':row['assignments'][0]['canonical_id']='UNKNOWN'
            elif change=='claim':row['assignments'][0]['claim_refs']=['c99']
            elif change=='relation':row['assignments'][0]['semantic_relation_refs']=['r99']
            else:row['assignments'][0]['evidence'][0]['occurrence']=2
            self.assertIn(expected,{e['code'] for e in audit.check_response(row,case,context)})

    def test_duplicate_component_assignment_is_not_independent_support(self):
        row,case,context=fixtures();row['assignments'].append(copy.deepcopy(row['assignments'][0]))
        self.assertIn('duplicate_component_assignment',{e['code'] for e in audit.check_response(row,case,context)})


if __name__=='__main__':unittest.main()
