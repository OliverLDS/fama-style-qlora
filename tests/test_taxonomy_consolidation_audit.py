"""Test mapping coverage, evidence references and nonassignment of ambiguous splits."""
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('consolidation_audit',ROOT/'scripts/data/audit_taxonomy_consolidation.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def fixture():
    def item(ident):
        return {'canonical_id':ident,'level':'sentence','dimension':'voice','name':'Synthetic voice item',
            'definition':'Synthetic category for testing.','inclusion_rules':[],'exclusion_rules':[],
            'allowed_values':[],'applicability':'Synthetic','evidence_requirements':[],
            'control_status':'candidate_presentation_control','preconditions':['Preserve claims'],
            'prohibited_effects':['No new claims'],'supporting_inventory_ids':['I1'] if ident=='S1' else ['I2'],
            'related_item_ids':[],'provisional_reason':None}
    a,b=item('S1'),item('S2');a['supporting_inventory_ids'].append('I2')
    taxonomy={'status':'proposed_not_frozen','canonical_items':[a,b],
        'inventory_mapping':[{'inventory_id':'I1','status':'mapped','canonical_ids':['S1'],'disambiguation_rule':None},
            {'inventory_id':'I2','status':'split_requires_context','canonical_ids':['S1','S2'],'disambiguation_rule':'Inspect the whole clause.'}]}
    inventory=[{'inventory_id':name,'level':'sentence','dimension':'voice','occurrence_count':1,
                'primary_context_counts':{'original_observed':1}} for name in ('I1','I2')]
    occurrences=[{'inventory_id':name,'annotation_id':'synthetic','json_path':'/'+name} for name in ('I1','I2')]
    return taxonomy,inventory,occurrences


class ConsolidationAuditTests(unittest.TestCase):
    def test_split_preserves_candidates_without_assigning_both(self):
        summary,rows=audit.check(*fixture())
        self.assertTrue(summary['mechanical_validation_passed'])
        self.assertEqual(rows[0]['canonical_id'],'S1')
        self.assertIsNone(rows[1]['canonical_id'])
        self.assertEqual(rows[1]['candidate_canonical_ids'],['S1','S2'])
        self.assertEqual(summary['categories_with_unambiguous_support'],1)
        self.assertFalse(summary['ready_to_freeze'])

    def test_unknown_references_and_duplicate_mapping_fail(self):
        t,i,o=fixture();t['inventory_mapping'][0]['canonical_ids']=['UNKNOWN']
        summary,rows=audit.check(t,i,o)
        self.assertIn('unknown_canonical_reference',{r['code'] for r in summary['mechanical_findings']})
        self.assertEqual(rows,[])
        t,i,o=fixture();t['inventory_mapping'].append(copy.deepcopy(t['inventory_mapping'][0]))
        self.assertIn('duplicate_mapping_id',{r['code'] for r in audit.check(t,i,o)[0]['mechanical_findings']})

    def test_missing_support_guards_or_context_rule_fail(self):
        for mutation,code in [('support','support_mapping_inverse_mismatch'),('guards','presentation_control_missing_guards'),('rule','split_missing_alternatives_or_rule')]:
            t,i,o=fixture()
            if mutation=='support':t['canonical_items'][0]['supporting_inventory_ids']=[]
            elif mutation=='guards':t['canonical_items'][0]['preconditions']=[]
            else:t['inventory_mapping'][1]['disambiguation_rule']=None
            self.assertIn(code,{r['code'] for r in audit.check(t,i,o)[0]['mechanical_findings']})


if __name__=='__main__':unittest.main()
