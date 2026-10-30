"""Check provenance and output ownership, not truth of semantic judgments."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.research import claim_bound_expression_review as review


class ExpressionReviewTests(unittest.TestCase):
    def fixture(self):
        source='Costs rise. Firms save. Demand grows.'
        output='Costs increase, and firms save. Demand grows.'
        source_units=['Costs rise.','Firms save.','Demand grows.']
        output_units=['Costs increase, and firms save.']*2+['Demand grows.']
        anchor=review.experiment.binding.anchored
        claims=[{'claim_id':f'c{i}','source_spans':[anchor(source,s)[1]]} for i,s in enumerate(source_units,1)]
        card={'record':{'source':source},'evidence':{'claim_index':claims},
              'plan':[{'output_slots':[{'claim_refs':['c1','c2']},{'claim_refs':['c3']}]}],
              'expression_operation':{'control_id':'FINITE_SUPPORTED_CONSEQUENCE','claim_refs':['c1','c2']}}
        assessment={'source_sha256':review.experiment.parent.base.sha(source),'output_sha256':review.experiment.parent.base.sha(output),
            'claims':[{'claim_id':c['claim_id'],'status':'retained','source_span':c['source_spans'][0],
                       'output_span':anchor(output,t)[1]} for c,t in zip(claims,output_units)],
            'output_claim_groups':[['c1','c2'],['c3']],'nonpacking_style_gain':'small',
            'operation_assessment':{'control_id':'FINITE_SUPPORTED_CONSEQUENCE','claim_refs':['c1','c2'],
                'form_present':True,'within_fixed_grouping':True,'output_evidence':[anchor(output,'firms save')[1]]}}
        return card,output,assessment

    def test_valid_assessment_and_operation_identity(self):
        card,out,a=self.fixture();self.assertTrue(review.validate_assessment(card,out,a))
        a['operation_assessment']['claim_refs']=['c3']
        with self.assertRaisesRegex(ValueError,'operation_identity'):review.validate_assessment(card,out,a)

    def test_reject_operation_evidence_from_unrelated_claim(self):
        card,out,a=self.fixture()
        a['operation_assessment']['output_evidence']=[review.experiment.binding.anchored(out,'Demand grows.')[1]]
        with self.assertRaisesRegex(ValueError,'operation_outside_claims'):review.validate_assessment(card,out,a)

    def test_grouping_and_text_identity_not_silent_repair(self):
        card,out,a=self.fixture();a['operation_assessment']['within_fixed_grouping']=False
        with self.assertRaisesRegex(ValueError,'grouping_judgment'):review.validate_assessment(card,out,a)
        card,out,a=self.fixture();a['output_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'text_identity_changed'):review.validate_assessment(card,out,a)

    @unittest.skipUnless((review.experiment.ROOT/'research_runs/claim_bound_expression_v1/review_v1/manifest.json').exists(),'Private replies unavailable')
    def test_private_reproduction_and_raw_bytes(self):
        root=review.experiment.ROOT/'research_runs/claim_bound_expression_v1'
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'review'
            review.run(root/'package_v1',root/'responses',output,root/'review_spec_v1/assessments.private.json')
            summary=json.loads((output/'summary.json').read_text())
            self.assertEqual(summary,json.loads((root/'review_v1/summary.json').read_text()))
            self.assertEqual(summary['accepted_outputs'],6);self.assertEqual(summary['claim_instances_reviewed'],30)
            self.assertFalse(summary['distinctive_author_style_verified'])
            for paper in summary['per_paper']:
                self.assertTrue(paper['conditions']['selective_expression']['operation_assessment']['form_present'])
                self.assertTrue(paper['conditions']['selective_expression']['operation_assessment']['within_fixed_grouping'])
            for name,digest in json.loads((output/'coverage.json').read_text())['response_file_hashes'].items():
                self.assertEqual(review.experiment.binding.file_sha(output/'raw'/name),digest)
                self.assertEqual(review.experiment.binding.file_sha(root/'responses'/name),digest)


if __name__=='__main__':unittest.main()
