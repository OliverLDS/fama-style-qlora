"""Assessment provenance checks; no assertions that mechanical checks prove meaning."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.research import source_repair_review as review


class SourceRepairReviewTests(unittest.TestCase):
    def fixture(self):
        source='Costs rise. Demand falls.'; output='Costs increase. Demand falls.'
        source_units=['Costs rise.','Demand falls.']; output_units=['Costs increase.','Demand falls.']
        claims=[{'claim_id':f'c{i}','source_spans':[review.experiment.bindings.anchored(source,s)[1]]} for i,s in enumerate(source_units,1)]
        card={'record':{'source':source},'evidence':{'claim_index':claims}}
        assessment={'source_sha256':review.experiment.base.sha(source),'output_sha256':review.experiment.base.sha(output),
            'claims':[{'claim_id':c['claim_id'],'status':'retained','source_span':c['source_spans'][0],
                       'output_span':review.experiment.bindings.anchored(output,t)[1]} for c,t in zip(claims,output_units)],
            'output_claim_groups':[['c1'],['c2']],'nonpacking_style_gain':'small'}
        return card,output,assessment

    def test_token_edit_distance(self):
        self.assertEqual(review.token_edit_distance('Costs rise.','Costs rise.'),0)
        self.assertEqual(review.token_edit_distance('Costs rise.','Costs fall.'),1)
        self.assertEqual(review.token_edit_distance('','Costs rise.'),2)
        self.assertEqual(review.token_edit_distance('A B','B A'),2)

    def test_evidence_identity_not_semantic_proof(self):
        card,output,assessment=self.fixture()
        self.assertTrue(review.validate_assessment(card,output,assessment))
        assessment['output_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'text_identity'):review.validate_assessment(card,output,assessment)

    def test_omitted_or_duplicate_claim_rejected(self):
        card,output,assessment=self.fixture()
        assessment['claims'][1]=copy.deepcopy(assessment['claims'][0])
        with self.assertRaisesRegex(ValueError,'claim_coverage'):review.validate_assessment(card,output,assessment)

    def test_wrong_source_or_output_anchor_rejected(self):
        card,output,assessment=self.fixture()
        assessment['claims'][0]['source_span']={'start':0,'end':1,'quote':'C'}
        with self.assertRaisesRegex(ValueError,'source_anchor'):review.validate_assessment(card,output,assessment)
        card,output,assessment=self.fixture(); assessment['claims'][0]['output_span']['quote']='Altered'
        with self.assertRaisesRegex(ValueError,'output_anchor'):review.validate_assessment(card,output,assessment)

    def test_swapped_output_groups_rejected(self):
        card,output,assessment=self.fixture();assessment['output_claim_groups']=[['c2'],['c1']]
        with self.assertRaisesRegex(ValueError,'group_evidence'):review.validate_assessment(card,output,assessment)

    @unittest.skipUnless((review.experiment.ROOT/'research_runs/source_repair_comparison_v1/review_v1/manifest.json').exists(),'Private responses unavailable')
    def test_private_reproduction_and_raw_preservation(self):
        root=review.experiment.ROOT/'research_runs/source_repair_comparison_v1'
        original=json.loads((root/'review_v1/summary.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/'audit'
            review.run(root/'package_v1',root/'responses',dest,root/'review_spec_v1/assessments.private.json')
            self.assertEqual(original,json.loads((dest/'summary.json').read_text()))
            coverage=json.loads((dest/'coverage.json').read_text())
            self.assertTrue(coverage['complete']);self.assertEqual(coverage['accepted_jobs'],16)
            for name,digest in coverage['response_file_hashes'].items():
                self.assertEqual(review.experiment.file_sha(dest/'raw'/name),digest)
                self.assertEqual(review.experiment.file_sha(root/'responses'/name),digest)
            metrics=review.experiment.read_lines(dest/'metrics.private.jsonl')
            self.assertEqual(sum(len(m['semantic_assessment']['claims']) for m in metrics),84)
            self.assertFalse(original['distinctive_author_style_verified'])
            self.assertEqual(sum(m['grouping_matches_evidence_review'] is True for m in metrics if m['mode']=='planned'),8)


if __name__=='__main__':unittest.main()
