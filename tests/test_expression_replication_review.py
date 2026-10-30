"""Keep existing forms distinct from new gains and preserve private raw replies."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.research import expression_replication_review as review


class ReplicationReviewTests(unittest.TestCase):
    def test_subject_already_present_is_not_initial_position_realization(self):
        card={'expression_operation':{'control_id':'FOREGROUND_SUPPORTED_RELATION_OR_FUNCTION'}}
        assessment={'form_observations':{'focus_as_subject':True,'focus_initial':False},
                    'operation_assessment':{'form_present':False},'incremental_style_benefit':'none_clear'}
        with patch.object(review.shared,'validate_assessment',return_value=True):
            self.assertTrue(review.validate_assessment(card,'synthetic output',assessment))
            assessment['operation_assessment']['form_present']=True
            with self.assertRaisesRegex(ValueError,'strict_form_observation_mismatch'):review.validate_assessment(card,'synthetic output',assessment)

    def test_finite_clause_and_specified_actor_are_distinct(self):
        card={'expression_operation':{'control_id':'FINITE_SUPPORTED_CONSEQUENCE'}}
        assessment={'form_observations':{'finite_predicate':True,'specified_actor_as_subject':False},
                    'operation_assessment':{'form_present':False},'incremental_style_benefit':'none_clear'}
        with patch.object(review.shared,'validate_assessment',return_value=True):
            self.assertTrue(review.validate_assessment(card,'synthetic output',assessment))
            assessment['form_observations']['specified_actor_as_subject']='unknown'
            with self.assertRaisesRegex(ValueError,'form_observations'):review.validate_assessment(card,'synthetic output',assessment)

    @unittest.skipUnless((review.experiment.ROOT/'research_runs/expression_replication_v1/review_v1/manifest.json').exists(),'Private replication replies unavailable')
    def test_private_rebuild_ceiling_cases_and_preserved_raw_files(self):
        root=review.experiment.ROOT/'research_runs/expression_replication_v1'
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'review'
            review.run(root/'package_v1',root/'responses',output,root/'review_spec_v1/assessments.private.json')
            summary=json.loads((output/'summary.json').read_text())
            self.assertEqual(summary,json.loads((root/'review_v1/summary.json').read_text()))
            self.assertEqual(summary['accepted_outputs'],12);self.assertEqual(summary['claim_instances_reviewed'],57)
            self.assertFalse(summary['distinctive_author_style_verified'])
            papers={p['paper_key']:p for p in summary['per_paper']}
            for p in papers.values():
                guided=p['conditions']['selective_expression']
                self.assertTrue(guided['operation_assessment']['form_present'])
                self.assertTrue(guided['operation_assessment']['within_fixed_grouping'])
            self.assertTrue(papers['F1965']['conditions']['grouping_control']['form_observations']['focus_as_subject'])
            for paper in ('FF2000','VL2018'):
                self.assertTrue(papers[paper]['conditions']['grouping_control']['form_observations']['finite_predicate'])
            hashes=json.loads((output/'coverage.json').read_text())['response_file_hashes']
            for name,digest in hashes.items():
                self.assertEqual(review.experiment.binding.file_sha(output/'raw'/name),digest)
                self.assertEqual(review.experiment.binding.file_sha(root/'responses'/name),digest)


if __name__=='__main__':unittest.main()
