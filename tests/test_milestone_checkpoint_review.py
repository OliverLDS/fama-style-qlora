"""Masked evidence binding and non-interchangeable checkpoint outcomes."""
import copy
import unittest

from fama_style.evaluation.milestone_checkpoint_4b import CONDITIONS,sha
from fama_style.research.milestone_checkpoint_review import join,summarize


class MilestoneReview(unittest.TestCase):
    def fixture(self):
        bundle=[];key=[];raw=[];judgments={'cases':[]}
        for i in range(8):
            rid=str(i);source=f'The input for case {i} has two claims.';outputs={};assessments={}
            for label,arm in zip('ABC',CONDITIONS):
                text=f'Output {label} for case {i} states a claim.';outputs[label]=text
                key.append({'example_id':rid,'label':label,'condition':arm,'output_sha256':sha(text)})
                raw.append({'example_id':rid,'condition':arm,'source':source,'output':text,'copied_source':False})
                assessments[label]={'minor_semantic_shifts':0 if label=='A' else 1 if label=='B' else 2,
                    'material_claim_change':label=='A','unresolved':False,'notes':'Synthetic evidence only.'}
            bundle.append({'example_id':rid,'source':source,'reference':source,'outputs':outputs})
            judgments['cases'].append({'example_id':rid,'ranking_best_first':[['B','C'],['A']],'outputs':assessments})
        return bundle,key,raw,judgments

    def test_meaning_budget_and_style_are_separate(self):
        summary=summarize(join(*self.fixture()))
        self.assertEqual(summary['arms']['base_ordinary']['material_claim_issues'],8)
        self.assertEqual(summary['arms']['base_source_structural_prior']['eligible_within_one_minor_shift_budget'],8)
        self.assertEqual(summary['arms']['adapter_ordinary']['material_claim_issues'],0)
        self.assertEqual(summary['arms']['adapter_ordinary']['eligible_within_one_minor_shift_budget'],0)
        self.assertEqual(summary['paired_against_base_ordinary']['adapter_ordinary']['generic_exposition_preferences']['better'],8)

    def test_changed_text_and_partial_judgments_are_rejected(self):
        bundle,key,raw,judgments=self.fixture();bundle[0]['outputs']['A']='Modified output.'
        with self.assertRaises(ValueError):join(bundle,key,raw,judgments)
        args=self.fixture();args[3]['cases'].pop()
        with self.assertRaises(ValueError):join(*args)
        args=self.fixture();args[3]['cases'][0]['ranking_best_first']=[['A','A'],['B']]
        with self.assertRaises(ValueError):join(*args)

    def test_unresolved_is_not_success_and_identical_texts_must_tie(self):
        args=self.fixture();args[3]['cases'][0]['outputs']['B']['unresolved']=True
        summary=summarize(join(*args))
        self.assertEqual(summary['arms']['base_source_structural_prior']['eligible_within_one_minor_shift_budget'],7)
        bundle,key,raw,judgments=self.fixture()
        raw[1]['output']=raw[0]['output'];bundle[0]['outputs']['B']=raw[0]['output'];key[1]['output_sha256']=key[0]['output_sha256']
        with self.assertRaises(ValueError):join(bundle,key,raw,judgments)


if __name__=='__main__':unittest.main()
