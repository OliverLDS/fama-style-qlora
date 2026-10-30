"""Complete, immutable mismatch cards and evidence-bound discovery judgments."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.research import boundary_error_audit as module


class BoundaryErrorAuditTests(unittest.TestCase):
    def fixture(self):
        annotation={'split':'training','annotation':{'plain_english':'Firms borrow. Banks lend.',
            'original_paragraph':'Firms borrow while banks lend.',
            'source_schema':{'sentences':[{'exact_text':'Firms borrow.','claim_ids':['C1']},
                {'exact_text':'Banks lend.','claim_ids':['C2']}]},
            'target_schema':{'sentences':[{'exact_text':'Firms borrow while banks lend.','claim_ids':['C1','C2']}]},
            'source_quality':{'needs_external_context':False},'review_notes':[]}}
        prediction={'paragraph_id':'q','paper_id':'p','gold_labels':[1],'predicted_labels':[0],
            'predicted_source_sentence_groups':[[1],[2]]}
        return prediction,annotation

    def assessment(self,card):
        return {'case_id':card['case_id'],'category':'supported_merge_missed',
            'relation':'parallel_actions','rationale':'Same-context parallel actions can be coordinated.',
            'confirmed_label_error':False,
            'source_evidence':[{'quote':card['left'],'occurrence':1},{'quote':card['right'],'occurrence':1}],
            'reference_evidence':[{'quote':card['target_sentences'][0],'occurrence':1}]}

    def test_error_enumeration_and_inputs_unchanged(self):
        prediction,annotation=self.fixture();before=copy.deepcopy((prediction,annotation))
        cards=module.make_cards([prediction],{'q':annotation})
        self.assertEqual(len(cards),1);self.assertEqual(cards[0]['kind'],'missed')
        self.assertEqual(cards[0]['boundary'],0)
        self.assertEqual((prediction,annotation),before)
        self.assertEqual(cards,module.make_cards([prediction],{'q':annotation}))
        prediction['predicted_labels']=[1];prediction['predicted_source_sentence_groups']=[[1,2]]
        self.assertEqual(module.make_cards([prediction],{'q':annotation}),[])

    def test_reserved_record_rejected_before_annotation_access(self):
        prediction,_=self.fixture()
        with self.assertRaisesRegex(ValueError,'training_only_audit'):
            module.make_cards([prediction],{'q':{'split':'reserved_test'}})

    def test_both_source_sides_and_exact_reference_evidence_required(self):
        prediction,annotation=self.fixture();card=module.make_cards([prediction],{'q':annotation})[0]
        module.validate_assessment(card,self.assessment(card))
        for mode in ('invented','missing_side','wrong_occurrence','missing_reference'):
            a=self.assessment(card)
            if mode=='invented':a['reference_evidence'][0]['quote']='Invented wording.'
            elif mode=='missing_side':a['source_evidence'].pop()
            elif mode=='wrong_occurrence':a['source_evidence'][0]['occurrence']=2
            else:a['reference_evidence']=[]
            with self.assertRaises(ValueError):module.validate_assessment(card,a)

    def test_categories_must_match_extra_or_missed_direction(self):
        prediction,annotation=self.fixture();card=module.make_cards([prediction],{'q':annotation})[0]
        a=self.assessment(card);a['category']='plausible_alternative'
        with self.assertRaisesRegex(ValueError,'category_kind'):module.validate_assessment(card,a)

    def test_composite_category_requires_shared_claim(self):
        prediction,annotation=self.fixture();card=module.make_cards([prediction],{'q':annotation})[0]
        a=self.assessment(card);a['category']='composite_claim_granularity'
        with self.assertRaisesRegex(ValueError,'no_composite_claim_basis'):module.validate_assessment(card,a)
        annotation['annotation']['source_schema']['sentences'][1]['claim_ids']=['C1']
        annotation['annotation']['target_schema']['sentences'][0]['claim_ids']=['C1']
        card=module.make_cards([prediction],{'q':annotation})[0]
        module.validate_assessment(card,self.assessment(card)|{'category':'composite_claim_granularity'})

    def test_confirmed_label_error_cannot_be_asserted_without_explanation(self):
        prediction,annotation=self.fixture();card=module.make_cards([prediction],{'q':annotation})[0]
        a=self.assessment(card);a['confirmed_label_error']=True
        with self.assertRaisesRegex(ValueError,'label_error_requires_explanation'):module.validate_assessment(card,a)
