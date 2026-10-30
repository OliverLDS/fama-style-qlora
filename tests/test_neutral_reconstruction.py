"""Synthetic tests for balanced selection and reference-free brief transport."""
import copy
from collections import Counter
import unittest

from fama_style.research import neutral_reconstruction as n


def brief():
    def claim(key,text):
        return dict({field:[] for field in n.CLAIM_LISTS},claim_id=key,actor_ids=['A1'],
                    proposition=text,predicate='adjustment')
    first=claim('C1','Firms may reduce investment when borrowing becomes more expensive.')
    first['modality']=['may'];first['conditions']=['borrowing becomes more expensive']
    second=claim('C2','The model compares two financing arrangements.')
    second['quantities']=['two financing arrangements']
    return {'brief_version':n.BRIEF_VERSION,'record_id':'SYN',
            'actors':[{'actor_id':'A1','name':'firms'}],'claims':[first,second],
            'supported_relationships':[{'from_claim':'C1','to_claim':'C2','type':'compares','condition':'conditional model','qualification':'not causal'}],
            'essential_terms':['investment'],'citations':[],'context_gaps':[],'needs_review':False}


class NeutralReconstruction(unittest.TestCase):
    def test_balanced_selection_is_order_independent_and_excludes_reserved(self):
        rows=[]
        for group in n.GROUPS:
            for paper in range(3):
                for para in range(3):
                    rows.append({'style_group':group,'paper_id':group+str(paper),
                                 'paragraph_id':group+str(paper)+str(para),'status':'annotation_candidate',
                                 'quality_flags':[],'word_count':100,'duplicate_of':None,
                                 'pdf_page_start':3,'pdf_page_end':3,'column_start':1,'column_end':1,
                                 'existing_split':'unassigned_new_acquisition',
                                 'original_paragraph':'Firms respond to financing constraints under specified conditions.'})
        reserved=copy.deepcopy(rows[0]);reserved.update(paragraph_id='reserved',paper_id='reserved',existing_split='reserved_test')
        rows.append(reserved)
        batches,roles=n.select(rows)
        self.assertEqual(batches,n.select(list(reversed(rows)))[0])
        self.assertEqual(len(roles),6)
        for batch in batches:
            self.assertEqual(len(batch),6)
            self.assertEqual(set(Counter(r['style_group'] for r in batch).values()),{2})
            self.assertEqual(len({r['paper_id'] for r in batch}),6)
            self.assertNotIn('reserved',[r['paragraph_id'] for r in batch])

    def test_opaque_claim_ids_preserve_edge_direction_and_qualifiers(self):
        original=brief();saved=copy.deepcopy(original)
        result=n.make_handoff(original,'A short synthetic reference paragraph.')
        self.assertEqual(original,saved)
        self.assertTrue(all(c['claim_id'].startswith('K_') for c in result['claims']))
        mapping={c['proposition']:c for c in result['claims']}
        first=mapping[original['claims'][0]['proposition']]
        self.assertEqual(first['modality'],['may'])
        edge=result['supported_relationships'][0]
        self.assertEqual(edge['from_claim'],first['claim_id'])
        self.assertEqual(edge['qualification'],'not causal')
        self.assertEqual(result,n.make_handoff(original,'A short synthetic reference paragraph.'))

    def test_extra_reference_or_plan_fields_and_unknown_edges_are_rejected(self):
        for key in ('original_paragraph','paragraph_plan','author_style'):
            value=brief();value[key]='unexpected reference material'
            with self.assertRaises(ValueError):n.validate_brief(value)
        value=brief();value['supported_relationships'][0]['to_claim']='unknown'
        with self.assertRaises(ValueError):n.validate_brief(value)

    def test_uncertain_briefs_and_long_verbatim_reference_spans_are_held(self):
        value=brief();value['context_gaps']=['missing antecedent']
        with self.assertRaises(ValueError):n.make_handoff(value,'Synthetic original.')
        value=brief()
        original='Synthetic reference wording includes enough distinct tokens to test a complete twelve token transport overlap.'
        value['claims'][0]['proposition']=original
        with self.assertRaises(ValueError):n.make_handoff(value,original)

    def test_valid_response_never_certifies_semantic_alignment(self):
        source={'record_id':'SYN','original_sha256':'a'*64}
        response=dict(source,source_audit={'status':'usable','transcription_verification':'not_performed','issues':[]},
                      brief=brief(),self_check={key:True for key in ('all_claims_included','no_added_claims_or_relations','logical_force_preserved','uncertainty_reported')})
        result=n.validate_response(response,source)
        self.assertTrue(result['eligible_for_semantic_assessment'])
        self.assertFalse(result['semantic_alignment_assessed'])
        response['original_sha256']='b'*64
        with self.assertRaises(ValueError):n.validate_response(response,source)


if __name__=='__main__':unittest.main()
