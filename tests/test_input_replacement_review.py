"""Five-condition review requires complete mappings and exact historical reuse."""
import copy
import unittest

from fama_style.training.teacher_pair_input_replacement import sha
from fama_style.research.input_replacement_review import CONDITIONS,join_masked,paired


class InputReplacementReview(unittest.TestCase):
    def fixture(self):
        raw={};bundle=[];key=[];assess=[];prefs=[];historical=[]
        for i in range(10):
            rid=str(i);source='Synthetic source '+rid;reference='Synthetic target '+rid
            split='development' if i<8 else 'training';outputs={}
            for label,condition in zip('ABCDE',CONDITIONS):
                text=condition+' output '+rid
                row={'example_id':rid,'source':source,'reference':reference,'split':split,'output':text}
                raw[(condition,rid)]=row;outputs[label]=text
                key.append({'example_id':rid,'label':label,'condition':condition})
                a={'example_id':rid,'label':label,'text_sha256':sha(text),
                    'material_claim_issue_identified':label!='E','notes':'Synthetic fixed judgment.',
                    'reused_text_bound_assessment':label in 'ABC'}
                assess.append(a)
                if label in 'ABC':historical.append(row|{'assessment':copy.deepcopy(a)})
            bundle.append({'example_id':rid,'source':source,'reference':reference,'split':split,'outputs':outputs})
            prefs.append({'example_id':rid,'ranking_best_first':[['E'],['D'],['C'],['B'],['A']]})
        return raw,bundle,key,assess,prefs,historical

    def test_every_condition_and_target_bound_to_exact_saved_text(self):
        args=self.fixture();result=join_masked(*args);self.assertEqual(len(result),50)
        raw,bundle,key,assess,prefs,historical=args
        for target in ('source','reference','split'):
            changed=copy.deepcopy(bundle);changed[0][target]='changed'
            with self.assertRaises(ValueError):join_masked(raw,changed,key,assess,prefs,historical)
        with self.assertRaises(ValueError):join_masked(raw,bundle,key[:-1],assess,prefs,historical)
        changed=copy.deepcopy(assess);changed[0]['material_claim_issue_identified']=False
        with self.assertRaises(ValueError):join_masked(raw,bundle,key,changed,prefs,historical)
        changed=copy.deepcopy(assess);changed[0]['text_sha256']='wrong'
        with self.assertRaises(ValueError):join_masked(raw,bundle,key,changed,prefs,historical)

    def test_unique_five_way_ranking_and_identical_text_ties(self):
        raw,bundle,key,assess,prefs,historical=self.fixture()
        changed=copy.deepcopy(prefs);changed[0]['ranking_best_first']=[['A','A','B','C','D']]
        with self.assertRaises(ValueError):join_masked(raw,bundle,key,assess,changed,historical)
        bundle[0]['outputs']['E']=bundle[0]['outputs']['D']
        with self.assertRaises(ValueError):join_masked(raw,bundle,key,assess,prefs,historical)

    def test_paired_comparison_retains_identity_and_zero_counts(self):
        result=join_masked(*self.fixture())
        control=[r for r in result if r['condition']=='control_epoch_2' and r['split']=='development']
        treatment=[r for r in result if r['condition']=='replacement_epoch_2' and r['split']=='development']
        summary=paired(control,treatment)
        self.assertEqual(summary['material_issue_transitions'],{'improved':8,'same':0,'worse':0})
        self.assertEqual(summary['generic_style_preferences'],{'better':8,'tie':0,'worse':0})
        with self.assertRaises(ValueError):paired(control,treatment[:-1])


if __name__=='__main__':unittest.main()
