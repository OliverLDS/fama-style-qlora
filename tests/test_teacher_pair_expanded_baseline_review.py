"""CPU checks for exact-evidence reuse and complete expanded label coverage."""
import copy
import unittest

from fama_style.training.teacher_pair_expanded_pilot import sha,prompt
from fama_style.research.teacher_pair_expanded_baseline_review import validate_labels,reuse_assessments


class ExpandedBaselineReview(unittest.TestCase):
    def fixture(self):
        groups={'training':[],'development':[]};audit=[]
        for split,n in [('training',46),('development',8)]:
            for i in range(n):
                row={'example_id':split+str(i),'split':split,'source':f'{split} source {i}',
                    'target':f'{split} target {i}'}
                row['prompt']=prompt(row['source']);groups[split].append(row)
                audit.append({'example_id':row['example_id'],'split':split,'masked_tokens':100,
                    'supervised_tokens':30,'full_tokens':130,'labels_match_target':True,'input_ids_sha256':'a'*64})
        config={'development_ids':[r['example_id'] for r in groups['development']],
            'training_probe_ids':[r['example_id'] for r in groups['training'][:2]]}
        generations=[];assessments=[]
        for row in groups['development']+groups['training'][:2]:
            generations.append({'example_id':row['example_id'],'split':row['split'],'source':row['source'],
                'reference':row['target'],'prompt_sha256':sha(row['prompt']),'output':'Synthetic output.',
                'empty':False,'copied_source':False})
            assessments.append({'job_id':row['example_id'],'text_sha256':sha('Synthetic output.')})
        return groups,audit,config,generations,assessments

    def test_all_54_masks_and_generation_budget_must_pass(self):
        groups,audit,*_=self.fixture();validate_labels(audit,groups)
        for field,value in [('masked_tokens',600),('labels_match_target',False),('input_ids_sha256','invalid'),
                            ('supervised_tokens',0),('split','reserved')]:
            changed=copy.deepcopy(audit);changed[0][field]=value
            with self.assertRaises(ValueError):validate_labels(changed,groups)
        with self.assertRaises(ValueError):validate_labels(audit[:-1],groups)
        changed=copy.deepcopy(audit);changed[-1]=changed[0]
        with self.assertRaises(ValueError):validate_labels(changed,groups)

    def test_assessment_reuse_needs_identical_source_reference_prompt_and_output(self):
        groups,audit,config,current,assessments=self.fixture()
        joined=reuse_assessments(current,copy.deepcopy(current),assessments,groups,config)
        self.assertEqual(len(joined),10)
        for field in ('source','reference','prompt_sha256','output','split'):
            old=copy.deepcopy(current);old[0][field]='Changed'
            with self.assertRaises(ValueError):reuse_assessments(current,old,assessments,groups,config)
        bad=copy.deepcopy(assessments);bad[0]['text_sha256']='changed'
        with self.assertRaises(ValueError):reuse_assessments(current,current,bad,groups,config)
        bad=copy.deepcopy(current);bad[0]['copied_source']=True
        with self.assertRaises(ValueError):reuse_assessments(bad,bad,assessments,groups,config)


if __name__=='__main__':unittest.main()
