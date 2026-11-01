"""Baseline review keeps held-out examples and training probes distinct."""
import copy
import unittest
from fama_style.training.teacher_pair_pilot import sha,prompt
from fama_style.research.teacher_pair_baseline_review import validate_labels,bind_generations


class TeacherPairBaselineReview(unittest.TestCase):
    def fixture(self):
        groups={'training':[],'development':[]};audit=[]
        for split,n in [('training',16),('development',8)]:
            for i in range(n):
                row={'example_id':split+str(i),'split':split,'source':'Synthetic input '+str(i),
                     'target':'Synthetic reference '+str(i)}
                row['prompt']=prompt(row['source']);groups[split].append(row)
                audit.append({'example_id':row['example_id'],'split':split,'labels_match_target':True,
                              'masked_tokens':20,'supervised_tokens':10,'full_tokens':30})
        config={'development_ids':[r['example_id'] for r in groups['development']],
                'training_probe_ids':[r['example_id'] for r in groups['training'][:2]]}
        generations=[]
        for row in groups['development']+groups['training'][:2]:
            generations.append({'example_id':row['example_id'],'split':row['split'],'source':row['source'],
                'reference':row['target'],'prompt_sha256':sha(row['prompt']),'output':' Synthetic output. '})
        return groups,audit,config,generations

    def test_label_audit_exact_coverage_and_nonempty_target_required(self):
        groups,audit,config,generations=self.fixture();validate_labels(audit,groups)
        with self.assertRaises(ValueError):validate_labels(audit[:-1],groups)
        changed=copy.deepcopy(audit);changed[0]['supervised_tokens']=0
        with self.assertRaises(ValueError):validate_labels(changed,groups)

    def test_generation_binding_preserves_raw_text_and_probe_roles(self):
        groups,audit,config,generations=self.fixture()
        result=bind_generations(generations,groups,config)
        self.assertEqual(result[0]['text'],' Synthetic output. ')
        self.assertEqual([r['split'] for r in result],['development']*8+['training']*2)
        for key in ('source','reference','prompt_sha256','split'):
            changed=copy.deepcopy(generations);changed[0][key]='changed'
            with self.assertRaises(ValueError):bind_generations(changed,groups,config)
        with self.assertRaises(ValueError):bind_generations(generations[:-1],groups,config)


if __name__=='__main__':unittest.main()
