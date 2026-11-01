"""Completed pilot review requires fixed stages, finite updates and bound masking."""
import copy
import unittest
from fama_style.training.teacher_pair_expanded_pilot import sha,prompt
from fama_style.research.teacher_pair_expanded_pilot_review import validate_execution,join_masked


class TeacherPairExpandedPilotReview(unittest.TestCase):
    def fixture(self):
        groups={'training':[],'development':[]}
        for split,n in [('training',46),('development',8)]:
            for i in range(n):
                source='Synthetic input '+split+str(i)
                groups[split].append({'example_id':split+str(i),'split':split,'source':source,
                    'target':'Synthetic reference '+split+str(i),'prompt':prompt(source)})
        config={'development_ids':[r['example_id'] for r in groups['development']],
                'training_probe_ids':[r['example_id'] for r in groups['training'][:2]]}
        expected=groups['development']+groups['training'][:2];snaps=[];key=[];assess=[];bundle=[];prefs=[]
        for name,step in [('base',0),('epoch_1',23),('epoch_2',46)]:
            generations=[{'example_id':r['example_id'],'split':r['split'],'source':r['source'],
                'reference':r['target'],'prompt_sha256':sha(r['prompt']),'output':name+' synthetic output.'} for r in expected]
            snaps.append({'name':name,'optimizer_step':step,'complete':True,'generations':generations,
                          'losses':[{'example_id':rid,'loss':2.,'supervised_tokens':10} for rid in config['development_ids']],
                          'token_weighted_development_loss':2.})
        for row in expected:
            rid=row['example_id'];outputs={}
            for label,snap in zip('ABC',snaps):
                text=next(r['output'] for r in snap['generations'] if r['example_id']==rid)
                outputs[label]=text;key.append({'example_id':rid,'label':label,'snapshot':snap['name']})
                assess.append({'example_id':rid,'label':label,'text_sha256':sha(text)})
            bundle.append({'example_id':rid,'split':row['split'],'source':row['source'],'reference':row['target'],'outputs':outputs})
            prefs.append({'example_id':rid,'split':row['split'],'ranking_best_first':[['A','B','C']]})
        report={'complete':True,'mode':'pilot','config':config,'training_called':True,
                'adapter_weights_changed':True,'adapter_reload_weights_match':True,'snapshots':snaps}
        state={'complete':True,'optimizer_steps':46,'training_called':True,'adapter_reload_weights_match':True}
        checks=[{'step':i,'finite':True,'norm':1.} for i in range(1,47)]
        return report,state,config,checks,groups,bundle,key,assess,prefs

    def test_execution_requires_all_steps_snapshots_and_reload(self):
        r,s,c,checks,*_=self.fixture();validate_execution(r,s,c,checks)
        for bad in (checks[:-1],checks[:-1]+[{'step':46,'finite':True,'norm':float('nan')}]):
            with self.assertRaises(ValueError):validate_execution(r,s,c,bad)
        bad=copy.deepcopy(r);bad['snapshots'][1]['optimizer_step']=9
        with self.assertRaises(ValueError):validate_execution(bad,s,c,checks)
        bad=copy.deepcopy(r);bad['adapter_reload_weights_match']=False
        with self.assertRaises(ValueError):validate_execution(bad,s,c,checks)

    def test_masked_coverage_and_exact_text_binding_required(self):
        r,s,c,checks,g,bundle,key,assess,prefs=self.fixture()
        result,mapping=join_masked(r,bundle,key,assess,prefs,g,c);self.assertEqual(len(result),30)
        for changed_key in (key[:-1],key[:-1]+[key[0]]):
            with self.assertRaises(ValueError):join_masked(r,bundle,changed_key,assess,prefs,g,c)
        changed=copy.deepcopy(bundle);changed[0]['outputs']['A']='edited'
        with self.assertRaises(ValueError):join_masked(r,changed,key,assess,prefs,g,c)
        changed=copy.deepcopy(assess);changed[0]['text_sha256']='wrong'
        with self.assertRaises(ValueError):join_masked(r,bundle,key,changed,prefs,g,c)

    def test_style_ties_allowed_but_duplicate_or_missing_labels_rejected(self):
        r,s,c,checks,g,bundle,key,assess,prefs=self.fixture()
        for ranking in ([['A','A'],['C']],[['A'],['B']], [[],['A','B','C']]):
            changed=copy.deepcopy(prefs);changed[0]['ranking_best_first']=ranking
            with self.assertRaises(ValueError):join_masked(r,bundle,key,assess,changed,g,c)


if __name__=='__main__':unittest.main()
