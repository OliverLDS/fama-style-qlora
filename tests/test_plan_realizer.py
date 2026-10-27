"""Matched schedules, labels, oracle leakage boundaries and CPU-only preflight."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.training import plan_realizer as e
from tests.test_corpus_baseline import FakeProcessor

class PlanRealizerTests(unittest.TestCase):
    def row(self,index=0,split='train'):
        source='Firms may invest if costs fall. Prices might change.'
        target='Firms may invest when costs decline. Prices might respond.'
        return {'id':str(index),'paper_key':'synthetic','split':split,'source':source,'reference':target,
            'source_sha256':e.sha(source),'reference_sha256':e.sha(target),'plan':e.presentation_plan(target)}

    def pack(self):
        return {'train':[self.row(i) for i in range(12)],'development':[self.row(i,'validation') for i in range(12,16)],'training_probe_ids':['0','1']}

    def reports(self):
        p=self.pack();status={'completed_arms':list(e.ARMS),'training_called':True,'updates':{a:[{'step':i,**u,'loss':2.0,'gradient_norm':0.5} for i,u in enumerate(e.schedules([r['id'] for r in p['train']]),1)] for a in e.ARMS}}
        losses=[{'model':m,'prompt_arm':a,'loss':2.0,'rows':[{'id':r['id'],'loss':2.0,'supervised_tokens':10} for r in p['development']]} for m in ('base',)+e.ARMS for a in e.ARMS]
        outputs=[{'model':m,'prompt_arm':a,'id':r['id'],'prompt_sha256':e.sha(e.prompt_for(r,a)),'output':'Synthetic result.','hit_output_limit':False,'synthetic':False} for m in ('base',)+e.ARMS for r in e.generation_rows(p) for a in e.ARMS]
        hashes={'initial':{a:{'tensor':'initial'} for a in e.ARMS},'trained':{a:{'tensor':'trained-'+a} for a in e.ARMS},'both_reload_verified':True}
        return p,status,losses,outputs,hashes

    def test_content_free_plan_and_prompt_target_exclusion(self):
        row=self.row();a=e.prompt_for(row,'direct');b=e.prompt_for(row,'planned')
        altered={**row,'reference':'PRIVATE_REFERENCE_ONLY','annotation_sha256':'PRIVATE_ANNOTATION'}
        self.assertEqual(a,e.prompt_for(altered,'direct'));self.assertEqual(b,e.prompt_for(altered,'planned'))
        self.assertTrue(a.endswith(row['source']));self.assertTrue(b.endswith(row['source']))
        bad=copy.deepcopy(row['plan']);bad['sentences'][0]['reference_words']='PRIVATE_REFERENCE'
        with self.assertRaises(ValueError):e.validate_plan(bad)
        self.assertNotIn('Firms',json.dumps(row['plan']))

    def test_identical_schedule_covers_each_example_once_per_epoch(self):
        p=self.pack();schedule=e.schedules([r['id'] for r in p['train']]);self.assertEqual(len(schedule),24)
        for epoch in range(1,5):
            keys=[i for u in schedule if u['epoch']==epoch for i in u['ids']]
            self.assertEqual(set(keys),{r['id'] for r in p['train']});self.assertEqual(len(keys),12)
        self.assertEqual(schedule,e.schedules([r['id'] for r in p['train']]))
        with self.assertRaises(ValueError):e.schedules(['duplicate']*12)

    def test_actual_labels_exclude_prompt_and_preserve_target(self):
        row=self.row();processor=FakeProcessor()
        with patch.dict(e.CONFIG,{'max_length':10000}):
            for arm in e.ARMS:
                data,audit=e.encode_pair(processor,row,arm);prefix=audit['prompt_tokens']
                self.assertEqual(data['labels'][:prefix],[-100]*prefix)
                self.assertEqual(data['labels'][prefix:],data['input_ids'][prefix:])
                self.assertEqual(' '.join(processor.decode(data['labels'][prefix:]).split()),row['reference'])
            render=processor.apply_chat_template;calls=[]
            def changed(*args,**kwargs):
                result=render(*args,**kwargs);calls.append(True)
                if len(calls)==3:result['input_ids'][0][-2]+=1
                return result
            processor.apply_chat_template=changed
            with self.assertRaises(ValueError):e.encode_pair(processor,row,'direct')
        with patch.dict(e.CONFIG,{'max_length':1}):
            with self.assertRaises(ValueError):e.encode_pair(FakeProcessor(),row,'direct')

    def test_completion_checks_fresh_adapters_schedule_loss_and_matrix(self):
        p,s,l,o,h=self.reports();e.validate_completion(p,s,l,o,h)
        for failure in ('schedule','gradient','weight_change','initialization','reload','loss','missing_generation'):
            ss,ll,oo,hh=copy.deepcopy((s,l,o,h))
            if failure=='schedule':ss['updates']['planned'][0]['ids']=['wrong']
            if failure=='gradient':ss['updates']['direct'][0]['gradient_norm']=float('nan')
            if failure=='weight_change':hh['trained']['direct']=hh['initial']['direct']
            if failure=='initialization':hh['initial']['planned']={'tensor':'different'}
            if failure=='reload':hh['both_reload_verified']=False
            if failure=='loss':ll[0]['loss']=3.0
            if failure=='missing_generation':oo.pop()
            with self.assertRaises(ValueError,msg=failure):e.validate_completion(p,ss,ll,oo,hh)

    def test_cutoff_stays_a_failed_sample_not_a_quality_measurement(self):
        row=self.row();p={'style_profile':{'unused':True}}
        out={'output':'Partial text','hit_output_limit':True}
        metric=e.output_metrics(p,row,out)
        self.assertFalse(metric['available']);self.assertIsNone(metric['source_similarity']);self.assertIsNone(metric['style_distance_gain_proxy']);self.assertEqual(metric['output'],'Partial text')

    def test_preflight_and_isolated_notebook_cannot_start_training(self):
        path=e.ROOT/'research_runs/plan_realizer_pilot_v1/input_pack/realizer_pack.json'
        if not path.exists():self.skipTest('Private pack absent in public checkout')
        pack=e.load_pack(path,e.file_sha(path));self.assertEqual(len(e.generation_rows(pack)),6)
        self.assertTrue(all(a['number_multiset_matches'] for a in pack['pair_audits']))
        with tempfile.TemporaryDirectory() as folder,patch.object(e,'run_training',side_effect=AssertionError('GPU')),patch.object(e,'secret_token',side_effect=AssertionError('Secret')):
            e.main(['--pack',str(path),'--expected-pack-sha',e.file_sha(path),'--dry-run','--output-dir',folder+'/dry'])
            self.assertFalse(json.loads(Path(folder+'/dry/preflight.json').read_text())['training_called'])
        book=json.loads((e.ROOT/'kaggle/plan_realizer_pilot.ipynb').read_text())
        for cell in book['cells']:
            self.assertFalse(cell.get('outputs'));self.assertIsNone(cell.get('execution_count'))
            if cell['cell_type']=='code' and not ''.join(cell['source']).startswith('%'):compile(''.join(cell['source']),'notebook','exec')
        with tempfile.TemporaryDirectory() as folder:
            setup=''.join(book['cells'][3]['source']).replace('/kaggle/working/_plan_realizer_runtime',folder+'/runtime')
            preflight=''.join(book['cells'][4]['source']).replace('list(Path("/kaggle/input").rglob("realizer_pack.json"))',repr([str(path)])).replace('/kaggle/working/plan_realizer_preflight',folder+'/preflight')
            disabled=''.join(book['cells'][5]['source'])
            self.assertIn('RUN_TRAINING = False',disabled)
            result=subprocess.run([sys.executable,'-I','-c',setup+'\n'+preflight+'\n'+disabled],cwd=folder,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('Training is disabled',result.stdout)

if __name__=='__main__':unittest.main()
