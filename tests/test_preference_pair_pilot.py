import ast
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from fama_style.training import preference_pair_pilot as p

ROOT=Path(__file__).resolve().parents[1]


def fixture(directory):
    pairs=[]
    def seq():return {'input_ids':[2,4,7,106],'labels':[-100,-100,7,106]}
    for i in range(17):
        row={'pair_id':'pair_'+str(i),'anchor_group_id':'group_'+str(i%11),'paper_id':'train_'+str(i%10),
            'example_id':'example_'+str(i%11),'source':'Synthetic source '+str(i),'split':'training',
            'chosen':'Synthetic chosen','rejected':'Synthetic rejected','encoded':{'chosen':seq(),'rejected':seq()}}
        row['prompt']=p.prompt(row['source'])
        for k in ('prompt','chosen','rejected'):row[k+'_sha256']=p.sha(row[k])
        row['token_audits']={k:{'input_ids_sha256':p.sha(json.dumps(v['input_ids'])),
            'labels_sha256':p.sha(json.dumps(v['labels']))} for k,v in row['encoded'].items()}
        row['anchor_balanced_weight']=1/(2 if i%11<6 else 1);pairs.append(row)
    dev=[]
    for i in range(8):
        row={'example_id':'dev_'+str(i),'paper_id':'dev_paper_'+str(i%4),'source':'Synthetic unseen source '+str(i),
            'target':'Synthetic reference hidden from generator','split':'development','encoded':seq(),
            'prefix_ids':[2,4],'prefix_ids_sha256':p.sha(json.dumps([2,4]))}
        row['prompt']=p.prompt(row['source'])
        for k in ('source','target','prompt'):row[k+'_sha256']=p.sha(row[k])
        dev.append(row)
    data={'training_pairs':pairs,'development':dev}
    config={'protocol':p.PROTOCOL,'model_id':p.MODEL_ID,'model_revision':p.REVISION,
        'hyperparameters':p.HYPER,'decoding':p.DECODING,'training_authorized':False,'max_runtime_seconds':3600,
        'token_helper_sha256':p.file_sha(Path(p.tokens.__file__)),'runtime_source_sha256':p.file_sha(Path(p.__file__)),
        'development_ids':[r['example_id'] for r in dev],'group_schedule':p.schedule(pairs)}
    data_path=directory/'data.json';config_path=directory/'config.json'
    def save():
        p.write_json(data_path,data);config['dataset_sha256']=p.file_sha(data_path);p.write_json(config_path,config)
    save()
    return data,config,data_path,config_path,save


class PreferencePilotTests(unittest.TestCase):
    def test_dpo_two_pass_coefficients_equal_joint_numerical_gradient(self):
        # Three shared toy parameters affect both response log probabilities.
        # Two sibling rows are averaged before the single optimizer update.
        theta=[0.2,-0.1,0.3]
        examples=[([-2.,1.,-.4],[.3,-1.,2.],-14.,-15.),([1.,-.2,2.],[-.4,1.,.1],-16.,-14.)]
        def scores(t,a,b):return -20+sum(x*y for x,y in zip(t,a)),-21+sum(x*y for x,y in zip(t,b))
        def loss(t):
            return sum(p.tokens.dpo_loss(*scores(t,a,b),rc,rr,.1) for a,b,rc,rr in examples)/2
        sequential=[0.,0.,0.]
        for a,b,rc,rr in examples:
            coeff=p.preference_coefficients(*scores(theta,a,b),rc,rr,.1)
            for j in range(3):sequential[j]+=(coeff['chosen_gradient']*a[j]+coeff['rejected_gradient']*b[j])/2
        for j in range(3):
            plus=theta[:];minus=theta[:];plus[j]+=1e-5;minus[j]-=1e-5
            self.assertAlmostEqual(sequential[j],(loss(plus)-loss(minus))/2e-5,places=8)

    def test_identity_and_extreme_margins_stay_finite(self):
        self.assertAlmostEqual(p.preference_coefficients(-12,-13,-12,-13)['loss'],math.log(2))
        for pc,pr in [(-1.,-100000.),(-100000.,-1.)]:
            values=p.preference_coefficients(pc,pr,-10.,-11.)
            self.assertTrue(all(math.isfinite(v) for v in values.values()))
            self.assertLessEqual(values['chosen_gradient'],0)
            self.assertGreaterEqual(values['rejected_gradient'],0)

    def test_group_schedule_balances_siblings_and_repeats_exact_exposure(self):
        with tempfile.TemporaryDirectory() as temp:
            data,_,_,_,_=fixture(Path(temp));rows=data['training_pairs'];schedule=p.schedule(rows)
            self.assertEqual(schedule,p.schedule(rows))
            for epoch in schedule:
                self.assertEqual(len(epoch['groups']),11)
                self.assertCountEqual([pid for g in epoch['groups'] for pid in g['pair_ids']],[r['pair_id'] for r in rows])
                for g in epoch['groups']:
                    self.assertEqual(sum(r['anchor_balanced_weight'] for r in rows if r['pair_id'] in g['pair_ids']),1)
            self.assertNotEqual(schedule[0]['groups'],schedule[1]['groups'])

    def test_masks_reject_padding_or_noncontiguous_supervision(self):
        self.assertEqual(p.validate_sequence({'input_ids':[2,4,7,106],'labels':[-100,-100,7,106]}),(2,2))
        for labels in ([2,4,7,106],[-100,4,-100,106],[-100,-100,-100,-100],[-100,-100,7,0]):
            with self.assertRaises(ValueError):p.validate_sequence({'input_ids':[2,4,7,106],'labels':labels})

    def test_loader_rejects_overlap_weights_token_and_data_mutations(self):
        mutations=[lambda d:d['development'][0].update(paper_id=d['training_pairs'][0]['paper_id']),
            lambda d:d['training_pairs'][0].update(anchor_balanced_weight=1.),
            lambda d:d['training_pairs'][0]['encoded']['chosen']['input_ids'].__setitem__(2,8),
            lambda d:d['development'][0]['prefix_ids'].__setitem__(0,3)]
        for change in mutations:
            with tempfile.TemporaryDirectory() as temp:
                data,config,dp,cp,save=fixture(Path(temp));p.load_inputs(dp,cp)
                change(data);save()
                with self.assertRaises(ValueError):p.load_inputs(dp,cp)
        with tempfile.TemporaryDirectory() as temp:
            _,_,dp,cp,_=fixture(Path(temp));dp.write_text(dp.read_text()+' ')
            with self.assertRaises(ValueError):p.load_inputs(dp,cp)

    def test_generation_messages_never_contain_reference_wording(self):
        row={'prompt':'Synthetic source prompt','target':'SECRET_REFERENCE','expected_generation':{'output':'SECRET_OUTPUT'},
            'encoded':{'input_ids':[2,3,4]}}
        self.assertEqual(p.messages(row),[{'role':'user','content':[{'type':'text','text':'Synthetic source prompt'}]}])

    def test_check_and_missing_authorization_work_without_ml_imports(self):
        with tempfile.TemporaryDirectory() as temp:
            _,_,dp,cp,_=fixture(Path(temp))
            code='''import builtins,sys
sys.path.insert(0,sys.argv.pop(1))
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in ('torch','transformers','peft','accelerate','bitsandbytes'):
        raise AssertionError('ML import during CPU check')
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from fama_style.training.preference_pair_pilot import main
main(sys.argv[1:])
'''
            command=[sys.executable,'-c',code,str(ROOT/'src'),'--data',str(dp),'--config',str(cp)]
            result=subprocess.run(command+['--check'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('explicit_bounded_training_authorization_required',result.stderr)

    def test_notebook_default_cpu_and_embedded_runtimes_match(self):
        spec=importlib.util.spec_from_file_location('build_preference_pair_pilot',ROOT/'scripts/build_preference_pair_pilot.py')
        builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);packet=directory/'packet';packet.mkdir()
            p.write_json(packet/'plan.json',{'runtime_source_sha256':p.file_sha(Path(p.__file__)),
                'token_helper_sha256':p.file_sha(Path(p.tokens.__file__)),'data_sha256':'dummy','config_sha256':'dummy'})
            output=builder.build(packet,directory/'pilot.ipynb');book=json.loads(output.read_text())
            sources=[''.join(c['source']) for c in book['cells'] if c['cell_type']=='code']
            for source in sources:compile(source,'notebook_cell','exec')
            tree=ast.parse(sources[1]);embedded=ast.literal_eval(tree.body[1].value)
            self.assertEqual(embedded['preference_pair_pilot.py'],Path(p.__file__).read_text())
            self.assertIn('RUN_APPROVED_PILOT=False',sources[0])
            self.assertIn('timeout=remaining()',sources[-1])
            self.assertLess(output.stat().st_size,1_000_000)
            self.assertTrue(all(not c.get('outputs') and c.get('execution_count') is None for c in book['cells']))
            p.write_json(packet/'manifest.json',{'files':{}})
            approval=directory/'approval.json'
            record={'explicit_user_approval':True,'total_updates':44,'maximum_runtime_seconds':3600,
                'visible_gpu_count':1,'packet_manifest_sha256':p.file_sha(packet/'manifest.json')}
            p.write_json(approval,record)
            approved=builder.build(packet,directory/'approved.ipynb',approval)
            approved_book=json.loads(approved.read_text())
            self.assertIn('RUN_APPROVED_PILOT=True',''.join(approved_book['cells'][1]['source']))
            record['total_updates']=45;p.write_json(approval,record)
            with self.assertRaises(ValueError):builder.build(packet,directory/'invalid.ipynb',approval)


if __name__=='__main__':unittest.main()
