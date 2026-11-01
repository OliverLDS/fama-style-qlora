import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from fama_style.training import sft_strength_pilot as r
from test_preference_pair_pilot import fixture as prior_fixture

ROOT=Path(__file__).resolve().parents[1]


def fixture(folder):
    data,previous,dp,cp,save_old=prior_fixture(folder)
    for row in data['training_pairs']:
        row.update(source_sha256=r.prior.sha(row['source']),input_kind='original')
    previous.update(adapter_sha256={'adapter_config.json':'synthetic','adapter_model.safetensors':'synthetic'},
        packages={'torch':'synthetic'},expected_generation_config={})
    save_old();cases=[]
    for row in data['training_pairs'][:11]:
        job={k:row[k] for k in ('pair_id','anchor_group_id','example_id','paper_id','input_kind','source','prompt','source_sha256','prompt_sha256')}
        job.update(prefix_ids=[2,4],prefix_ids_sha256=r.prior.sha(json.dumps([2,4])));cases.append(job)
    payload={'pilot_data_json':dp.read_text(),'pilot_config_json':cp.read_text(),'training_cases':cases}
    config={'protocol':r.PROTOCOL,'hyperparameters':r.HYPER,'training_authorized':True,'max_runtime_seconds':1800,
        'runtime_source_sha256':r.prior.file_sha(Path(r.__file__)),'prior_runtime_sha256':r.prior.file_sha(Path(r.prior.__file__)),
        'token_helper_sha256':r.prior.file_sha(Path(r.prior.tokens.__file__)),'prior_data_sha256':r.prior.file_sha(dp),
        'prior_config_sha256':r.prior.file_sha(cp),'training_case_ids':[c['pair_id'] for c in cases],
        'initial_chosen_scores':{p['pair_id']:{'total_log_probability':-4.,'supervised_tokens':2,'mean_nll':2.} for p in data['training_pairs']}}
    for k in ('model_id','model_revision','adapter_sha256','packages','expected_generation_config','decoding','group_schedule'):config[k]=previous[k]
    new_dp=folder/'strength.json';new_cp=folder/'strength_config.json'
    def save():
        r.prior.write_json(new_dp,payload);config['dataset_sha256']=r.prior.file_sha(new_dp);r.prior.write_json(new_cp,config)
    save();return payload,config,new_dp,new_cp,save


class SFTStrengthTests(unittest.TestCase):
    def test_only_learning_rate_changes_and_every_update_is_ten_times_control(self):
        previous={k:v for k,v in r.prior.HYPER.items() if k!='beta'}
        self.assertEqual([k for k in previous if previous[k]!=r.HYPER[k]],['learning_rate'])
        for step in range(1,23):
            self.assertAlmostEqual(r.learning_rate(step),10*1e-6*(1-(step-1)/22))
        for step in (0,23,True):
            with self.assertRaises(ValueError):r.learning_rate(step)

    def test_frozen_data_group_order_and_generation_boundaries_are_enforced(self):
        changes=[lambda p,c:p.update(pilot_data_json=p['pilot_data_json']+' '),
            lambda p,c:p['training_cases'][0]['prefix_ids'].append(7),
            lambda p,c:p['training_cases'][0].update(source='Changed source'),
            lambda p,c:c.update(training_authorized=False),
            lambda p,c:c.update(max_runtime_seconds=3600),
            lambda p,c:c['initial_chosen_scores'].pop(next(iter(c['initial_chosen_scores']))),
            lambda p,c:c['group_schedule'][0]['groups'].reverse()]
        for i,change in enumerate(changes):
            with self.subTest(change=i),tempfile.TemporaryDirectory() as temp:
                payload,config,dp,cp,save=fixture(Path(temp));pairs,dev,cases,_=r.load_inputs(dp,cp)
                self.assertEqual((len(pairs),len(dev),len(cases)),(17,8,11))
                change(payload,config);save()
                with self.assertRaises(ValueError):r.load_inputs(dp,cp)

    def test_check_needs_no_ml_or_weights_and_training_requires_explicit_flag(self):
        with tempfile.TemporaryDirectory() as temp:
            _,_,dp,cp,_=fixture(Path(temp))
            code='''import sys,builtins
sys.path.insert(0,sys.argv.pop(1))
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in ('torch','transformers','peft','accelerate','bitsandbytes'):
        raise AssertionError('ML imported in CPU check')
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from fama_style.training.sft_strength_pilot import main
main(sys.argv[1:])
'''
            command=[sys.executable,'-c',code,str(ROOT/'src'),'--data',str(dp),'--config',str(cp)]
            result=subprocess.run(command+['--check'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('explicit_bounded_sft_authorization_required',result.stderr)

    def test_new_runtime_never_invokes_prior_training_or_preference_gradients(self):
        tree=ast.parse(Path(r.__file__).read_text());calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertFalse(any(isinstance(c.func,ast.Attribute) and c.func.attr in ('preference_coefficients','dpo_loss','run_training') for c in calls))
        self.assertIn('final_evaluation_frozen',Path(r.__file__).read_text())

    def test_notebook_embeds_exact_runtimes_and_matches_bounded_approval(self):
        spec=importlib.util.spec_from_file_location('build_strength',ROOT/'scripts/build_sft_strength_pilot.py')
        builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            plan={'runtime_source_sha256':r.prior.file_sha(Path(r.__file__)),'prior_runtime_sha256':r.prior.file_sha(Path(r.prior.__file__)),
                'token_helper_sha256':r.prior.file_sha(Path(r.prior.tokens.__file__)),'config_sha256':'synthetic','dataset_sha256':'synthetic'}
            approval={'user_authorized':True,'optimizer_updates':22,'maximum_runtime_seconds':1800,'dpo_training_authorized':False,
                'full_training_authorized':False,**{k:plan[k] for k in ('config_sha256','dataset_sha256','runtime_source_sha256')}}
            r.prior.write_json(folder/'plan.json',plan);r.prior.write_json(folder/'authorization.json',approval)
            path=builder.build(folder,folder/'notebook.ipynb');book=json.loads(path.read_text())
            cells=[''.join(c['source']) for c in book['cells'] if c['cell_type']=='code']
            for source in cells:compile(source,'notebook_cell','exec')
            embedded=ast.literal_eval(ast.parse(cells[1]).body[1].value)
            self.assertEqual(embedded['sft_strength_pilot.py'],Path(r.__file__).read_text())
            self.assertEqual(embedded['preference_pair_pilot.py'],Path(r.prior.__file__).read_text())
            self.assertIn('timeout=remaining()',cells[-1]);self.assertIn('--train-authorized',cells[-1])
            self.assertLess(path.stat().st_size,1_000_000)
            self.assertTrue(all(not c.get('outputs') and c.get('execution_count') is None for c in book['cells']))
            approval['optimizer_updates']=44;r.prior.write_json(folder/'authorization.json',approval)
            with self.assertRaises(ValueError):builder.build(folder,folder/'bad.ipynb')


if __name__=='__main__':unittest.main()
