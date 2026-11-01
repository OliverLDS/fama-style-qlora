import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from fama_style.training import dpo_strength_pilot as r,sft_strength_pilot as sft
from test_sft_strength_pilot import fixture as sft_fixture

ROOT=Path(__file__).resolve().parents[1]


def fixture(folder):
    payload,c,dp,cp,save=sft_fixture(folder)
    c.update(protocol=r.PROTOCOL,hyperparameters=r.HYPER,runtime_source_sha256=r.prior.file_sha(Path(r.__file__)),
        reference_scores={pid:{'chosen':dict(score),'rejected':dict(score)} for pid,score in c['initial_chosen_scores'].items()})
    save();return payload,c,dp,cp,save


def builder_module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class DPOStrengthTests(unittest.TestCase):
    def test_matches_SFT_schedule_and_keeps_fixed_original_reference_settings(self):
        self.assertEqual({k:v for k,v in r.HYPER.items() if k!='beta'},sft.HYPER)
        self.assertEqual(r.HYPER['beta'],.1);self.assertEqual(r.prior.HYPER['learning_rate'],1e-6)
        for step in range(1,23):self.assertEqual(r.learning_rate(step),sft.learning_rate(step))

    def test_reference_cache_scores_masks_and_beta_are_validated(self):
        changes=[lambda p,c:c['reference_scores'].pop(next(iter(c['reference_scores']))),
            lambda p,c:c['reference_scores'][next(iter(c['reference_scores']))]['rejected'].update(supervised_tokens=3),
            lambda p,c:c['reference_scores'][next(iter(c['reference_scores']))]['chosen'].update(total_log_probability=float('nan')),
            lambda p,c:c.update(hyperparameters={**r.HYPER,'beta':.2}),
            lambda p,c:p['training_cases'][0]['prefix_ids'].append(106)]
        for i,change in enumerate(changes):
            with self.subTest(change=i),tempfile.TemporaryDirectory() as temp:
                payload,c,dp,cp,save=fixture(Path(temp));pairs,dev,cases,_=r.load_inputs(dp,cp)
                self.assertEqual((len(pairs),len(dev),len(cases)),(17,8,11));change(payload,c);save()
                with self.assertRaises(ValueError):r.load_inputs(dp,cp)

    def test_check_without_ml_and_execution_flag_is_required(self):
        with tempfile.TemporaryDirectory() as temp:
            _,_,dp,cp,_=fixture(Path(temp))
            code='''import sys,builtins
sys.path.insert(0,sys.argv.pop(1))
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in ('torch','transformers','peft','accelerate','bitsandbytes'):
        raise AssertionError('ML imported during CPU check')
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from fama_style.training.dpo_strength_pilot import main
main(sys.argv[1:])
'''
            command=[sys.executable,'-c',code,str(ROOT/'src'),'--data',str(dp),'--config',str(cp)]
            check=subprocess.run(command+['--check'],capture_output=True,text=True)
            self.assertEqual(check.returncode,0,check.stderr)
            missing=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(missing.returncode,0)
            self.assertIn('explicit_bounded_dpo_authorization_required',missing.stderr)

    def test_derivation_reproduces_module_and_rejects_modified_SFT_source(self):
        builder=builder_module('build_dpo_strength_runtime');source=Path(sft.__file__).read_text()
        self.assertEqual(builder.derive(source),Path(r.__file__).read_text())
        with self.assertRaises(ValueError):builder.derive(source+'\n')
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'dpo.py';builder.build(output)
            self.assertEqual(output.read_text(),Path(r.__file__).read_text())
            output.write_text('modified')
            with self.assertRaises(ValueError):builder.build(output)

    def test_notebook_embeds_exact_code_and_rejects_broader_authorization(self):
        builder=builder_module('build_dpo_strength_pilot')
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            plan={'runtime_source_sha256':r.prior.file_sha(Path(r.__file__)),'prior_runtime_sha256':r.prior.file_sha(Path(r.prior.__file__)),
                'token_helper_sha256':r.prior.file_sha(Path(r.prior.tokens.__file__)),'dataset_sha256':'synthetic','config_sha256':'synthetic'}
            approval={'user_authorized':True,'dpo_training_authorized':True,'new_SFT_training_authorized':False,
                'full_training_authorized':False,'optimizer_updates':22,'maximum_runtime_seconds':1800,
                **{k:plan[k] for k in ('dataset_sha256','config_sha256','runtime_source_sha256')}}
            r.prior.write_json(folder/'plan.json',plan);r.prior.write_json(folder/'authorization.json',approval)
            output=builder.build(folder,folder/'notebook.ipynb');book=json.loads(output.read_text())
            sources=[''.join(c['source']) for c in book['cells'] if c['cell_type']=='code']
            for source in sources:compile(source,'notebook_cell','exec')
            embedded=ast.literal_eval(ast.parse(sources[1]).body[1].value)
            self.assertEqual(embedded['dpo_strength_pilot.py'],Path(r.__file__).read_text())
            self.assertIn('--train-authorized',sources[-1]);self.assertIn('timeout=remaining()',sources[-1])
            self.assertLess(output.stat().st_size,1_000_000)
            self.assertTrue(all(not c.get('outputs') and c.get('execution_count') is None for c in book['cells']))
            approval['new_SFT_training_authorized']=True;r.prior.write_json(folder/'authorization.json',approval)
            with self.assertRaises(ValueError):builder.build(folder,folder/'bad.ipynb')


if __name__=='__main__':unittest.main()
