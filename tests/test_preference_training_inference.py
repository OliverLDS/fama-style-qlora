import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from fama_style.evaluation import preference_training_inference as r

ROOT=Path(__file__).resolve().parents[1]


def fixture(folder):
    jobs=[]
    for i in range(11):
        source='Synthetic economics source '+str(i)
        job={'pair_id':'pair_'+str(i),'anchor_group_id':'group_'+str(i),'example_id':'example_'+str(i),
            'paper_id':'paper_'+str(i%10),'input_kind':'original' if i<9 else 'variant','family':'scope',
            'source':source,'prompt':r.prompt(source),'source_sha256':r.sha(source),'prompt_sha256':r.sha(r.prompt(source)),
            'prefix_ids':[2,7,106,8],'prefix_ids_sha256':r.sha(json.dumps([2,7,106,8]))}
        jobs.append(job)
    config={'protocol':r.PROTOCOL,'model_id':r.MODEL_ID,'model_revision':r.REVISION,'decoding':r.DECODING,
        'arms':list(r.ARMS),'seed':1729,'max_length':1024,'maximum_runtime_seconds':1800,'training_authorized':False,
        'optimizer_steps':0,'runtime_source_sha256':r.file_sha(Path(r.__file__)),'pair_ids':[x['pair_id'] for x in jobs],
        'adapter_sha256':{a:{'adapter_config.json':'fixture','adapter_model.safetensors':'fixture'} for a in r.ARMS}}
    jp=folder/'jobs.jsonl';cp=folder/'config.json'
    def save():
        jp.write_text(''.join(json.dumps(j)+'\n' for j in jobs));config['jobs_sha256']=r.file_sha(jp);r.write_json(cp,config)
    save();return jobs,config,jp,cp,save


class PreferenceTrainingInferenceTests(unittest.TestCase):
    def test_selection_prefers_original_without_using_output_or_input_order(self):
        rows=[{'pair_id':'V2','anchor_group_id':'g2','input_kind':'variant'},
              {'pair_id':'V1','anchor_group_id':'g1','input_kind':'variant'},
              {'pair_id':'O1','anchor_group_id':'g1','input_kind':'original'}]
        self.assertEqual([x['pair_id'] for x in r.select_rows(rows)],['O1','V2'])
        self.assertEqual(r.select_rows(rows),r.select_rows(list(reversed(rows))))
        rows[2]['output']='An intentionally poor synthetic response'
        self.assertEqual([x['pair_id'] for x in r.select_rows(rows)],['O1','V2'])

    def test_loader_rejects_reference_leakage_prefix_corruption_and_training(self):
        changes=[lambda jobs,c:jobs[0].update(chosen='Hidden target'),
            lambda jobs,c:jobs[0].update(rejected='Archived control'),
            lambda jobs,c:jobs[0]['prefix_ids'].__setitem__(1,9),
            lambda jobs,c:jobs[0].update(prompt=jobs[0]['prompt']+' changed'),
            lambda jobs,c:jobs[1].update(anchor_group_id=jobs[0]['anchor_group_id']),
            lambda jobs,c:c.update(training_authorized=True),lambda jobs,c:c.update(optimizer_steps=1),
            lambda jobs,c:c.update(maximum_runtime_seconds=3600)]
        for i,change in enumerate(changes):
            with self.subTest(change=i),tempfile.TemporaryDirectory() as temp:
                jobs,c,jp,cp,save=fixture(Path(temp));r.load_inputs(jp,cp);change(jobs,c);save()
                with self.assertRaises(ValueError):r.load_inputs(jp,cp)

    def test_generation_messages_exclude_reference_even_if_present_elsewhere(self):
        row={'prompt':'Synthetic input','chosen':'Do not leak this','rejected':'Do not leak this either'}
        self.assertEqual(r.messages(row),[{'role':'user','content':[{'type':'text','text':'Synthetic input'}]}])

    def test_check_has_no_ml_imports_and_execution_requires_explicit_flag(self):
        with tempfile.TemporaryDirectory() as temp:
            _,_,jp,cp,_=fixture(Path(temp))
            program='''import builtins,sys
sys.path.insert(0,sys.argv.pop(1))
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in ('torch','transformers','peft','accelerate','bitsandbytes'):
        raise AssertionError('ML dependency imported by check')
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from fama_style.evaluation.preference_training_inference import main
main(sys.argv[1:])
'''
            command=[sys.executable,'-c',program,str(ROOT/'src'),'--jobs',str(jp),'--config',str(cp)]
            result=subprocess.run(command+['--check'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('explicit_inference_flag_two_adapters_and_bounded_runtime_required',result.stderr)

    def test_adapter_file_mutation_rejected_without_loading_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            r.write_json(folder/'adapter_config.json',{'base_model_name_or_path':r.MODEL_ID,'r':8,'lora_alpha':16,'lora_dropout':.05})
            (folder/'adapter_model.safetensors').write_bytes(b'synthetic weights, not a real model')
            expected={p.name:r.file_sha(p) for p in folder.iterdir()};r.verify_adapter(folder,expected)
            (folder/'adapter_model.safetensors').write_bytes(b'changed')
            with self.assertRaises(ValueError):r.verify_adapter(folder,expected)

    def test_self_contained_notebook_compiles_and_has_zero_training_calls(self):
        spec=importlib.util.spec_from_file_location('build_training_inference',ROOT/'scripts/build_preference_training_inference.py')
        builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);r.write_json(folder/'plan.json',{'runtime_source_sha256':r.file_sha(Path(r.__file__)),
                'optimizer_steps':0,'training_authorized':False,'jobs_sha256':'synthetic','config_sha256':'synthetic'})
            book=json.loads(builder.build(folder,folder/'test.ipynb').read_text())
            cells=[''.join(c['source']) for c in book['cells'] if c['cell_type']=='code']
            for source in cells:compile(source,'notebook_cell','exec')
            tree=ast.parse(cells[1]);embedded=ast.literal_eval(tree.body[1].value.args[0])
            self.assertEqual(embedded,Path(r.__file__).read_text())
            self.assertIn('timeout=remaining()',cells[-1]);self.assertIn('--run-inference',cells[-1])
            self.assertNotIn('--train-authorized',cells[-1])
            self.assertTrue(all(not c.get('outputs') and c.get('execution_count') is None for c in book['cells']))
            calls=[n for n in ast.walk(ast.parse(embedded)) if isinstance(n,ast.Call)]
            self.assertFalse(any(isinstance(c.func,ast.Attribute) and c.func.attr in ('backward','step','save_pretrained') for c in calls))


if __name__=='__main__':unittest.main()
