"""Matched targets/order/budget and pre-gradient control guards, without ML imports."""
import ast
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fama_style.research.teacher_pair_input_replacement_packet import replace_inputs
from fama_style.training import teacher_pair_input_replacement as runtime
from fama_style.training import teacher_pair_expanded_pilot as control_runtime

ROOT=Path(__file__).resolve().parents[1]


class InputReplacement(unittest.TestCase):
    def fixture(self,root):
        records=[];anchors=[]
        for split,n in [('training',46),('development',8)]:
            for i in range(n):
                rid=split+str(i);source=f'Original {split} source {i}.';target=f'Human {split} target {i}.'
                row={'example_id':rid,'split':split,'paper_id':split+'paper','author_group_id':split+'group',
                    'authors':[split+' author'],'source':source,'target':target,'source_sha256':runtime.sha(source),
                    'target_sha256':runtime.sha(target),'prompt':runtime.prompt(source),'pilot_eligible':True}
                records.append(row)
                if split=='training' and i<12:
                    variant=f'Variant training source {i}.'
                    anchors.append({'example_id':rid,'split':'training','family':('scope','time','numbers','reasoning')[i//3],
                        'source':source,'chosen':target,'faithful_variant':variant,'source_sha256':row['source_sha256'],
                        'target_sha256':row['target_sha256'],'variant_sha256':runtime.sha(variant),
                        'review':{'notes':'Synthetic accepted construction.'},'original_curation':{'original_needs_review':False}})
        changed,audit=replace_inputs(records,anchors)
        control={'protocol':'teacher_pair_reverse_pilot_v2','model_id':runtime.MODEL_ID,'model_revision':runtime.REVISION,
            'hyperparameters':runtime.HYPER,'decoding':runtime.DECODING,'quantization':'NF4 double quantization; FP32 compute',
            'dataset_sha256':runtime.sha('original dataset'),'gpu_execution_authorized':False,
            'training_ids':[r['example_id'] for r in records if r['split']=='training'],
            'development_ids':[r['example_id'] for r in records if r['split']=='development'],
            'training_probe_ids':['training44','training45']}
        data=root/'replacement.jsonl';data.write_text(''.join(json.dumps(r)+'\n' for r in changed))
        config=copy.deepcopy(control);config.update(protocol=runtime.PROTOCOL,dataset_sha256=runtime.sha(data.read_text()),
            control_config=copy.deepcopy(control),control_runtime_sha256=runtime.sha('frozen control runtime'),
            replaced_training_ids=[a['example_id'] for a in anchors],fresh_adapter_required=True)
        selected=[r for r in records if r['example_id'] in control['development_ids']+control['training_probe_ids']]
        selected.sort(key=lambda r:(r['split']!='development',control['development_ids'].index(r['example_id']) if r['split']=='development' else control['training_probe_ids'].index(r['example_id'])))
        snap={'name':'base','complete':True,'optimizer_step':0,'token_weighted_development_loss':2.,
            'generations':[{'example_id':r['example_id'],'source':r['source'],'reference':r['target'],'output':'Synthetic output.'} for r in selected],
            'losses':[{'example_id':rid,'supervised_tokens':10,'loss':2.} for rid in control['development_ids']]}
        baseline={'mode':'baseline','complete':True,'training_called':False,'config':control,'label_preflight_passed':True,
            'label_preflight_rows':54,'runtime':{'model_revision':runtime.REVISION,'runtime_source_sha256':config['control_runtime_sha256']},'snapshots':[snap]}
        base=root/'baseline.json';base.write_text(json.dumps(baseline));config['control_baseline_sha256']=runtime.sha(base.read_text())
        cfg=root/'config.json';cfg.write_text(json.dumps(config))
        return records,anchors,changed,audit,data,cfg,base,config,baseline

    def test_exactly_twelve_inputs_change_without_targets_order_or_metadata_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            original,anchors,changed,audit,*_=self.fixture(Path(folder))
            self.assertEqual(len(audit),12)
            self.assertEqual(sum(a['source']!=b['source'] for a,b in zip(original,changed)),12)
            for a,b in zip(original,changed):
                for k in a:
                    if k not in ('source','source_sha256','prompt'):self.assertEqual(a[k],b[k])
                if a['split']=='development':self.assertEqual(a,b)
            bad=copy.deepcopy(anchors);bad[0]['chosen']='Changed human target'
            with self.assertRaises(ValueError):replace_inputs(original,bad)
            with self.assertRaises(ValueError):replace_inputs(original,anchors[:-1])

    def test_cpu_check_and_explicit_training_gate_use_no_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            *_,data,cfg,base,config,baseline=self.fixture(Path(folder))
            runtime.load_inputs(data,cfg);runtime.validate_baseline(base,config)
            with patch.object(runtime,'run_training',side_effect=AssertionError('No GPU')):
                with contextlib.redirect_stdout(io.StringIO()):runtime.main(['--data',str(data),'--config',str(cfg),'--check'])
                with self.assertRaises(ValueError):runtime.main(['--data',str(data),'--config',str(cfg),'--mode','pilot'])
            bad=copy.deepcopy(config);bad['control_config']['hyperparameters']['learning_rate']=9e-5
            with self.assertRaises(ValueError):runtime.validate_control_config(bad)
            bad=copy.deepcopy(config);bad['replaced_training_ids'][0]='training44'
            with self.assertRaises(ValueError):runtime.validate_control_config(bad)
            base.write_text(base.read_text()+'\n')
            with self.assertRaises(ValueError):runtime.validate_baseline(base,config)

    def test_common_base_must_match_before_training(self):
        with tempfile.TemporaryDirectory() as folder:
            *_,baseline=self.fixture(Path(folder));snap=copy.deepcopy(baseline['snapshots'][0])
            self.assertTrue(runtime.compare_base(snap,baseline)['passed'])
            changed=copy.deepcopy(snap);changed['losses'][0]['loss']+=5e-7
            runtime.compare_base(changed,baseline)
            for kind in ('output','loss','tokens'):
                changed=copy.deepcopy(snap)
                if kind=='output':changed['generations'][0]['output']='Different'
                elif kind=='loss':changed['losses'][0]['loss']+=2e-6
                else:changed['losses'][0]['supervised_tokens']=11
                with self.assertRaises(ValueError):runtime.compare_base(changed,baseline)

    def test_training_arguments_and_lora_configuration_match_control(self):
        self.assertEqual(runtime.HYPER,control_runtime.HYPER);self.assertEqual(runtime.DECODING,control_runtime.DECODING)
        source=Path(runtime.__file__).read_text();old=Path(control_runtime.__file__).read_text()
        def calls(text,name):
            return [ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(text)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id==name]
        for name in ('TrainingArguments','LoraConfig'):self.assertEqual(calls(source,name),calls(old,name))
        self.assertLess(source.index("report['control_base_check']=compare_base"),source.index('model=get_peft_model'))
        self.assertIn('fresh_adapter_requires_zero_initialized_B_matrices',source)
        for n in ast.parse(source).body:
            if isinstance(n,(ast.Import,ast.ImportFrom)):
                names=[a.name for a in n.names] if isinstance(n,ast.Import) else [n.module]
                self.assertFalse(any(name.split('.')[0] in ('torch','transformers','peft') for name in names))

    def test_approved_notebook_pins_three_files_and_rejects_scope_changes(self):
        source=Path(runtime.__file__).read_text();path=ROOT/'kaggle/input_replacement_4b.ipynb'
        book=json.loads(path.read_text());self.assertIn(source,'\n'.join(''.join(c['source']) for c in book['cells']))
        self.assertLess(path.stat().st_size,1_000_000)
        launch=''.join(book['cells'][-1]['source']);tree=ast.parse(launch)
        self.assertIn('RUN_PILOT = True',launch)
        self.assertLess(launch.index('hashlib.sha256(path.read_bytes())'),launch.index("'--train-authorized'"))
        auth=json.loads((ROOT/'docs/input_replacement_v1/execution.json').read_text())
        expected=next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='expected' for t in n.targets))
        self.assertEqual(expected,{k:auth[k] for k in ('dataset_sha256','config_sha256','baseline_evaluation_sha256')})
        spec=importlib.util.spec_from_file_location('replacement_builder',ROOT/'scripts/build_input_replacement_4b.py')
        builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
        plan=json.loads((ROOT/'docs/input_replacement_v1/experiment_plan.json').read_text())
        builder.validate_authorization(plan,auth,source,auth['packet_manifest_sha256'])
        for key,value in [('optimizer_steps',47),('fresh_adapter_from_base',False),('dataset_sha256','wrong'),('full_training_authorized',True)]:
            changed=copy.deepcopy(auth);changed[key]=value
            with self.assertRaises(ValueError):builder.validate_authorization(plan,changed,source,auth['packet_manifest_sha256'])
        for cell in book['cells']:
            if cell['cell_type']=='code':self.assertIsNone(cell['execution_count']);self.assertEqual(cell['outputs'],[])


if __name__=='__main__':unittest.main()
