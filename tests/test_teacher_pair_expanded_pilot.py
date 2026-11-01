"""CPU isolation, frozen budget, label boundaries and v2 baseline gates."""
import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fama_style.training import teacher_pair_expanded_pilot as runtime
from fama_style.research.teacher_pair_expanded_packet import build

ROOT=Path(__file__).resolve().parents[1]


class ExpandedTeacherPairPilot(unittest.TestCase):
    def package(self,root):
        curated=root/'curated';curated.mkdir()
        groups={}
        for split,count in [('training',46),('development',8)]:
            groups[split]=[]
            for i in range(count):
                source=f'Neutral {split} source {i}.';target=f'Academic {split} target {i}.'
                groups[split].append({'example_id':split+str(i),'paper_id':split+'paper',
                    'author_group_id':split+'group','authors':[split+' author'],'split':split,
                    'source':source,'target':target,'source_sha256':runtime.sha(source),
                    'target_sha256':runtime.sha(target),'prompt':runtime.prompt(source),'pilot_eligible':True})
            (curated/(split+'.private.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in groups[split]))
        runtime.write_json(curated/'lineage.json',{
            'preserved_development_ids':[r['example_id'] for r in groups['development']],
            'preserved_training_probe_ids':['training0','training1']})
        runtime.write_json(curated/'manifest.json',{'files':{
            p.name:runtime.sha(p.read_text()) for p in curated.iterdir() if p.is_file()}})
        output=root/'out';build(curated,output)
        return output,output/'upload/teacher_pair_expanded.jsonl',output/'upload/teacher_pair_expanded_config.json'

    def test_cpu_check_and_training_gate_and_complete_packet_reproduction(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out,data,path=self.package(root)
            with patch.object(runtime,'run_training',side_effect=AssertionError('No model/GPU')):
                with contextlib.redirect_stdout(io.StringIO()):
                    runtime.main(['--data',str(data),'--config',str(path),'--check'])
                with self.assertRaises(ValueError):
                    runtime.main(['--data',str(data),'--config',str(path),'--mode','pilot'])
            build(root/'curated',root/'replica')
            for file in out.rglob('*'):
                if file.is_file():self.assertEqual(file.read_bytes(),(root/'replica'/file.relative_to(out)).read_bytes())
            with self.assertRaises(ValueError):build(root/'curated',out)
            plan=json.loads((out/'plan.json').read_text())
            self.assertEqual(plan['optimizer_updates'],46)
            self.assertFalse(plan['training_authorized'])

    def test_rehashed_wrong_counts_author_leak_and_budget_are_rejected(self):
        for issue in ('count','author','budget','authorization'):
            with tempfile.TemporaryDirectory() as folder:
                out,data,path=self.package(Path(folder));records=[json.loads(l) for l in data.read_text().splitlines()]
                config=json.loads(path.read_text())
                if issue=='count':records.pop(0)
                elif issue=='author':records[-1]['authors']=records[0]['authors']
                elif issue=='budget':config['hyperparameters']['optimizer_steps']=16
                else:config['gpu_execution_authorized']=True
                data.write_text(''.join(json.dumps(r)+'\n' for r in records))
                config['dataset_sha256']=runtime.sha(data.read_text());runtime.write_json(path,config)
                with self.assertRaises(ValueError):runtime.load_inputs(data,path)

    def test_fresh_baseline_requires_all_label_checks_and_current_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            out,data,path=self.package(Path(folder));config=json.loads(path.read_text())
            report={'complete':True,'mode':'baseline','config':config,'label_preflight_passed':True,
                'label_preflight_rows':54,'runtime':{'model_revision':runtime.REVISION,
                    'runtime_source_sha256':runtime.sha(Path(runtime.__file__).read_text())},
                'snapshots':[{'complete':True,'name':'base','optimizer_step':0,
                    'losses':[{'example_id':rid} for rid in config['development_ids']],
                    'generations':[{'example_id':rid} for rid in config['development_ids']+config['training_probe_ids']],
                    'token_weighted_development_loss':2.9}]}
            baseline=out/'baseline.json';runtime.write_json(baseline,report)
            runtime.validate_baseline(baseline,config)
            for key,value in [('label_preflight_rows',24),('label_preflight_passed',False),('runtime',{})]:
                altered=dict(report);altered[key]=value;runtime.write_json(baseline,altered)
                with self.assertRaises(ValueError):runtime.validate_baseline(baseline,config)

    def test_target_labels_never_truncate_or_supervise_prompt(self):
        self.assertEqual(runtime.mask_labels([1,2],[1,2,3],'Target.','Target.'),[-100,-100,3])
        for args in (([1],[1]*1025,'Target.','Target.'),([1,2],[1,4,3],'Target.','Target.'),
                     ([1],[1,2],'Wrong.','Target.')):
            with self.assertRaises(ValueError):runtime.mask_labels(*args)

    def test_clean_self_contained_notebooks_match_authorized_scope_and_runtime(self):
        source=Path(runtime.__file__).read_text()
        for mode in ('base','pilot'):
            path=ROOT/'kaggle'/('teacher_pair_expanded_'+mode+'_4b.ipynb')
            book=json.loads(path.read_text());text='\n'.join(''.join(c['source']) for c in book['cells'])
            self.assertIn(source,text);self.assertLess(path.stat().st_size,1_000_000)
            launch=''.join(book['cells'][-1]['source']);ast.parse(launch)
            if mode=='pilot':
                self.assertIn('RUN_PILOT = True',launch)
                authorization=json.loads((ROOT/'docs/teacher_pair_pilot_v2/pilot_execution.json').read_text())
                tree=ast.parse(launch)
                expected=next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=='expected_baseline_sha256' for t in n.targets))
                self.assertEqual(expected,authorization['baseline_evaluation_sha256'])
                self.assertLess(launch.index('hashlib.sha256(baseline[0].read_bytes())'),launch.index("'--train-authorized'"))
            self.assertLess(launch.index('hashlib.sha256(path.read_bytes())'),launch.index("command="))
            for cell in book['cells']:
                if cell['cell_type']=='code':
                    self.assertIsNone(cell['execution_count']);self.assertEqual(cell['outputs'],[])
        tree=ast.parse(source)
        for node in tree.body:
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module]
                self.assertFalse(any(n.split('.')[0] in ('torch','transformers','peft','datasets') for n in names))

    def test_authorization_rejects_scope_and_hash_changes(self):
        import copy
        import importlib.util
        path=ROOT/'scripts/build_teacher_pair_expanded_pilot.py'
        spec=importlib.util.spec_from_file_location('expanded_builder',path)
        builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
        plan=json.loads((ROOT/'docs/teacher_pair_pilot_v2/experiment_plan.json').read_text())
        authorization=json.loads((ROOT/'docs/teacher_pair_pilot_v2/pilot_execution.json').read_text())
        source=Path(runtime.__file__).read_text()
        with tempfile.TemporaryDirectory() as folder:
            packet=Path(folder);(packet/'manifest.json').write_text('{}')
            authorization['packet_manifest_sha256']=runtime.sha('{}')
            builder.validate_authorization(plan,authorization,source,packet)
            for key,value in [('optimizer_steps',47),('training_rows',16),('full_training_authorized',True),
                              ('runtime_source_sha256','wrong'),('dataset_sha256','wrong'),
                              ('baseline_evaluation_sha256','missing'),('packet_manifest_sha256','wrong')]:
                altered=copy.deepcopy(authorization);altered[key]=value
                with self.assertRaises(ValueError):builder.validate_authorization(plan,altered,source,packet)


if __name__=='__main__':unittest.main()
