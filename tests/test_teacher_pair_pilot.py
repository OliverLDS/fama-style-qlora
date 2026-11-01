"""CPU checks for reverse-pair isolation, target masks and training authorization."""
import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from fama_style.training import teacher_pair_pilot as runtime
from fama_style.research.teacher_pair_selection import build

ROOT=Path(__file__).resolve().parents[1]


class TeacherPairPilot(unittest.TestCase):
    def fixture(self,root):
        parent=root/'parent';(parent/'datasets').mkdir(parents=True);decisions={};groups={}
        for split,n in [('training',16),('development',8)]:
            group=[]
            for i in range(n):
                source=f'Neutral source {split} {i}.';target=f'Academic target {split} {i}.';rid=split+str(i)
                paper=split+str(i//2);authors=[split+' author '+str(i//2)]
                row={'example_id':rid,'paper_id':paper,'author_group_id':'AG_'+paper,'authors':authors,'split':split,
                     'source':source,'target':target,'source_sha256':runtime.sha(source),'target_sha256':runtime.sha(target),
                     'needs_review':False,'needs_external_context':False,'source_quality_status':'uncertain',
                     'publication_year':2010,'pdf_sha256':runtime.sha(paper)}
                group.append(row);decisions[rid]={'decision':'approve_limited_pilot','source_sha256':row['source_sha256'],
                    'target_sha256':row['target_sha256'],'alignment_notes':'Synthetic matching content.','context_disposition':'Locally defined.'}
            groups[split]=group
        for split,name in [('training','train.provisional.private.jsonl'),('development','development.private.jsonl')]:
            (parent/'datasets'/name).write_text(''.join(json.dumps(r)+'\n' for r in groups[split]))
        files={str(p.relative_to(parent)):__import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in parent.rglob('*') if p.is_file()}
        (parent/'manifest.json').write_text(json.dumps({'files':files}))
        path=root/'decisions.json';path.write_text(json.dumps({'decisions':decisions}))
        return parent,path

    def package(self,root):
        parent,decisions=self.fixture(root);build(parent,decisions,root/'out')
        return root/'out/upload/teacher_pair_pilot.jsonl',root/'out/upload/teacher_pair_config.json'

    def test_split_guard_and_cpu_cli_never_call_training(self):
        with tempfile.TemporaryDirectory() as folder:
            a,b=self.package(Path(folder));groups,config=runtime.load_inputs(a,b)
            self.assertEqual([len(v) for v in groups.values()],[16,8])
            with patch.object(runtime,'run_training',side_effect=AssertionError('No GPU')):
                with contextlib.redirect_stdout(io.StringIO()):runtime.main(['--data',str(a),'--config',str(b),'--check'])
                with self.assertRaises(ValueError):runtime.main(['--data',str(a),'--config',str(b),'--mode','pilot'])

    def test_target_mask_excludes_prompt_and_rejects_boundary_or_target_changes(self):
        self.assertEqual(runtime.mask_labels([1,2],[1,2,3,4],'Target.','Target.'),[-100,-100,3,4])
        for args in (([1,2],[1,5,3],'Target.','Target.'),([1,2],[1,2],'','Target.'),
                     ([1],[1,2],'Wrong.','Target.')):
            with self.assertRaises(ValueError):runtime.mask_labels(*args)

    def test_author_overlap_and_wrong_input_direction_rejected_even_after_rehash(self):
        for issue in ('author','direction'):
            with tempfile.TemporaryDirectory() as folder:
                a,b=self.package(Path(folder));records=[json.loads(l) for l in a.read_text().splitlines()]
                if issue=='author':records[16]['authors']=records[0]['authors']
                else:records[0]['source'],records[0]['target']=records[0]['target'],records[0]['source']
                a.write_text(''.join(json.dumps(r)+'\n' for r in records));config=json.loads(b.read_text())
                config['dataset_sha256']=runtime.sha(a.read_text());b.write_text(json.dumps(config))
                with self.assertRaises(ValueError):runtime.load_inputs(a,b)

    def test_audit_before_hash_and_complete_candidate_coverage_required(self):
        for issue in ('hash','coverage'):
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);parent,path=self.fixture(root);decision=json.loads(path.read_text())
                if issue=='hash':decision['decisions']['training0']['source_sha256']='wrong'
                else:del decision['decisions']['training0']
                path.write_text(json.dumps(decision))
                with self.assertRaises(ValueError):build(parent,path,root/'out')

    def test_baseline_needs_exact_ids_and_finite_loss(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);a,b=self.package(root);config=json.loads(b.read_text())
            snap={'complete':True,'name':'base','optimizer_step':0,
                  'losses':[{'example_id':rid} for rid in config['development_ids']],
                  'generations':[{'example_id':rid} for rid in config['development_ids']+config['training_probe_ids']],
                  'token_weighted_development_loss':1.0}
            report={'complete':True,'mode':'baseline','config':config,'snapshots':[snap]}
            path=root/'evaluation.json';path.write_text(json.dumps(report));runtime.validate_baseline(path,config)
            snap['generations'][0]['example_id']='wrong';path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):runtime.validate_baseline(path,config)

    def test_all_ml_imports_are_inside_run_training_and_notebooks_are_clean(self):
        source=(ROOT/'src/fama_style/training/teacher_pair_pilot.py').read_text();tree=ast.parse(source)
        for node in tree.body:
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module]
                self.assertFalse(any(n.split('.')[0] in ('torch','transformers','peft','datasets') for n in names))
        for filename in ('teacher_pair_base_4b.ipynb','teacher_pair_pilot_4b.ipynb'):
            path=ROOT/'kaggle'/filename;book=json.loads(path.read_text())
            text='\n'.join(''.join(c['source']) for c in book['cells'])
            self.assertIn(source,text);self.assertLess(path.stat().st_size,1_000_000)
            if filename.startswith('teacher_pair_pilot'):
                self.assertIn('RUN_PILOT = True',text)
                self.assertIn('baseline_evaluation_sha256',text)
                self.assertIn('differs from the frozen, reviewed pilot inputs',text)
            for cell in book['cells']:
                if cell['cell_type']=='code':self.assertIsNone(cell['execution_count']);self.assertEqual(cell['outputs'],[])

    def test_authorized_notebook_uses_approved_hashes_before_training(self):
        import hashlib
        authorization=json.loads((ROOT/'docs/teacher_pair_pilot_v1/pilot_execution.json').read_text())
        self.assertTrue(authorization['user_authorized'])
        self.assertFalse(authorization['full_training_authorized'])
        self.assertEqual((authorization['epochs'],authorization['optimizer_steps']),(2,16))
        source=(ROOT/'src/fama_style/training/teacher_pair_pilot.py').read_bytes()
        self.assertEqual(authorization['runtime_source_sha256'],hashlib.sha256(source).hexdigest())
        book=json.loads((ROOT/'kaggle/teacher_pair_pilot_4b.ipynb').read_text())
        launch=''.join(book['cells'][-1]['source']);tree=ast.parse(launch)
        expected=next(ast.literal_eval(node.value) for node in ast.walk(tree)
                      if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='expected' for t in node.targets))
        self.assertEqual(expected,{k:authorization[k] for k in ('baseline_evaluation_sha256','dataset_sha256','config_sha256')})
        self.assertLess(launch.index('hashlib.sha256(path.read_bytes())'),launch.index("'--train-authorized'"))


if __name__=='__main__':unittest.main()
