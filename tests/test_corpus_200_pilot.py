"""CPU checks for the whole 200-pair pilot, its labels, split and notebook."""
import ast
from collections import Counter
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fama_style.training import corpus_200_pilot as runtime
from fama_style.research.corpus_200_packet import build


class Corpus200Pilot(unittest.TestCase):
    def package(self,root):
        parents=[]
        for split,count in [('training',50),('development',4)]:
            folder=root/split;folder.mkdir();rows=[]
            for group in runtime.GROUPS:
                for i in range(count):
                    source=f'Neutral {split} {group} passage {i}.'
                    target=f'Academic {split} {group} paragraph {i}.'
                    rows.append({'id':f'{split}|{group}|{i}','parent_id':f'parent|{split}|{group}|{i}',
                        'paper_id':f'{split}|{group}|{i%2}','style_group':group,'authors':[group+' author'],
                        'split':split,'plain_english':source,'original_paragraph':target,
                        'source_sha256':runtime.sha(source),'target_sha256':runtime.sha(target),
                        'quality':{'text_pair_status':'automated_pilot_approved'}})
            path=folder/'pairs.private.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            runtime.write_json(folder/'manifest.json',{'files':{path.name:runtime.sha(path.read_text())}})
            parents.append(path)
        output=root/'packet';build(*parents,output)
        return parents,output,output/'upload/corpus_200_pairs.jsonl',output/'upload/corpus_200_config.json'

    def test_full_packet_cpu_check_does_not_load_model_and_requires_run_flag(self):
        with tempfile.TemporaryDirectory() as folder:
            parents,out,data,config=self.package(Path(folder))
            with patch.object(runtime,'run_training',side_effect=AssertionError('No ML on CPU')):
                with contextlib.redirect_stdout(io.StringIO()):runtime.main(['--data',str(data),'--config',str(config),'--check'])
                with self.assertRaises(ValueError):runtime.main(['--data',str(data),'--config',str(config)])
                with self.assertRaises(ValueError):runtime.main(['--data',str(data),'--config',str(config),'--train-authorized','--max-runtime-seconds','5401'])
            splits,c=runtime.load_inputs(data,config)
            self.assertEqual(len(splits['training']),200)
            self.assertEqual(len(splits['development']),16)
            self.assertEqual(Counter(r['style_group'] for r in splits['training']),Counter({g:50 for g in runtime.GROUPS}))
            self.assertEqual(len(c['detector_case_ids']),8)
            self.assertEqual(len({r['paper_id'] for r in splits['development']}),8)
            # Known author groups intentionally occur in both paper splits.
            self.assertEqual({a for r in splits['training'] for a in r['authors']},{a for r in splits['development'] for a in r['authors']})

    def test_wrong_role_count_paper_and_update_budget_fail_even_after_rehash(self):
        for issue in ['reserved','count','paper','budget','prompt']:
            with self.subTest(issue=issue), tempfile.TemporaryDirectory() as folder:
                parents,out,data,path=self.package(Path(folder));rows=[json.loads(l) for l in data.read_text().splitlines()];config=json.loads(path.read_text())
                if issue=='reserved':rows[-1]['split']='reserved_test'
                elif issue=='count':rows.pop(0)
                elif issue=='paper':rows[-1]['paper_id']=rows[0]['paper_id']
                elif issue=='budget':config['hyperparameters']['optimizer_steps']=200
                else:rows[-1]['prompt']+=' Target: '+rows[-1]['target']
                data.write_text(''.join(json.dumps(r)+'\n' for r in rows));config['dataset_sha256']=runtime.sha(data.read_text());runtime.write_json(path,config)
                with self.assertRaises(ValueError):runtime.load_inputs(data,path)

    def test_supervised_response_boundaries_and_no_truncation(self):
        self.assertEqual(runtime.mask_labels([1,2],[1,2,3,4],'Target text.','Target text.'),[-100,-100,3,4])
        cases=[([1,2],[1,7,3],'Target.','Target.'),([1],[1],'Target.','Target.'),
               ([1],[1]*1537,'Target.','Target.'),([1],[1,2],'Wrong.','Target.')]
        for case in cases:
            with self.assertRaises(ValueError):runtime.mask_labels(*case)
        self.assertEqual(runtime.learning_rate(1),2e-5)
        self.assertGreater(runtime.learning_rate(100),0)
        for step in [0,101,1.5]:
            with self.assertRaises(ValueError):runtime.learning_rate(step)

    def test_packet_is_reproducible_and_parents_are_hash_verified(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);parents,out,data,path=self.package(root)
            replica=root/'replica';build(*parents,replica)
            for file in out.rglob('*'):
                if file.is_file():self.assertEqual(file.read_bytes(),(replica/file.relative_to(out)).read_bytes())
            with self.assertRaises(ValueError):build(*parents,out)
            parents[0].write_text(parents[0].read_text()+'\n')
            with self.assertRaises(ValueError):build(*parents,root/'changed')

    def test_self_contained_notebook_has_no_private_rows_and_imports_stay_cpu_safe(self):
        spec=importlib.util.spec_from_file_location('build_corpus_200_pilot',ROOT/'scripts/build_corpus_200_pilot.py')
        builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as folder:
            parents,out,data,path=self.package(Path(folder));notebook=Path(folder)/'pilot.ipynb'
            builder.build_notebook(out,notebook);book=json.loads(notebook.read_text())
            text='\n'.join(''.join(c['source']) for c in book['cells'])
            self.assertIn(Path(runtime.__file__).read_text(),text)
            self.assertNotIn('Neutral training Fama_coauthors passage 0.',text)
            self.assertLess(notebook.stat().st_size,1_000_000)
            for cell in book['cells']:
                if cell['cell_type']=='code':
                    self.assertIsNone(cell['execution_count']);self.assertEqual(cell['outputs'],[])
                    source=''.join(cell['source'])
                    if not source.startswith('%%writefile'):ast.parse(source)
            setup=''.join(book['cells'][1]['source']);launch=''.join(book['cells'][-1]['source'])
            self.assertIn("os.environ['CUDA_VISIBLE_DEVICES']='0'",setup)
            self.assertIn('timeout=remaining()',launch)
            self.assertIn("'--train-authorized'",launch)
        tree=ast.parse(Path(runtime.__file__).read_text())
        for node in tree.body:
            if isinstance(node,ast.Import):
                self.assertFalse(any(a.name.split('.')[0] in {'torch','transformers','peft','huggingface_hub'} for a in node.names))
            if isinstance(node,ast.ImportFrom):self.assertNotIn((node.module or '').split('.')[0],{'torch','transformers','peft','huggingface_hub'})


if __name__=='__main__':unittest.main()
