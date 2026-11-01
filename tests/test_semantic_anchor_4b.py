"""Matched inference jobs, target-only scoring and no-training guarantees on CPU."""
import ast
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from fama_style.evaluation import semantic_anchor_4b as runtime
from fama_style.research.semantic_anchor_inference_packet import build

ROOT=Path(__file__).resolve().parents[1]


def freeze(folder):
    (folder/'manifest.json').write_text(json.dumps({'files':{
        str(p.relative_to(folder)):runtime.file_sha(p) for p in folder.rglob('*') if p.is_file()}}))


class SemanticAnchorInference(unittest.TestCase):
    def fixture(self,root):
        anchors=root/'anchors';anchors.mkdir();adapter=root/'adapter';adapter.mkdir();review=root/'review';review.mkdir()
        records=[]
        for i in range(12):
            source=f'Original source {i}.';variant=f'Variant source {i}.';chosen=f'Faithful completion {i}.';rejected=f'Unfaithful completion {i}.'
            records.append({'example_id':str(i),'paper_id':'paper'+str(i),'family':('scope','time','numbers','reasoning')[i//3],
                'split':'training','source':source,'faithful_variant':variant,'chosen':chosen,'rejected':rejected,
                'target_sha256':runtime.sha(chosen),'rejected_sha256':runtime.sha(rejected)})
        (anchors/'anchors.private.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records));freeze(anchors)
        (adapter/'adapter_model.safetensors').write_bytes(b'Synthetic hash-only adapter fixture. Never loaded.')
        (adapter/'adapter_config.json').write_text(json.dumps({'base_model_name_or_path':runtime.MODEL_ID,'r':8,'lora_alpha':16,'lora_dropout':0.05}))
        (review/'summary.json').write_text(json.dumps({'execution_complete':True,'optimizer_steps':46,
            'checkpoints':{'2':{'sha256':runtime.file_sha(adapter/'adapter_model.safetensors')}}}))
        (review/'receipt.json').write_text(json.dumps({'download_files_sha256':{
            'teacher_pair_pilot_v2/epoch_2/adapter/'+name:runtime.file_sha(adapter/name) for name in ('adapter_config.json','adapter_model.safetensors')}}))
        freeze(review)
        output=root/'out';build(anchors,adapter,review,output)
        return output,adapter

    def test_cpu_cli_never_loads_model_and_packet_reproduces(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out,adapter=self.fixture(root)
            jobs=out/'upload/semantic_anchor_jobs.jsonl';config=out/'upload/semantic_anchor_config.json'
            with patch.object(runtime,'run_training',side_effect=AssertionError('No GPU or weights')):
                with contextlib.redirect_stdout(io.StringIO()):runtime.main(['--jobs',str(jobs),'--config',str(config),'--adapter',str(adapter),'--check'])
                with self.assertRaises(ValueError):runtime.main(['--jobs',str(jobs),'--config',str(config),'--adapter',str(adapter)])
            build(root/'anchors',adapter,root/'review',root/'replica')
            for p in out.rglob('*'):
                if p.is_file():self.assertEqual(p.read_bytes(),(root/'replica'/p.relative_to(out)).read_bytes())
            with self.assertRaises(ValueError):build(root/'anchors',adapter,root/'review',out)

    def test_rehashed_target_leakage_and_mismatched_conditions_are_rejected(self):
        for issue in ('prompt','reference','split','coverage'):
            with tempfile.TemporaryDirectory() as folder:
                out,adapter=self.fixture(Path(folder));a=out/'upload/semantic_anchor_jobs.jsonl';b=out/'upload/semantic_anchor_config.json'
                jobs=[json.loads(l) for l in a.read_text().splitlines()];config=json.loads(b.read_text())
                if issue=='prompt':jobs[0]['prompt']+='\nReference: '+jobs[0]['chosen']
                elif issue=='reference':jobs[0]['chosen']='Changed';jobs[0]['target_sha256']=runtime.sha('Changed')
                elif issue=='split':jobs[0]['split']='development'
                else:jobs.pop()
                a.write_text(''.join(json.dumps(r)+'\n' for r in jobs));config['jobs_sha256']=runtime.file_sha(a);b.write_text(json.dumps(config))
                with self.assertRaises(ValueError):runtime.load_inputs(a,b)

    def test_exact_adapter_files_required(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out,adapter=self.fixture(root)
            config=json.loads((out/'upload/semantic_anchor_config.json').read_text());runtime.verify_adapter(adapter,config)
            (adapter/'adapter_model.safetensors').write_bytes(b'Changed')
            with self.assertRaises(ValueError):runtime.verify_adapter(adapter,config)
            with self.assertRaises(ValueError):build(root/'anchors',adapter,root/'review',root/'changed')

    def test_labels_exclude_prefix_and_refuse_truncation_or_decoding_changes(self):
        self.assertEqual(runtime.mask_labels([1,2],[1,2,3,4],'Text.','Text.'),[-100,-100,3,4])
        for args in (([1],[1]*1025,'Text.','Text.'),([1,2],[1,4,3],'Text.','Text.'),([1],[1,2],'Wrong','Text.')):
            with self.assertRaises(ValueError):runtime.mask_labels(*args)

    def test_likelihood_sign_ties_and_length_disagreement(self):
        result=runtime.margin({'mean_nll':2.,'supervised_tokens':100},{'mean_nll':2.01,'supervised_tokens':99})
        self.assertEqual(result['mean_nll_preference'],'chosen');self.assertEqual(result['total_nll_preference'],'rejected')
        self.assertTrue(result['mean_total_disagree'])
        self.assertEqual(runtime.margin({'mean_nll':2.,'supervised_tokens':100},{'mean_nll':2.,'supervised_tokens':100})['mean_nll_preference'],'tie')
        for loss in (float('nan'),float('inf'),-1):
            with self.assertRaises(ValueError):runtime.margin({'mean_nll':loss,'supervised_tokens':1},{'mean_nll':1.,'supervised_tokens':1})

    def test_generation_messages_have_no_reference_or_ledger(self):
        job={'prompt':'Ordinary rewrite input.','chosen':'SECRET_REFERENCE','rejected':'SECRET_REJECTION','claims':['SECRET_LEDGER']}
        serialized=json.dumps(runtime.messages(job))
        for forbidden in ('SECRET_REFERENCE','SECRET_REJECTION','SECRET_LEDGER'):self.assertNotIn(forbidden,serialized)

    def test_secret_errors_do_not_reveal_underlying_exception(self):
        client=types.SimpleNamespace(get_secret=lambda name:(_ for _ in ()).throw(RuntimeError('sensitive-placeholder')))
        module=types.SimpleNamespace(UserSecretsClient=lambda:client)
        stream=io.StringIO()
        with patch.dict(sys.modules,{'kaggle_secrets':module}),contextlib.redirect_stdout(stream):
            with self.assertRaises(RuntimeError) as error:runtime.load_hf_token()
        self.assertNotIn('sensitive-placeholder',str(error.exception));self.assertEqual(stream.getvalue(),'')

    def test_runtime_contains_no_training_calls_and_notebook_is_self_contained(self):
        source=Path(runtime.__file__).read_text();tree=ast.parse(source)
        forbidden={'Trainer','TrainingArguments','LoraConfig','get_peft_model','prepare_model_for_kbit_training'}
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                if isinstance(node.func,ast.Name):self.assertNotIn(node.func.id,forbidden)
                if isinstance(node.func,ast.Attribute):self.assertNotIn(node.func.attr,{'train','backward','step','zero_grad'})
        for node in tree.body:
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module]
                self.assertFalse(any(n.split('.')[0] in ('torch','transformers','peft') for n in names))
        notebook=ROOT/'kaggle/semantic_anchor_4b.ipynb';book=json.loads(notebook.read_text())
        self.assertIn(source,'\n'.join(''.join(c['source']) for c in book['cells']))
        self.assertLess(notebook.stat().st_size,1_000_000)
        launch=''.join(book['cells'][-1]['source']);ast.parse(launch)
        self.assertLess(launch.index('hashlib.sha256(path.read_bytes())'),launch.index("'--run-inference'"))
        for cell in book['cells']:
            if cell['cell_type']=='code':self.assertIsNone(cell['execution_count']);self.assertEqual(cell['outputs'],[])


if __name__=='__main__':unittest.main()
