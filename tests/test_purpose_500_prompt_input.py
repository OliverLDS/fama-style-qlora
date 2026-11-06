import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fama_style.evaluation import purpose_500_prompt_input as runtime


def load(name,file):
    s=importlib.util.spec_from_file_location(name,ROOT/file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


builder=load('purpose_prompt_builder','scripts/build_purpose_500_prompt_input.py')
projection=load('purpose_brief_projection','scripts/data/complete_purpose_500_diagnostic.py')


def fixture(root):
    adapter=root/'adapter';adapter.mkdir()
    for name in ['adapter_config.json','adapter_model.safetensors']:(adapter/name).write_text('Synthetic '+name)
    original=[];jobs=[];pdfs={};fresh={};selected=[]
    for i,g in enumerate(runtime.GROUPS):
        cid='content_'+str(i);selected.append(cid);paper='paper_'+str(i);pdfs[paper]=runtime.sha(paper)
        reference='Reference-only target '+str(i)
        for route in runtime.ROUTES:
            source='Synthetic '+route+' input '+str(i)
            if route=='fresh_reconstruction':fresh[cid]=runtime.sha(source)
            else:original.append(dict(example_id=cid if route=='existing_input' else cid+'::gemini_input_v1',
                source=source,reference=reference,paper_id=paper,style_group=g))
            for condition in runtime.PROMPTS:
                jobs.append(dict(example_id=cid+'::'+route+'::'+condition,content_id=cid,paper_id=paper,
                    style_group=g,reference=reference,source=source,source_sha256=runtime.sha(source),
                    reference_sha256=runtime.sha(reference),input_condition=route,prompt_condition=condition,
                    prompt_sha256=runtime.sha(runtime.prompt(source,condition)),canonical_pdf_sha256=pdfs[paper]))
    parent=dict(protocol='gpt6_purpose_500_v1',complete=True,optimizer_steps=250,training_examples_seen=500,
        adapter_reload_weights_match=True,adapter_files_sha256={p.name:runtime.file_sha(p) for p in adapter.iterdir()},
        config=dict(model_id=runtime.MODEL_ID,model_revision=runtime.REVISION,packages=runtime.PACKAGES,
                    decoding=runtime.DECODING,canonical_pdf_by_paper=pdfs),
        snapshots=[dict(name=name,complete=True,generations=original) for name in ['base','epoch_1']])
    report=root/'evaluation.json';runtime.write_json(report,parent)
    data=root/'data.jsonl';builder.write_rows(data,jobs)
    config=dict(protocol=runtime.PROTOCOL,model_id=runtime.MODEL_ID,model_revision=runtime.REVISION,
        prompts=runtime.PROMPTS,decoding=runtime.DECODING,packages=runtime.PACKAGES,max_runtime_seconds=runtime.MAX_SECONDS,
        training_authorized=False,runtime_source_sha256=runtime.file_sha(Path(runtime.__file__)),dataset_sha256=runtime.file_sha(data),
        parent_report_sha256=runtime.file_sha(report),selected_content_ids=selected,job_ids=[r['example_id'] for r in jobs],
        fresh_source_sha256_by_content=fresh)
    path=root/'config.json';runtime.write_json(path,config)
    return report,adapter,data,path,jobs,config


class PromptInputTests(unittest.TestCase):
    def test_cpu_check_covers_factorial_without_loading_model(self):
        with tempfile.TemporaryDirectory() as t:
            report,adapter,data,c,_,_=fixture(Path(t))
            with patch.object(runtime,'run_training',side_effect=AssertionError('No model')):
                runtime.main(['--report',str(report),'--adapter',str(adapter),'--data',str(data),'--config',str(c),'--check'])

    def test_changed_existing_control_is_rejected_even_if_rehashed(self):
        with tempfile.TemporaryDirectory() as t:
            report,adapter,data,c,jobs,config=fixture(Path(t))
            jobs[0]['source']='Changed control';jobs[0]['source_sha256']=runtime.sha(jobs[0]['source'])
            jobs[0]['prompt_sha256']=runtime.sha(runtime.prompt(jobs[0]['source'],jobs[0]['prompt_condition']))
            builder.write_rows(data,jobs);config['dataset_sha256']=runtime.file_sha(data);runtime.write_json(c,config)
            with self.assertRaisesRegex(ValueError,'existing_control_changed'):runtime.load_inputs(report,adapter,data,c)

    def test_prompt_specific_reference_change_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            report,adapter,data,c,jobs,config=fixture(Path(t));jobs[-1]['reference']='Different reference'
            builder.write_rows(data,jobs);config['dataset_sha256']=runtime.file_sha(data);runtime.write_json(c,config)
            with self.assertRaisesRegex(ValueError,'reference_or_paper_changed'):runtime.load_inputs(report,adapter,data,c)

    def test_adapter_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            report,adapter,data,c,_,_=fixture(Path(t));(adapter/'adapter_model.safetensors').write_text('Changed')
            with self.assertRaisesRegex(ValueError,'adapter_changed'):runtime.load_inputs(report,adapter,data,c)

    def test_inference_only_and_reference_not_added_to_prompt(self):
        self.assertNotIn('reference-only-secret',runtime.prompt('Given source','reexpress'))
        source=Path(runtime.__file__).read_text();tree=ast.parse(source)
        for node in tree.body:
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                names=[x.name for x in node.names] if isinstance(node,ast.Import) else [node.module or '']
                self.assertFalse(any(n.split('.')[0] in ['torch','peft','transformers'] for n in names))
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                self.assertNotIn(node.func.attr,['backward','step','save_pretrained'])
        self.assertIn("text=prompt(row['source'],row['prompt_condition'])",source)

    def test_named_key_points_projection_is_lossless_and_excludes_relations(self):
        row=dict(id='one',target_sha256='a'*64,status='ready',notes=[],brief=dict(topic='Topic',
            purpose=dict(primary='report_result',secondary=[]),key_points={'Claim':'Exact supplied value'},
            anchors={},relationships=[dict(type='unsupported_label')]))
        normalized,p,a=projection.project(row)
        self.assertEqual(p['brief']['key_points'],['Claim: Exact supplied value'])
        self.assertNotIn('relationships',p['brief'])
        self.assertEqual(row['brief']['key_points'],{'Claim':'Exact supplied value'})
        self.assertEqual(normalized['brief']['relationships'],row['brief']['relationships'])
        self.assertEqual(a['content_edits'],0)

    def test_notebook_and_actual_packet_when_available(self):
        if not builder.PACKET.exists():self.skipTest('Private packet not distributed')
        rows,_=builder.check();self.assertEqual(len(rows),24)
        notebook=json.loads(builder.NOTEBOOK.read_text());self.assertLess(builder.NOTEBOOK.stat().st_size,1_000_000)
        code=[''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']
        self.assertEqual(code[2].split('\n',1)[1],Path(runtime.__file__).read_text())
        for s in code:ast.parse(s.split('\n',1)[1] if s.startswith('%%writefile') else s)
        text=builder.NOTEBOOK.read_text()
        for r in rows:
            self.assertNotIn(r['reference'],text)
            self.assertNotIn(r['source'],text)


if __name__=='__main__':unittest.main()
