"""CPU checks for the isolated purpose-only release and Kaggle runtime."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fama_style.training import gpt6_purpose_500 as runtime

spec=importlib.util.spec_from_file_location('build_gpt6_purpose_500',ROOT/'scripts/build_gpt6_purpose_500.py')
builder=importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def fixture(directory):
    rows=[];pdfs={}
    for i in range(500):
        g=runtime.GROUPS[i%4];paper='train_'+g
        rows.append(dict(example_id='P%03d'%(i+1),content_id='P%03d'%(i+1),parent_id='parent_%d'%i,
            paper_id=paper,style_group=g,authors=['Synthetic author'],split='training',
            source='Neutral training source %d.'%i,target='Synthetic target paragraph %d.'%i,
            pilot_eligible=True,input_condition='training_gpt6_purpose_only'))
        pdfs[paper]=runtime.sha(paper)
    for g in runtime.GROUPS:
        for p in range(2):
            paper='development_'+g+str(p);pdfs[paper]=runtime.sha(paper)
            for k in range(2):
                cid='dev_'+g+str(p)+str(k)
                r=dict(example_id=cid,content_id=cid,parent_id='parent_'+cid,paper_id=paper,
                    style_group=g,authors=['Synthetic author'],split='development',
                    source='Existing source '+cid,target='Held-out target '+cid,
                    pilot_eligible=True,input_condition='existing_input')
                rows.append(r)
                if k==0:
                    rows.append(dict(r,example_id=cid+'::gemini_input_v1',source='Gemini variant '+cid,input_condition='gemini_input'))
    for r in rows:
        r.update(source_sha256=runtime.sha(r['source']),target_sha256=runtime.sha(r['target']),
                 prompt=runtime.prompt(r['source'],training=r['split']=='training'))
    data=directory/'pairs.jsonl';builder.write_rows(data,rows)
    config=dict(protocol=runtime.PROTOCOL,model_id=runtime.MODEL_ID,model_revision=runtime.REVISION,
        hyperparameters=runtime.HYPER,decoding=runtime.DECODING,packages=runtime.PACKAGES,
        max_runtime_seconds=runtime.MAX_SECONDS,training_authorized=True,
        quantization='NF4 double quantization; FP32 compute',runtime_source_sha256=builder.digest(Path(runtime.__file__)),
        dataset_sha256=builder.digest(data),canonical_pdf_by_paper=pdfs,
        training_instruction=runtime.INSTRUCTION,inference_instruction=runtime.INSTRUCTION,
        training_probe_ids=['P001','P002','P003','P004'])
    for split in ('training','development'):
        config[split+'_ids']=[r['example_id'] for r in rows if r['split']==split]
    config['gemini_development_ids']=[r['example_id'] for r in rows if r['input_condition']=='gemini_input']
    config['parent_detector_content_ids']=[r['content_id'] for r in rows if r['input_condition']=='gemini_input']
    path=directory/'config.json';runtime.write_json(path,config)
    return rows,config,data,path


class Purpose500Tests(unittest.TestCase):
    def test_cpu_check_and_training_authorization(self):
        with tempfile.TemporaryDirectory() as d:
            _,_,data,config=fixture(Path(d))
            with patch.object(runtime,'run_training',side_effect=AssertionError('No ML allowed')):
                runtime.main(['--data',str(data),'--config',str(config),'--check'])
                with self.assertRaisesRegex(ValueError,'authorization'):
                    runtime.main(['--data',str(data),'--config',str(config)])

    def test_alias_pdf_leakage_detected_even_when_paper_ids_differ(self):
        with tempfile.TemporaryDirectory() as d:
            rows,c,data,p=fixture(Path(d))
            dev=next(r['paper_id'] for r in rows if r['split']=='development')
            c['canonical_pdf_by_paper'][dev]=c['canonical_pdf_by_paper'][rows[0]['paper_id']]
            runtime.write_json(p,c)
            with self.assertRaisesRegex(ValueError,'canonical_PDF_leakage'): runtime.load_inputs(data,p)

    def test_changed_source_and_prompt_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            rows,c,data,p=fixture(Path(d));rows[0]['source']='Changed prose'
            builder.write_rows(data,rows);c['dataset_sha256']=builder.digest(data);runtime.write_json(p,c)
            with self.assertRaisesRegex(ValueError,'text_or_prompt_changed'): runtime.load_inputs(data,p)

    def test_target_only_supervision_and_no_truncation(self):
        self.assertEqual(runtime.mask_labels([1,2],[1,2,3,4],'target','target'),[-100,-100,3,4])
        with self.assertRaises(ValueError):runtime.mask_labels([1,2],[1,9,3],'target','target')
        with self.assertRaises(ValueError):runtime.mask_labels([1],[1,2],'wrong','target')
        self.assertEqual(runtime.TRAIN_INSTRUCTION,runtime.INSTRUCTION)
        self.assertNotIn('Retain its substantive',runtime.TRAIN_INSTRUCTION)

    def test_all_500_adjudicated_without_source_edits(self):
        candidates=[]
        for i in range(1,501):
            target='Target '+str(i)+(builder.OLD_CLAUSE if i==139 else '.')
            source='AI source '+str(i)
            candidates.append(dict(id='P%03d'%i,source=source,source_sha256=runtime.sha(source),
                target=target,target_sha256=runtime.sha(target),original_paragraph=target,
                provenance={},quality={'followup':{'reason':'historical warning'}},training_ready=False))
        before=copy.deepcopy(candidates);final,audit=builder.adjudicate(candidates)
        self.assertEqual(candidates,before)
        self.assertEqual(len(final),500)
        self.assertTrue(all(r['training_ready'] for r in final))
        self.assertEqual([r['source'] for r in final],[r['source'] for r in candidates])
        self.assertEqual(sum(a['target']!=b['target'] for a,b in zip(candidates,final)),1)
        self.assertIn('lower expected returns',final[138]['target'])
        self.assertIn('not independent verification',audit[399]['decision'])
        self.assertIn('nonblocking',audit[55]['decision'])

    def test_no_module_level_ml_imports_and_secret_not_printed(self):
        source=Path(runtime.__file__).read_text();tree=ast.parse(source)
        for node in tree.body:
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or '']
                self.assertFalse(any(n.split('.')[0] in ('torch','transformers','peft','kaggle_secrets') for n in names))
        self.assertIn("get_secret('HF_TOKEN')",source)
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='print':
                self.assertFalse(any(isinstance(x,ast.Name) and x.id=='token' for x in ast.walk(node)))

    def test_private_packet_and_self_contained_notebook_when_available(self):
        if not builder.RELEASE.exists():self.skipTest('Private dataset is not distributed')
        splits,_=builder.check(builder.RELEASE)
        self.assertEqual([len(splits[s]) for s in ('training','development')],[500,24])
        bookpath=ROOT/'kaggle/gpt6_purpose_500_4b.ipynb';book=json.loads(bookpath.read_text())
        self.assertLess(bookpath.stat().st_size,1_000_000)
        codes=[''.join(c['source']) for c in book['cells'] if c['cell_type']=='code']
        self.assertEqual(codes[2].split('\n',1)[1],Path(runtime.__file__).read_text())
        for s in codes:ast.parse(s.split('\n',1)[1] if s.startswith('%%writefile') else s)
        self.assertIn("CUDA_VISIBLE_DEVICES']='0'",codes[0])
        self.assertTrue(all(not c['outputs'] and c['execution_count'] is None for c in book['cells'] if c['cell_type']=='code'))
        text=bookpath.read_text()
        for r in (splits['training'][138],splits['development'][0]):
            self.assertNotIn(r['target'],text)
            self.assertNotIn(r['source'],text)


if __name__=='__main__':unittest.main()
