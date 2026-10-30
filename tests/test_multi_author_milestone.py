"""Dataset separation, prior isolation and CPU-only prepared-baseline contracts."""
import ast
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.research import multi_author_milestone as m
from fama_style.evaluation import multi_author_baseline as runtime
from fama_style.research.author_control_replication import verify_files

ROOT=Path(__file__).resolve().parents[1]


class MultiAuthorMilestoneTests(unittest.TestCase):
    def fixture(self,root):
        audit=root/'audit';audit.mkdir();papers=[];rows=[]
        for i in range(10):
            split='training' if i<7 else 'development';label=chr(65+i)
            papers.append({'paper_id':'p'+label,'authors':['Writer '+label],'title':'Study '+label,
                'publication_year':1950+i,'version_year':1950+i,'pdf_sha256':label*64,
                'publication_evidence_url':'https://example.org/'+label,'split':split})
            text='In example '+label+', firms may invest. Investors supply funds.'
            target='In example '+label+', investors supply funds, and firms may invest.'
            a={'plain_english':text,'original_paragraph':target,'rewrite_status':'simplified',
                'semantic_ledger':{'claims':[{'alignment_status':'uncertain' if i==6 else 'preserved'}]},
                'source_quality':{'status':'uncertain','needs_external_context':True},'needs_review':True,
                'source_schema':{'sentences':[{'exact_text':s} for s in text.split('. ')[:-1]]}}
            a['source_schema']['sentences']=[{'exact_text':'In example '+label+', firms may invest.'},{'exact_text':'Investors supply funds.'}]
            if i==9:a['plain_english']=target;a['source_schema']={'sentences':[{'exact_text':target}]}
            rows.append({'paragraph_id':'row'+label,'paper_id':'p'+label,'split':split,'schema_valid':True,
                'audit':{'errors':[],'program_errors':[]},'annotation':a})
        # The excluded body is intentionally not JSON-decodable.
        (audit/'audits.private.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows)+
            '{"split":"reserved_test","annotation": UNREADABLE_RESERVED_BODY}\n')
        m.file_manifest(audit)
        papers.append({'paper_id':'reserved','authors':['Reserved Author'],'split':'reserved_test','publication_year':1900})
        registry=root/'registry.json';registry.write_text(json.dumps({'papers':papers}))
        spot=root/'spot.json';spot.write_text(json.dumps({'records':[{'paragraph_id':'rowI','assessment':'visible_epistemic_strength_change'}]}))
        return audit,registry,spot

    def test_quarantine_metadata_and_detector_controls_preserve_split_boundaries(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);audit,registry,spot=self.fixture(root);result=m.build(audit,registry,spot,root/'out')
            self.assertEqual((result['training_candidates'],result['development_examples'],result['quarantined']),(6,1,3))
            self.assertEqual(result['baseline_jobs'],2);self.assertEqual(result['detector_controls'],12)
            self.assertFalse(result['training_approved']);self.assertFalse(result['reserved_content_used'])
            texts=(root/'out/upload/multi_author_base_jobs.jsonl').read_text()
            self.assertNotIn('Reserved Author',texts);self.assertNotIn('original_paragraph',texts)
            key=[json.loads(line) for line in (root/'out/detector_controls/PRIVATE_KEY.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['paper_id'] in {'p'+chr(65+i) for i in range(6)} for r in key))
            with (root/'out/detector_controls/results_template.csv').open() as f:
                entries=list(csv.DictReader(f))
            self.assertEqual(len(entries),12);self.assertTrue(all(r['status']=='not_tested' for r in entries))
            verify_files(root/'out')
            with self.assertRaises(ValueError):m.build(audit,registry,spot,root/'out')

    def test_profile_and_generator_do_not_depend_on_development_references(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);audit,registry,spot=self.fixture(root)
            selected,_=m.select_pairs(m.nonreserved_rows(audit),json.loads(registry.read_text()),json.loads(spot.read_text()))
            train=[r for r in selected if r['split']=='training'];dev=next(r for r in selected if r['split']=='development')
            profile=m.profile(train)
            with self.assertRaises(ValueError):m.profile([dev])
            for condition in m.CONDITIONS:
                first=m.make_job(dev,profile,condition);poison=copy.deepcopy(dev)
                poison.update(target='PRIVATE_REFERENCE',authors=['PRIVATE_AUTHOR'],title='PRIVATE_TITLE',target_schema='PRIVATE_ORACLE')
                self.assertEqual(first,m.make_job(poison,profile,condition));self.assertNotIn('PRIVATE_',str(first))
            plan=m.structural_prior(dev,profile)
            self.assertLessEqual(len(plan['sentence_structure']['source_sentence_numbers']),2)

    def test_author_overlap_or_reserved_rows_cannot_enter_candidate_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);audit,registry,spot=self.fixture(root);rows=m.nonreserved_rows(audit)
            meta=json.loads(registry.read_text());meta['papers'][7]['authors']=meta['papers'][0]['authors']
            with self.assertRaisesRegex(ValueError,'author_crosses'):m.select_pairs(rows,meta,json.loads(spot.read_text()))
            rows[0]['split']='reserved_test'
            with self.assertRaisesRegex(ValueError,'reserved'):m.select_pairs(rows,json.loads(registry.read_text()),json.loads(spot.read_text()))

    def test_duplicate_sources_are_quarantined_without_rewriting(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);audit,registry,spot=self.fixture(root);rows=m.nonreserved_rows(audit)
            rows[1]['annotation']['plain_english']=rows[0]['annotation']['plain_english']
            _,quarantine=m.select_pairs(rows,json.loads(registry.read_text()),json.loads(spot.read_text()))
            self.assertIn('duplicate_source_or_target',next(r for r in quarantine if r['paragraph_id']=='rowB')['reasons'])

    def test_baseline_cpu_check_never_calls_gpu_backend_and_rejects_extra_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);audit,registry,spot=self.fixture(root);m.build(audit,registry,spot,root/'out')
            jobs=root/'out/upload/multi_author_base_jobs.jsonl';config=root/'out/upload/multi_author_baseline_config.json'
            with patch.object(runtime,'run_training',side_effect=AssertionError('GPU execution forbidden')):
                runtime.main(['--jobs',str(jobs),'--config',str(config),'--check'])
            rows=[json.loads(line) for line in jobs.read_text().splitlines()];rows[0]['reference']='PRIVATE'
            jobs.write_text(''.join(json.dumps(r)+'\n' for r in rows));cfg=json.loads(config.read_text())
            cfg['jobs_sha256']=m.file_sha(jobs);config.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError,'Unapproved'):runtime.load_inputs(jobs,config)

    def test_secret_uses_kaggle_client_before_environment_without_printing(self):
        secret='hf_'+'Q'*30;fake=types.ModuleType('kaggle_secrets')
        class Client:
            def get_secret(self,name):
                self.name=name;return secret
        fake.UserSecretsClient=Client
        with patch.dict(sys.modules,{'kaggle_secrets':fake}),patch.object(runtime.Path,'exists',return_value=True),patch.dict(runtime.os.environ,{'HF_TOKEN':'wrong-local-value'}),contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(runtime.load_hf_token(),secret)
        self.assertEqual(stream.getvalue(),'')
        self.assertNotIn(secret,runtime.safe_error(RuntimeError('Failure '+secret),secret))

    def test_incremental_generation_save_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'outputs.jsonl';job={'job_id':'j','example_id':'x','condition':'ordinary','source':'Synthetic source.','prompt_sha256':'sha'}
            runtime.collect_outputs([job],lambda job:{'output':'Synthetic output.','synthetic':True},path)
            self.assertEqual(json.loads(path.read_text())['output'],'Synthetic output.')
            with self.assertRaises(FileExistsError):runtime.collect_outputs([job],lambda job:{'output':'Wrong'},path)

    def test_public_notebook_is_small_self_contained_and_inference_only(self):
        path=ROOT/'kaggle/multi_author_baseline.ipynb';book=json.loads(path.read_text())
        self.assertLess(path.stat().st_size,1_000_000)
        self.assertTrue(all(not c.get('outputs') and c.get('execution_count') is None for c in book['cells']))
        code=''.join(book['cells'][-1]['source']);tree=ast.parse(code)
        self.assertFalse(any(isinstance(n,ast.ImportFrom) and n.module and n.module.startswith('fama_style') for n in ast.walk(tree)))
        module=ast.parse((ROOT/'src/fama_style/evaluation/multi_author_baseline.py').read_text())
        top=[n for n in module.body if isinstance(n,(ast.Import,ast.ImportFrom))]
        self.assertFalse(any(getattr(n,'module','') in ('torch','transformers','huggingface_hub') for n in top))
        self.assertNotIn('.backward(',code);self.assertNotIn('Trainer(',code)


if __name__=='__main__':unittest.main()
