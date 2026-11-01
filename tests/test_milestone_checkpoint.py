"""Frozen subset, reuse, source-only prior and complete blind exports on CPU."""
import ast
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fama_style.evaluation import milestone_checkpoint_4b as runtime
from fama_style.research.milestone_checkpoint_packet import select_frozen,reuse_generation
from fama_style.research.milestone_checkpoint_export import collect_complete,export
from fama_style.research.multi_author_milestone import make_job

ROOT=Path(__file__).resolve().parents[1]


def freeze(folder):
    (folder/'manifest.json').write_text(json.dumps({'files':{str(p.relative_to(folder)):runtime.file_sha(p) for p in folder.rglob('*') if p.is_file()}}))


class MilestoneCheckpoint(unittest.TestCase):
    def fixture(self,root):
        package=root/'packet';(package/'upload').mkdir(parents=True);(package/'private_key').mkdir()
        jobs=[];refs=[];subset=[];reused=[];generated=[];tokens={};audit=[]
        missing={'2','6','7'}
        for i in range(8):
            rid=str(i);source=f'Neutral source for case {i}.';target=f'Human reference for case {i}.'
            refs.append({'example_id':rid,'paper_id':str(i%4),'source':source,'target':target,'needs_review':False,'needs_external_context':False})
            subset.append({'example_id':rid,'paper_id':str(i%4),'round':i//4+1,'source_sha256':runtime.sha(source),'target_sha256':runtime.sha(target)})
            for arm in runtime.CONDITIONS:
                prompt=runtime.INSTRUCTION+('\nOptional source-only structural prior: retain order.' if arm=='base_source_structural_prior' else '')+'\n\nPassage:\n'+source
                run=rid in missing or arm=='base_source_structural_prior'
                job={'job_id':rid+'__'+arm,'example_id':rid,'condition':arm,'source':source,'source_sha256':runtime.sha(source),
                     'prompt':prompt,'prompt_sha256':runtime.sha(prompt),'run_required':run};jobs.append(job)
                if not run:tokens[job['job_id']]=20
                text='Identical control output.' if rid=='0' and arm!='base_source_structural_prior' else f'{arm} case {i} output.'
                output={k:job[k] for k in ('job_id','example_id','condition','source','source_sha256','prompt_sha256')}|{
                    'output':text,'output_tokens':10,'empty':False,'output_limited':False,'copied_source':False}
                (generated if run else reused).append(output)
                audit.append({'job_id':job['job_id'],'input_tokens':20,'input_ids_sha256':'a'*64,'run_required':run,
                              'context_guard':4096 if arm=='base_source_structural_prior' else 1024})
        job_path=package/'upload/milestone_checkpoint_jobs.jsonl';job_path.write_text(''.join(json.dumps(j)+'\n' for j in jobs))
        config={'protocol':runtime.PROTOCOL,'model_id':runtime.MODEL_ID,'model_revision':runtime.REVISION,
            'decoding':runtime.DECODING,'seed':1729,'context_limit':4096,'quantization':'NF4 double quantization; FP32 compute',
            'training_authorized':False,'jobs_sha256':runtime.file_sha(job_path),'example_ids':[str(i) for i in range(8)],
            'reused_prompt_tokens':tokens,'expected_packages':{'torch':'fixture'},'expected_generation_config':{'test':'frozen'},
            'adapter_sha256':{}}
        (package/'upload/milestone_checkpoint_config.json').write_text(json.dumps(config))
        (package/'private_key/reference_key.private.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in refs))
        (package/'private_key/reused_outputs.private.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in reused))
        (package/'private_key/frozen_subset.json').write_text(json.dumps(subset))
        source=Path(runtime.__file__).read_text()
        (package/'plan.json').write_text(json.dumps({'runtime_source_sha256':runtime.sha(source),'initial_calibration_pair_words_estimate':366}))
        (package/'detector_policy.json').write_text(json.dumps({'target_max_project_words':8000}));freeze(package)
        download=root/'download';folder=download/'milestone_checkpoint_4b_v1';folder.mkdir(parents=True)
        report={'complete':True,'training_called':False,'config':config,'generations':generated,'runtime':{
            'runtime_source_sha256':runtime.sha(source),'packages':config['expected_packages'],'model_id':runtime.MODEL_ID,
            'model_revision':runtime.REVISION,'gpu_count':1,'gpu':'Tesla T4','evaluation_references_supplied':False}}
        state={'complete':True,'training_called':False,'optimizer_steps':0,'completed_jobs':14}
        for name,value in [('evaluation.json',report),('state.json',state),('token_audit.json',audit),('generation_config.json',config['expected_generation_config'])]:
            (folder/name).write_text(json.dumps(value))
        (download/'milestone_checkpoint_4b.py').write_text(source)
        return package,download,jobs,config,reused,report,state,subset,refs

    def test_fixed_subset_reader_never_parses_unselected_bodies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);*_,subset,refs=self.fixture(root)
            path=root/'dev.jsonl'
            enriched=[r|{'split':'development','source_sha256':runtime.sha(r['source']),'target_sha256':runtime.sha(r['target']),'source_sentences':[r['source']]} for r in refs]
            path.write_text('Unreadable unselected body\n'+''.join(json.dumps(r)+'\n' for r in enriched))
            self.assertEqual(len(select_frozen(path,subset)),8)
            wrong=copy.deepcopy(subset);wrong[0]['source_sha256']='wrong'
            with self.assertRaises(ValueError):select_frozen(path,wrong)

    def test_cpu_check_coverage_and_prompt_contamination_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            package,download,jobs,config,*_=self.fixture(Path(tmp))
            a=package/'upload/milestone_checkpoint_jobs.jsonl';b=package/'upload/milestone_checkpoint_config.json'
            with patch.object(runtime,'run_training',side_effect=AssertionError('No models')):
                with contextlib.redirect_stdout(io.StringIO()):runtime.main(['--jobs',str(a),'--config',str(b),'--check'])
                with self.assertRaises(ValueError):runtime.main(['--jobs',str(a),'--config',str(b)])
            changed=copy.deepcopy(jobs);changed[0]['reference']='Secret target'
            a.write_text(''.join(json.dumps(j)+'\n' for j in changed));config['jobs_sha256']=runtime.file_sha(a);b.write_text(json.dumps(config))
            with self.assertRaises(ValueError):runtime.load_inputs(a,b)

    def test_reuse_requires_exact_source_reference_and_prompt(self):
        job={'example_id':'a','condition':'base_ordinary','job_id':'a__base_ordinary','source':'Input','source_sha256':runtime.sha('Input'),'prompt_sha256':runtime.sha('Prompt')}
        g={'example_id':'a','source':'Input','reference':'Human','prompt_sha256':job['prompt_sha256'],
           'output':'Saved output','output_tokens':2,'empty':False,'output_limited':False,'copied_source':False}
        self.assertEqual(reuse_generation(job,g,'Human')['output'],'Saved output')
        for key in ('source','reference','prompt_sha256'):
            bad=dict(g);bad[key]='changed'
            with self.assertRaises(ValueError):reuse_generation(job,bad,'Human')

    def test_structural_prompt_does_not_use_evaluation_target(self):
        row={'example_id':'a','split':'development','source':'First claim. Second claim.','source_sentences':['First claim.','Second claim.'],'target':'REFERENCE_ONE'}
        style={'reference_sentence_words':{'p25':16,'p75':34},'reference_semicolons_per_100_words':{'p25':0,'p75':0}}
        a=make_job(row,style,'source_structural_prior');row['target']='DIFFERENT_REFERENCE'
        self.assertEqual(a,make_job(row,style,'source_structural_prior'))
        self.assertNotIn('REFERENCE_ONE',a['prompt'])

    def test_incomplete_outputs_block_exports_and_complete_export_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);package,download,jobs,config,reused,report,state,*_=self.fixture(root)
            bad=dict(state);bad['complete']=False
            with self.assertRaises(ValueError):collect_complete(jobs,reused,report,bad,config)
            broken=copy.deepcopy(report);broken['generations'].pop()
            with self.assertRaises(ValueError):collect_complete(jobs,reused,broken,state,config)
            summary=export(package,download,root/'a');export(package,download,root/'b')
            self.assertEqual(summary['detector_slots'],40);self.assertEqual(summary['model_outputs'],24)
            self.assertEqual(summary['unique_detector_texts'],39);self.assertEqual(summary['detector_calls_made'],0)
            for p in (root/'a').rglob('*'):
                if p.is_file():self.assertEqual(p.read_bytes(),(root/'b'/p.relative_to(root/'a')).read_bytes())
            with (root/'a/detector/results_template.csv').open() as f:entries=list(csv.DictReader(f))
            self.assertTrue(all(e['status']=='not_tested' and not e['provider_label'] and not e['score'] for e in entries))
            with self.assertRaises(ValueError):export(package,download,root/'a')

    def test_notebook_is_self_contained_and_has_no_training_calls(self):
        source=Path(runtime.__file__).read_text();tree=ast.parse(source)
        for n in ast.walk(tree):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):self.assertNotIn(n.func.attr,{'train','backward','step','zero_grad'})
        for n in tree.body:
            if isinstance(n,(ast.Import,ast.ImportFrom)):
                names=[a.name for a in n.names] if isinstance(n,ast.Import) else [n.module]
                self.assertFalse(any(s.split('.')[0] in ('torch','transformers','peft') for s in names))
        path=ROOT/'kaggle/milestone_checkpoint_4b.ipynb';book=json.loads(path.read_text())
        self.assertIn(source,'\n'.join(''.join(c['source']) for c in book['cells']));self.assertLess(path.stat().st_size,1_000_000)
        for c in book['cells']:
            if c['cell_type']=='code':self.assertIsNone(c['execution_count']);self.assertEqual(c['outputs'],[])


if __name__=='__main__':unittest.main()
