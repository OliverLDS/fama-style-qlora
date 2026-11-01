"""CPU contracts for matched, prose-only 4B inference."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fama_style.evaluation import reconstruction_prose_4b as runtime
from fama_style.research.local_reconstruction import make_jobs

ROOT=Path(__file__).resolve().parents[1]


class ReconstructionProse4B(unittest.TestCase):
    def jobs(self):
        base=[]
        for i in range(3):
            rid='SYN'+str(i)
            brief={'actors':[{'actor_id':'A','name':'firms'}],
                   'claims':[{'claim_id':'C','actor_ids':['A'],'predicate':'investment',
                              'proposition':'Firms may invest.','modality':['may']}],
                   'supported_relationships':[],'essential_terms':[],'citations':[],'context_gaps':[]}
            for method,source in [('direct_rewrite','PRIVATE_REFERENCE_'+rid),
                                  ('neutral_reconstruction',json.dumps({'record_id':rid,'content_brief':brief}))]:
                base.append({'record_id':rid,'method':method,'source':source})
        return make_jobs(base)

    def save(self,root,jobs):
        jobpath=root/'jobs.jsonl';jobpath.write_text(''.join(json.dumps(r)+'\n' for r in jobs))
        config={'protocol':runtime.PROTOCOL,'model_id':runtime.MODEL_ID,'model_revision':runtime.MODEL_REVISION,
                'decoding':runtime.DECODING,'seed':1729,'quantization':'NF4 double quantization; FP32 compute',
                'record_ids':sorted({r['record_id'] for r in jobs}),'jobs_sha256':runtime.sha(jobpath.read_text())}
        path=root/'config.json';path.write_text(json.dumps(config));return jobpath,path

    def test_nine_jobs_checked_without_model_or_gpu_execution(self):
        jobs=self.jobs()
        with tempfile.TemporaryDirectory() as folder:
            a,b=self.save(Path(folder),jobs)
            with patch.object(runtime,'run_training',side_effect=AssertionError('No inference in CPU check')):
                with contextlib.redirect_stdout(io.StringIO()):runtime.main(['--jobs',str(a),'--config',str(b),'--check'])
            for row in jobs:
                if row['method']!='direct_rewrite':self.assertNotIn('PRIVATE_REFERENCE',row['prompt'])

    def test_revision_precision_and_decoding_changes_rejected(self):
        for key,value in [('model_revision','a'*40),('decoding',{}),('quantization','FP16')]:
            with tempfile.TemporaryDirectory() as folder:
                a,b=self.save(Path(folder),self.jobs());config=json.loads(b.read_text())
                config[key]=value;b.write_text(json.dumps(config))
                with self.assertRaises(ValueError):runtime.load_inputs(a,b)

    def test_missing_duplicate_or_extra_reference_fields_rejected(self):
        jobs=self.jobs()
        for changed in (jobs[:-1],jobs[:-1]+[jobs[0]], [jobs[0]|{'reference':'unexpected'}]+jobs[1:]):
            with tempfile.TemporaryDirectory() as folder:
                a,b=self.save(Path(folder),changed)
                with self.assertRaises(ValueError):runtime.load_inputs(a,b)

    def test_job_or_prompt_byte_changes_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            a,b=self.save(Path(folder),self.jobs());a.write_text(a.read_text()+'\n')
            with self.assertRaises(ValueError):runtime.load_inputs(a,b)
        jobs=self.jobs();jobs[0]['prompt']+=' modified'
        with tempfile.TemporaryDirectory() as folder:
            a,b=self.save(Path(folder),jobs)
            with self.assertRaises(ValueError):runtime.load_inputs(a,b)

    def test_prose_flags_do_not_rewrite_outputs_or_require_json(self):
        text='A synthetic economics paragraph.'
        self.assertEqual(runtime.classify_output(text),{'status':'generated','format_flags':[]})
        self.assertEqual(runtime.classify_output('BLOCKED: Missing context.')['status'],'blocked')
        self.assertEqual(runtime.classify_output('  ')['status'],'empty')
        self.assertIn('blocked_without_reason',runtime.classify_output('BLOCKED:')['format_flags'])
        self.assertIn('markdown_fence',runtime.classify_output('```text\n'+text+'\n```')['format_flags'])
        self.assertIn('json_instead_of_prose',runtime.classify_output('{"text":"Synthetic"}')['format_flags'])
        self.assertEqual(text,'A synthetic economics paragraph.')

    def test_context_budget_is_checked_without_truncating(self):
        runtime.check_budget(4096-512)
        with self.assertRaises(ValueError):runtime.check_budget(4096-511)

    def test_notebook_self_contained_private_output_free_and_below_limit(self):
        path=ROOT/'kaggle/reconstruction_prose_4b.ipynb';book=json.loads(path.read_text())
        source='\n'.join(''.join(c['source']) for c in book['cells'])
        self.assertIn((ROOT/'src/fama_style/evaluation/reconstruction_prose_4b.py').read_text(),source)
        self.assertLess(path.stat().st_size,1_000_000)
        for cell in book['cells']:
            if cell['cell_type']=='code':
                self.assertIsNone(cell['execution_count']);self.assertEqual(cell['outputs'],[])


if __name__=='__main__':unittest.main()
