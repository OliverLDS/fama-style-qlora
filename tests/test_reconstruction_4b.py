"""CPU contracts for paired Gemma 4B inference; no weights or GPU needed."""
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

from fama_style.evaluation import reconstruction_4b as runtime
from fama_style.research.reconstruction_4b_packet import make_jobs

ROOT=Path(__file__).resolve().parents[1]


class Reconstruction4B(unittest.TestCase):
    def jobs(self):
        refs=[{'record_id':'synthetic'+str(i),'original_paragraph':'PRIVATE_REFERENCE_'+str(i),
               'authors':['PRIVATE_AUTHOR'],'paper_id':'PRIVATE_PAPER'} for i in range(3)]
        briefs={r['record_id']:{'record_id':r['record_id'],'claims':['Synthetic economic claim.']} for r in refs}
        return refs,briefs,make_jobs(refs,briefs)

    def save(self,path,jobs):
        jobs_path=path/'jobs.jsonl';jobs_path.write_text(''.join(json.dumps(r)+'\n' for r in jobs))
        config={'protocol':runtime.PROTOCOL,'model_id':runtime.MODEL_ID,'seed':1729,
                'decoding':runtime.DECODING,'quantization':'NF4 double quantization; FP32 compute',
                'record_ids':sorted({r['record_id'] for r in jobs}),'jobs_sha256':runtime.sha(jobs_path.read_text())}
        config_path=path/'config.json';config_path.write_text(json.dumps(config))
        return jobs_path,config_path

    def test_reference_metadata_cannot_enter_reconstruction_prompts(self):
        refs,briefs,jobs=self.jobs();before=copy.deepcopy((refs,briefs))
        self.assertEqual(len(jobs),6)
        for job in jobs:
            if job['method']=='neutral_reconstruction':
                self.assertNotIn('PRIVATE_',job['prompt'])
                self.assertEqual(set(json.loads(job['source'])),{'record_id','content_brief'})
            self.assertIn(runtime.COMMON,job['prompt'])
        self.assertEqual((refs,briefs),before)

    def test_matching_coverage_and_no_gpu_check(self):
        _,_,jobs=self.jobs()
        with tempfile.TemporaryDirectory() as folder:
            a,b=self.save(Path(folder),jobs)
            with patch.object(runtime,'run_training',side_effect=AssertionError('GPU must not run')):
                with contextlib.redirect_stdout(io.StringIO()):runtime.main(['--jobs',str(a),'--config',str(b),'--check'])
            a.write_text(a.read_text()+'\n')
            with self.assertRaises(ValueError):runtime.load_inputs(a,b)

    def test_prompt_and_duplicate_pair_tampering_rejected_even_after_rehash(self):
        for change in ('prompt','duplicate'):
            _,_,jobs=self.jobs()
            if change=='prompt':jobs[0]['prompt']+=' EXTRA'
            else:jobs[-1]=jobs[0].copy()
            with tempfile.TemporaryDirectory() as folder:
                a,b=self.save(Path(folder),jobs)
                with self.assertRaises(ValueError):runtime.load_inputs(a,b)

    def test_raw_format_errors_and_blocked_cases_not_repaired_or_dropped(self):
        job={'record_id':'synthetic','method':'neutral_reconstruction'}
        row=job|{'status':'blocked','text':None,'reason':'Missing condition.'}
        self.assertEqual(runtime.parse_output(json.dumps(row),job)['status'],'blocked')
        self.assertEqual(runtime.parse_output('```json\n'+json.dumps(row)+'\n```',job)['status'],'blocked')
        for text in ('Commentary\n```json\n'+json.dumps(row)+'\n```',json.dumps(row|{'record_id':'wrong'}),
                     json.dumps(row|{'reason':''}),json.dumps([row])):
            self.assertEqual(runtime.parse_output(text,job)['status'],'format_error')
        row.update(status='generated',text='Synthetic output.',reason='')
        self.assertEqual(runtime.parse_output(json.dumps(row),job)['status'],'generated')

    def test_no_prompt_truncation(self):
        runtime.check_budget(runtime.DECODING['context_limit']-runtime.DECODING['max_new_tokens'])
        with self.assertRaises(ValueError):runtime.check_budget(runtime.DECODING['context_limit'])

    def test_kaggle_reads_secret_without_overwriting_environment_or_fallback(self):
        fake=types.ModuleType('kaggle_secrets');client=types.SimpleNamespace(get_secret=lambda key:'synthetic-secret')
        fake.UserSecretsClient=lambda:client
        with patch.dict(sys.modules,{'kaggle_secrets':fake}),patch.object(runtime.Path,'exists',return_value=True):
            with patch.dict(runtime.os.environ,{'HF_TOKEN':'synthetic-environment-value'}):
                self.assertEqual(runtime.load_hf_token(),'synthetic-secret')
                self.assertEqual(runtime.os.environ['HF_TOKEN'],'synthetic-environment-value')
                def unavailable(key):raise ValueError('secret unavailable')
                client.get_secret=unavailable
                with self.assertRaises(RuntimeError):runtime.load_hf_token()

    def test_notebook_embeds_runtime_and_has_no_outputs_or_private_content(self):
        book=json.loads((ROOT/'kaggle/reconstruction_4b.ipynb').read_text())
        text='\n'.join(''.join(cell['source']) for cell in book['cells'])
        self.assertIn((ROOT/'src/fama_style/evaluation/reconstruction_4b.py').read_text(),text)
        self.assertLess((ROOT/'kaggle/reconstruction_4b.ipynb').stat().st_size,1_000_000)
        for cell in book['cells']:
            if cell['cell_type']=='code':
                self.assertIsNone(cell['execution_count']);self.assertEqual(cell['outputs'],[])


if __name__=='__main__':unittest.main()
