"""Synthetic CPU checks for plan leakage, freezing, equal conditions and scoring."""
import copy
import json
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.rhetoric.concise_plan import (synthetic_pack,select_four,jobs_for,render_plan,
    validate_plan,plan_metrics,evaluate,audit_tokens,main,CONDITIONS,CONTEXT_LIMIT,MAX_NEW_TOKENS)


def outputs_for(pack,jobs):
    sources={r['id']:r['source'] for r in pack['rows']}
    return [{'id':j['id'],'condition':j['condition'],'prompt_sha256':j['prompt_sha256'],
             'output':sources[j['id']],'hit_output_limit':False,'synthetic':True} for j in jobs]


class ConcisePlanTests(unittest.TestCase):
    def test_only_closed_plan_fields_and_no_reference_wording(self):
        pack=synthetic_pack();pack['rows'][0]['reference']='REFERENCE_SENTINEL_MUST_NOT_REACH_GENERATION'
        jobs=jobs_for(pack)
        self.assertEqual(len(jobs),8)
        self.assertNotIn('REFERENCE_SENTINEL',json.dumps(jobs))
        self.assertTrue(all(set(j)=={'id','condition','prompt','prompt_sha256','source_sha256'} for j in jobs))
        plan=copy.deepcopy(pack['rows'][0]['plan']);plan['reference_words']='Forbidden content'
        with self.assertRaises(ValueError):validate_plan(plan)
        plan=copy.deepcopy(pack['rows'][0]['plan']);plan['sentences'][0]['shape_hint']='Invent a causal explanation'
        with self.assertRaises(ValueError):render_plan(plan)

    def test_training_only_balanced_selection_and_equal_prompt_prefixes(self):
        pack=synthetic_pack();selected=select_four(pack['rows'])
        self.assertEqual(len(selected),4)
        self.assertEqual(sum(r['plan']['sentence_count']>r['source_sentence_count'] for r in selected),2)
        changed=copy.deepcopy(pack['rows']);changed[0]['split']='validation'
        with self.assertRaises(ValueError):select_four(changed)
        jobs=jobs_for(pack)
        for row in pack['rows']:
            pair={j['condition']:j for j in jobs if j['id']==row['id']}
            self.assertTrue(pair[CONDITIONS[1]]['prompt'].startswith(pair[CONDITIONS[0]]['prompt']))

    def test_partial_length_zip_does_not_claim_perfect_plan(self):
        plan=synthetic_pack()['rows'][0]['plan']
        metrics=plan_metrics('Firms report revenue.',plan)
        self.assertFalse(metrics['sentence_count_match'])
        self.assertLess(metrics['word_band_match'],1)

    def test_evaluation_rejects_missing_swapped_cutoff_and_duplicate_jobs(self):
        pack=synthetic_pack();jobs=jobs_for(pack);outputs=outputs_for(pack,jobs)
        records,summary=evaluate(pack,jobs,outputs,synthetic=True)
        self.assertEqual(len(records),8);self.assertEqual(summary['execution_kind'],'synthetic_wiring_check')
        self.assertIsNone(summary['conditions'][CONDITIONS[0]]['nli_proxy_passes'])
        self.assertFalse(summary['full_mapping_training_ready'])
        for bad in (outputs[:-1],outputs+[outputs[0]]):
            with self.assertRaises(ValueError):evaluate(pack,jobs,bad,synthetic=True)
        bad=copy.deepcopy(outputs);bad[0]['hit_output_limit']=True
        with self.assertRaises(ValueError):evaluate(pack,jobs,bad,synthetic=True)
        bad=copy.deepcopy(outputs);bad[0]['prompt_sha256']='changed'
        with self.assertRaises(ValueError):evaluate(pack,jobs,bad,synthetic=True)
        with self.assertRaises(ValueError):evaluate(pack,jobs+[jobs[0]],outputs,synthetic=True)

    def test_context_and_dry_run_never_access_secret_or_model(self):
        job=jobs_for(synthetic_pack())[0]
        self.assertEqual(audit_tokens(job,[1]*(CONTEXT_LIMIT-MAX_NEW_TOKENS))['input_tokens'],CONTEXT_LIMIT-MAX_NEW_TOKENS)
        with self.assertRaises(ValueError):audit_tokens(job,[1]*(CONTEXT_LIMIT-MAX_NEW_TOKENS+1))
        with tempfile.TemporaryDirectory() as d,patch('fama_style.rhetoric.concise_plan.secret_token',side_effect=AssertionError('Secret access forbidden')):
            main(['--dry-run','--output-dir',str(Path(d)/'dry')])
            with self.assertRaises(ValueError):main(['--dry-run','--output-dir',str(Path(d)/'dry')])


class ConciseNotebookTests(unittest.TestCase):
    def test_export_is_standalone_and_current_with_cpu_dry_run(self):
        root=Path(__file__).resolve().parents[1];path=root/'kaggle/concise_plan_realization.ipynb'
        book=json.loads(path.read_text());self.assertEqual(len(book['cells']),6)
        self.assertTrue(all(not c.get('outputs') and c.get('execution_count') is None for c in book['cells']))
        for cell in book['cells']:
            source=''.join(cell['source'])
            if cell['cell_type']=='code' and not source.startswith('%'):compile(source,'notebook_cell','exec')
        setup=''.join(book['cells'][3]['source'])
        with tempfile.TemporaryDirectory() as d:
            runtime=Path(d)/'src';setup=setup.replace('/kaggle/working/_concise_plan_runtime/src',str(runtime))
            namespace={};exec(setup,namespace)
            for name,source in namespace['files'].items():self.assertEqual(source,(root/'src'/name).read_text())
            result=subprocess.run([sys.executable,'-m','fama_style.rhetoric.concise_plan','--dry-run','--output-dir',str(Path(d)/'dry')],
                cwd=d,env={**os.environ,'PYTHONPATH':str(runtime)},capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('synthetic_wiring_check_complete',result.stdout)


if __name__=='__main__':unittest.main()
