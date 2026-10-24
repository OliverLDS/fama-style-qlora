"""Matched candidate generation contracts, without ML weights or secrets."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.rhetoric import style_tradeoff as e


class StyleTradeoffTests(unittest.TestCase):
    def outputs(self,pack):
        source={r['id']:r['source'] for r in pack['rows']}
        return [{**{k:j[k] for k in ('id','condition','seed','prompt_sha256')},'output':source[j['id']],
                 'hit_output_limit':False,'synthetic':True} for j in e.jobs_for(pack)]

    def test_references_never_enter_prompts_and_sampling_is_matched(self):
        pack=e.synthetic_pack();jobs=e.jobs_for(pack)
        self.assertEqual(len(jobs),24)
        self.assertEqual(len({(j['id'],j['condition'],j['seed']) for j in jobs}),24)
        changed=copy.deepcopy(pack)
        for r in changed['rows']:r['reference']='REFERENCE_SENTINEL'
        self.assertEqual(jobs,e.jobs_for(changed))
        for r in pack['rows']:
            self.assertEqual({j['seed'] for j in jobs if j['id']==r['id'] and j['condition']=='A_direct'},set(e.SEEDS))
        changed['rows'][0]['split']='test'
        with self.assertRaises(ValueError):e.jobs_for(changed)

    def test_operations_use_empirical_support_and_existing_source_relations(self):
        prior={'features':{'explicit_contrast':{'source_present':80,'reference_present':118},
                           'author_voice':{'source_present':61,'reference_present':83}}}
        self.assertEqual(len(e.operations('We invest, but banks lend.',prior)),2)
        self.assertEqual(e.operations('Firms invest.',prior),[])
        prior['features']['explicit_contrast']['source_present']=118
        self.assertEqual([o['operation'] for o in e.operations('We invest, but banks lend.',prior)],['author_voice'])
        with self.assertRaises(ValueError):e.empirical_prior([{'split':'validation'}]*247)

    def test_output_sets_cannot_be_swapped_truncated_or_silently_completed(self):
        pack=e.synthetic_pack();outs=self.outputs(pack)
        rows,summary=e.evaluate(pack,outs,synthetic=True)
        self.assertEqual(len(rows),24);self.assertFalse(summary['training_called'])
        self.assertIsNone(summary['semantic_accuracy'])
        self.assertTrue(all(r['exact_copy'] for r in rows))
        for mut in ('duplicate','cutoff','empty','missing'):
            out=copy.deepcopy(outs)
            if mut=='duplicate':out[-1]=out[0]
            if mut=='cutoff':out[0]['hit_output_limit']=True
            if mut=='empty':out[0]['output']=' '
            if mut=='missing':out.pop()
            with self.assertRaises(ValueError,msg=mut):e.evaluate(pack,out,synthetic=True)
        with self.assertRaises(ValueError):e.evaluate(pack,outs,synthetic=False)

    def test_risk_warnings_remain_in_scores_without_a_binary_gate(self):
        pack=e.synthetic_pack();outs=self.outputs(pack)
        for o in outs:o.update(output='We invest; however, banks may lend.',synthetic=False)
        semantic={(o['id'],o['condition'],o['seed']):{'status':'risk_signal'} for o in outs}
        rows,summary=e.evaluate(pack,outs,semantic)
        self.assertEqual(len(rows),24)
        self.assertTrue(all(r['old_verifier_status']=='risk_signal' for r in rows))
        self.assertTrue(all(r['true_drift_severity']=='unknown' for r in rows))
        self.assertEqual(summary['conditions']['A_direct']['outputs'],12)

    def test_context_preflight_does_not_truncate(self):
        j=e.jobs_for(e.synthetic_pack())[0]
        self.assertEqual(e.audit_tokens(j,[1])['input_tokens'],1)
        with self.assertRaises(ValueError):e.audit_tokens(j,[1]*4096)
        j['prompt']+='changed'
        with self.assertRaises(ValueError):e.audit_tokens(j,[1])

    def test_dry_run_never_enters_runtime_or_reads_a_secret(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(e,'run_training',side_effect=AssertionError('runtime')),patch.object(e,'secret_token',side_effect=AssertionError('secret')):
            e.main(['--dry-run','--output-dir',str(Path(folder)/'out')])
            result=json.loads((Path(folder)/'out/synthetic_evaluation.json').read_text())
            self.assertEqual(result['status'],'synthetic_wiring_check')

    def test_export_is_self_contained_blank_and_has_no_real_paragraphs(self):
        path=e.ROOT/'kaggle/style_tradeoff.ipynb'
        if not path.exists():self.skipTest('Notebook not built yet')
        book=json.loads(path.read_text());code='\n'.join(''.join(c['source']) for c in book['cells'] if c['cell_type']=='code')
        self.assertTrue(all(not c.get('outputs') and c.get('execution_count') is None for c in book['cells']))
        self.assertIn('--execute-inference',code);self.assertNotIn('hf_'+('x'*30),code)
        self.assertIn('fama_style/rhetoric/style_tradeoff.py',code)
        for c in book['cells']:
            if c['cell_type']=='code' and not ''.join(c['source']).startswith('%'):compile(''.join(c['source']),'notebook','exec')
        # Execute only the embedded-source installation and CPU preflight cells.
        with tempfile.TemporaryDirectory() as folder:
            setup=''.join(book['cells'][3]['source']).replace('/kaggle/working/_style_tradeoff_runtime',folder+'/runtime')
            preflight=''.join(book['cells'][4]['source']).replace('list(Path("/kaggle/input").rglob("style_tradeoff_pack.json"))',repr([str(e.ROOT/'research_runs/style_tradeoff_candidates_v1/input_pack/style_tradeoff_pack.json')]))
            preflight=preflight.replace('str(matches[0])','str(matches[0])').replace('/kaggle/working/style_tradeoff_preflight',folder+'/preflight')
            if not (e.ROOT/'research_runs/style_tradeoff_candidates_v1/input_pack/style_tradeoff_pack.json').exists():return
            run=subprocess.run([sys.executable,'-c',setup+'\n'+preflight],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertTrue((Path(folder)/'preflight/preflight.json').exists())


if __name__=='__main__':unittest.main()
