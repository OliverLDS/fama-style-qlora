"""Factor isolation, marker diagnostics and standalone CPU checks."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.rhetoric import ledger_ablation as e


class LedgerAblationTests(unittest.TestCase):
    def test_factorial_prompts_preserve_rich_controls_and_reference_exclusion(self):
        pack=e.synthetic_pack();jobs=e.jobs_for(pack);self.assertEqual(len(jobs),8)
        for row in pack['rows']:
            group={j['condition']:j['prompt'] for j in jobs if j['id']==row['id']}
            self.assertEqual(group['C_cue_rich'],row['parent_prompts']['no_grouping'])
            self.assertEqual(group['D_cue_rich_grouping'],row['parent_prompts']['grouping'])
            rich=e.constraint_ledger(row['source']);simple=copy.deepcopy(rich)
            for claim in simple['claims']:del claim['protected_evidence']
            dumps=lambda x:json.dumps(x,ensure_ascii=False,separators=(',',':'))
            self.assertEqual(group['A_sentence_only'],group['C_cue_rich'].replace(dumps(rich),dumps(simple),1))
            self.assertEqual(group['B_sentence_grouping'],group['D_cue_rich_grouping'].replace(dumps(rich),dumps(simple),1))
            self.assertTrue(group['B_sentence_grouping'].startswith(group['A_sentence_only']))
            self.assertTrue(group['D_cue_rich_grouping'].startswith(group['C_cue_rich']))
        changed=copy.deepcopy(pack)
        for row in changed['rows']:row['reference']='REFERENCE_SENTINEL'
        self.assertEqual(jobs,e.jobs_for(changed));self.assertNotIn('REFERENCE_SENTINEL',json.dumps(e.jobs_for(changed)))
        changed['rows'][0]['split']='test'
        with self.assertRaises(ValueError):e.jobs_for(changed)

    def test_markers_relative_to_source_are_warnings_not_semantic_proof(self):
        source='The authors (1974) suggest that firms may invest. Some firms hold cash.'
        text='This has a modality of ‘may’, quantitative evidence of 1974, and ‘some’ firms.'
        result=e.artifact_checks(source,text)
        self.assertTrue(result['artifact_warning']);self.assertEqual(result['newly_quoted_protected_cues'],['may','some'])
        self.assertIn('modality',result['new_metalinguistic_terms']);self.assertTrue(result['new_numeric_evidence_phrases'])
        self.assertFalse(result['semantic_proof']);self.assertFalse(e.artifact_checks(source,source)['artifact_warning'])
        quoted='The source uses the term modality and says ‘may’.'
        self.assertFalse(e.artifact_checks(quoted,quoted)['artifact_warning'])
        self.assertFalse(e.artifact_checks(source,'The authors (1974) suggest firms could invest.')['artifact_warning'])

    def fixture(self):
        p=e.synthetic_pack();jobs=e.jobs_for(p);rows={r['id']:r for r in p['rows']}
        outputs=[{'id':j['id'],'condition':j['condition'],'prompt_sha256':j['prompt_sha256'],'output':rows[j['id']]['source'],
            'hit_output_limit':False,'synthetic':False} for j in jobs]
        semantic={(r['id'],r['condition']):{'status':'model_consistent','semantic_proof':False} for r in outputs}
        return p,jobs,outputs,semantic

    def test_control_replay_and_unknowns_are_reported(self):
        p,jobs,outputs,semantic=self.fixture()
        key=(outputs[0]['id'],outputs[0]['condition']);semantic[key]={'status':'abstain','semantic_proof':False}
        records,summary=e.evaluate(p,jobs,outputs,semantic,canary_passed=True)
        self.assertEqual(len(records),8);self.assertEqual(len(summary['within_case_contrasts']),8)
        self.assertTrue(all(c['rows']==2 for c in summary['conditions'].values()))
        self.assertEqual(summary['conditions']['C_cue_rich']['rich_control_hash_matches'],2)
        self.assertEqual(summary['conditions']['A_sentence_only']['verifier_status_counts']['abstain'],1)
        self.assertFalse(summary['training_ready']);self.assertTrue(summary['selection_post_hoc'])
        _,summary=e.evaluate(p,jobs,outputs,semantic,canary_passed=False)
        self.assertIsNone(summary['conditions']['A_sentence_only']['valid_semantic_proxy_passes'])

    def test_missing_swapped_cutoff_and_changed_prompts_fail(self):
        p,jobs,outputs,semantic=self.fixture()
        missing=dict(semantic);missing.pop(next(iter(missing)))
        with self.assertRaises(ValueError):e.evaluate(p,jobs,outputs,missing,canary_passed=True)
        for key,value in (('hit_output_limit',True),('prompt_sha256','wrong'),('synthetic',True)):
            modified=copy.deepcopy(outputs);modified[0][key]=value
            with self.assertRaises(ValueError):e.evaluate(p,jobs,modified,semantic,canary_passed=True)
        modified=copy.deepcopy(outputs);modified[1]=modified[0]
        with self.assertRaises(ValueError):e.evaluate(p,jobs,modified,semantic,canary_passed=True)
        changed=copy.deepcopy(p);changed['rows'][0]['parent_prompts']['grouping']='changed'
        with self.assertRaises(ValueError):e.jobs_for(changed)

    def test_standalone_export_dry_runs_without_private_data(self):
        root=Path(__file__).resolve().parents[1];book=json.loads((root/'kaggle/ledger_grouping_ablation.ipynb').read_text())
        setup=next(''.join(c['source']) for c in book['cells'] if c['cell_type']=='code' and 'files = json.loads' in ''.join(c['source']))
        self.assertTrue(all(not c.get('outputs') for c in book['cells']))
        with tempfile.TemporaryDirectory() as tmp:
            runtime=Path(tmp)/'runtime';ns={};exec(setup.replace('/kaggle/working/_ledger_ablation_runtime',str(runtime)),ns)
            for name,source in ns['files'].items():self.assertEqual(source,(root/name).read_text())
            result=subprocess.run([sys.executable,'-m','fama_style.rhetoric.ledger_ablation','--dry-run','--output-dir',str(Path(tmp)/'dry')],cwd=tmp,
                env={**os.environ,'PYTHONPATH':str(runtime/'src'),'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr);self.assertIn('8 jobs; no model calls',result.stdout)


if __name__=='__main__':unittest.main()
