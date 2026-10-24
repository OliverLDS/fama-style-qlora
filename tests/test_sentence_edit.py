"""Structural edit contracts and honest fallback accounting, without ML weights."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.rhetoric import sentence_edit as e

ENTAIL={'entailment':.99,'neutral':.01,'contradiction':0.}


class SentenceEditTests(unittest.TestCase):
    def test_baseline_is_exact_and_references_are_never_in_prompts(self):
        pack=e.synthetic_pack();jobs=e.jobs_for(pack);self.assertEqual(len(jobs),4)
        for row in pack['rows']:
            group={j['condition']:j['prompt'] for j in jobs if j['id']==row['id']}
            self.assertEqual(group['A_baseline'],row['baseline_prompt'])
            self.assertTrue(group['B_edit_script'].startswith(group['A_baseline']))
            self.assertIn('No new IDs, deletion, reordering',group['B_edit_script'])
        changed=copy.deepcopy(pack)
        for r in changed['rows']:r['reference']='REFERENCE_SENTINEL'
        self.assertEqual(jobs,e.jobs_for(changed));self.assertNotIn('REFERENCE_SENTINEL',json.dumps(e.jobs_for(changed)))
        changed['rows'][0]['split']='test'
        with self.assertRaises(ValueError):e.jobs_for(changed)

    def test_edit_order_cannot_reorder_source_and_boundary_count_is_preserved(self):
        source='Firms invest. Banks lend.'
        raw=json.dumps({'edits':[{'claim_id':'c2','replacement':'Banks finance firms.'},{'claim_id':'c1','replacement':'Firms buy assets.'}]})
        p=e.parse_edits(source,raw);self.assertTrue(p['format_valid'])
        self.assertEqual(p['proposed_paragraph'],'Firms buy assets. Banks finance firms.')
        self.assertEqual(len(e.sentences(p['proposed_paragraph'])),2)
        fenced=e.parse_edits(source,'```json\n'+raw+'\n```')
        self.assertTrue(fenced['format_valid']);self.assertFalse(fenced['raw_contract_valid']);self.assertTrue(fenced['outer_fence_removed'])

    def test_invalid_scripts_are_not_repaired_or_turned_into_source_copies(self):
        source='Firms invest. Banks lend.'
        bad=[{'edits':[{'claim_id':'c3','replacement':'New facts.'}]},
             {'edits':[{'claim_id':'c1','replacement':'Firms buy assets.'},{'claim_id':'c1','replacement':'Firms invest.'}]},
             {'edits':[{'claim_id':'c1','replacement':''}]},
             {'edits':[{'claim_id':'c1','replacement':'Firms invest. Banks save.'}]},
             {'edits':[{'claim_id':'c1','replacement':'Firms invest. banks save.'}]},
             {'edits':[{'claim_id':'c1','replacement':'Firms buy assets','reason':'extra field'}]},
             {'edits':[],'paragraph':'extra'}]
        for obj in bad:
            p=e.parse_edits(source,json.dumps(obj));self.assertFalse(p['format_valid'],obj)
            result=e.assess_edits(source,p,lambda a,b:1/0)
            self.assertIsNone(result['proposed_paragraph']);self.assertIsNone(result['guarded_paragraph'])
        for raw in ('prefix {"edits":[]}','{"edits":[],"edits":[]}','```\n{"edits":[]}\n```'):
            self.assertFalse(e.parse_edits(source,raw)['format_valid'])

    def test_rejected_and_unknown_edits_retain_original_text(self):
        source='Firms invest. Banks lend.'
        p=e.parse_edits(source,json.dumps({'edits':[{'claim_id':'c1','replacement':'Firms buy assets.'},{'claim_id':'c2','replacement':'Banks cause a crisis.'}]}))
        with patch.object(e.verifier,'verify',side_effect=[{'status':'model_consistent','semantic_proof':False},{'status':'risk_signal','semantic_proof':False}]):
            a=e.assess_edits(source,p,lambda a,b:ENTAIL)
        self.assertEqual(a['guarded_paragraph'],'Firms buy assets. Banks lend.')
        self.assertEqual(a['proposed_nontrivial_edits'],2);self.assertEqual(a['accepted_nontrivial_edits'],1);self.assertEqual(a['rejected_risk_edits'],1)
        unknown=e.assess_edits(source,p,lambda a,b:None)
        self.assertEqual(unknown['guarded_paragraph'],source);self.assertEqual(unknown['abstained_edits'],2)
        noops=e.assess_edits(source,e.parse_edits(source,'{"edits":[{"claim_id":"c1","replacement":"Firms invest."}]}'),lambda a,b:1/0)
        self.assertEqual(noops['noop_edits'],1);self.assertEqual(noops['accepted_nontrivial_edits'],0)

    def fixture(self,invalid=False):
        p=e.synthetic_pack();jobs=e.jobs_for(p);rows={r['id']:r for r in p['rows']}
        raw='{"bad":[]}' if invalid else '{"edits":[]}'
        outputs=[{'id':j['id'],'condition':j['condition'],'prompt_sha256':j['prompt_sha256'],'output':rows[j['id']]['source'] if j['condition']=='A_baseline' else raw,'hit_output_limit':False,'synthetic':False} for j in jobs]
        patches={r['id']:e.assess_edits(r['source'],e.parse_edits(r['source'],raw),lambda a,b:None) for r in p['rows']}
        semantic={(r['id'],v):{'status':'model_consistent','semantic_proof':False} for r in p['rows'] for v in (('A_baseline',) if invalid else e.VARIANTS)}
        return p,jobs,outputs,patches,semantic

    def test_no_edits_and_fallback_copies_are_not_style_successes(self):
        p,jobs,out,patches,semantic=self.fixture()
        records,summary=e.evaluate(p,jobs,out,semantic,patch_checks=patches,canary_passed=True)
        self.assertEqual(len(records),6);self.assertEqual(summary['generated_jobs'],4)
        g=summary['variants']['B_guarded'];self.assertEqual(g['exact_copies_available_only'],2);self.assertEqual(g['joint_proxy_successes'],0)
        self.assertEqual(g['source_sentence_count_matches'],2);self.assertFalse(summary['training_ready'])
        p,jobs,out,patches,semantic=self.fixture(invalid=True)
        _,summary=e.evaluate(p,jobs,out,semantic,patch_checks=patches,canary_passed=True)
        self.assertEqual(summary['variants']['B_proposed']['available_paragraphs'],0)
        self.assertEqual(summary['variants']['B_guarded']['verifier_status_counts']['invalid_script'],2)
        self.assertIsNone(summary['variants']['B_guarded']['mean_style_gain_proxy_available_only'])

    def test_output_identity_cutoff_and_derived_text_guards(self):
        p,jobs,out,patches,semantic=self.fixture()
        for field,value in (('hit_output_limit',True),('synthetic',True),('prompt_sha256','wrong')):
            changed=copy.deepcopy(out);changed[0][field]=value
            with self.assertRaises(ValueError):e.evaluate(p,jobs,changed,semantic,patch_checks=patches,canary_passed=True)
        missing=dict(semantic);missing.pop(next(iter(missing)))
        with self.assertRaises(ValueError):e.evaluate(p,jobs,out,missing,patch_checks=patches,canary_passed=True)
        changed=copy.deepcopy(patches);changed[p['rows'][0]['id']]['guarded_paragraph']='Invented text.'
        with self.assertRaises(ValueError):e.evaluate(p,jobs,out,semantic,patch_checks=changed,canary_passed=True)
        _,summary=e.evaluate(p,jobs,out,semantic,patch_checks=patches,canary_passed=False)
        self.assertIsNone(summary['variants']['B_guarded']['valid_semantic_proxy_passes'])

    def test_standalone_notebook_dry_run(self):
        root=Path(__file__).resolve().parents[1];book=json.loads((root/'kaggle/sentence_preserving_edit.ipynb').read_text())
        setup=next(''.join(c['source']) for c in book['cells'] if c['cell_type']=='code' and 'files = json.loads' in ''.join(c['source']))
        self.assertTrue(all(not c.get('outputs') for c in book['cells']))
        with tempfile.TemporaryDirectory() as tmp:
            runtime=Path(tmp)/'runtime';ns={};exec(setup.replace('/kaggle/working/_sentence_edit_runtime',str(runtime)),ns)
            for name,source in ns['files'].items():self.assertEqual(source,(root/name).read_text())
            r=subprocess.run([sys.executable,'-m','fama_style.rhetoric.sentence_edit','--dry-run','--output-dir',str(Path(tmp)/'dry')],cwd=tmp,
                env={**os.environ,'PYTHONPATH':str(runtime/'src'),'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,timeout=20)
            self.assertEqual(r.returncode,0,r.stderr);self.assertIn('4 jobs; no model calls',r.stdout)


if __name__=='__main__':unittest.main()
