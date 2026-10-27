"""Context intervention, evidence binding and isolated CPU notebook checks."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.rhetoric import group_context_restoration as e

class ContextTests(unittest.TestCase):
    def pack(self):
        p=e.parent.synthetic_pack()
        p['rows']=[{**r,'selected_group_indices':[1]} for r in p['rows']]
        return p

    def outputs(self,p):
        return [{**{k:j[k] for k in ('id','condition','seed','group_index','sampling_seed','max_new_tokens','prompt_sha256')},'output':'Synthetic response.','synthetic':True,'hit_output_limit':False} for j in e.jobs_for(p)]

    def test_matched_intervention_reference_exclusion(self):
        p=self.pack();jobs=e.jobs_for(p)
        old={(j['id'],j['seed'],j['group_index']):j for j in e.parent.jobs_for(p) if j['condition']=='B_groupwise'}
        for a,b in zip(jobs[::2],jobs[1::2]):
            a,b=sorted([a,b],key=lambda j:j['condition'])
            self.assertEqual(a['prompt'],old[(a['id'],a['seed'],a['group_index'])]['prompt'])
            self.assertTrue(b['prompt'].startswith(a['prompt']+e.CONTEXT_RULE))
            for k in ('sampling_seed','max_new_tokens','seed','group_index'):self.assertEqual(a[k],b[k])
            self.assertIn('not an additional rewrite assignment',b['prompt'])
            self.assertIn('Only rewrite the assigned group',b['prompt'])
        altered=copy.deepcopy(p)
        for r in altered['rows']:r['reference']='SECRET_REFERENCE';r['semantic_obligations_evaluation_only']=['PRIVATE_NOTE']
        self.assertEqual(jobs,e.jobs_for(altered))

    def test_failure_retention_and_identity_checks(self):
        p=self.pack();out=self.outputs(p);out[0]['hit_output_limit']=True
        records,summary=e.evaluate(p,out,synthetic=True)
        self.assertFalse(records[0]['available']);self.assertIsNone(records[0]['diagnostics'])
        self.assertEqual(records[0]['output'],'Synthetic response.')
        with self.assertRaises(ValueError):e.evaluate(p,out[:-1],synthetic=True)
        out[1]['sampling_seed']+=1
        with self.assertRaises(ValueError):e.evaluate(p,out,synthetic=True)

    def test_evidence_hash_binding_and_full_verdict_retention(self):
        p=self.pack();out=self.outputs(p);evidence={}
        for o in out:
            o['synthetic']=False;r=next(r for r in p['rows'] if r['id']==o['id'])
            group=' '.join(u['text'] for u in r['source_units'] if u['unit_id'] in r['groups'][o['group_index']-1])
            evidence[e.job_key(o)]={'output_sha256':e.sha(o['output']),'source_group_sha256':e.sha(group),'context_sha256':e.sha(r['source']),
                'local_group_verifier':{'nested':['evidence']},'full_context_to_output_nli':None}
        records,summary=e.evaluate(p,out,evidence)
        self.assertEqual(records[0]['diagnostics'],evidence[e.job_key(out[0])]);self.assertIsNone(summary['semantic_accuracy'])
        evidence[e.job_key(out[0])]['context_sha256']='wrong'
        with self.assertRaises(ValueError):e.evaluate(p,out,evidence)
        with self.assertRaises(ValueError):e.evaluate(p,out)

    def test_over_context_rejected(self):
        with self.assertRaises(ValueError):e.audit_tokens(e.jobs_for(self.pack())[0],[1]*4096)

    def test_prepared_real_pack_and_notebook_cpu_only(self):
        path=e.ROOT/'research_runs/group_context_restoration_v1/input_pack/context_pack.json'
        if not path.exists():self.skipTest('Private pack absent in public checkout')
        pack=e.load_pack(path,e.file_sha(path));self.assertEqual(len(e.jobs_for(pack)),18)
        self.assertEqual({(j['id'],j['group_index']) for j in e.jobs_for(pack)},set(e.SELECTION))
        with tempfile.TemporaryDirectory() as folder,patch.object(e,'secret_token',side_effect=AssertionError('secret')),patch.object(e,'run_training',side_effect=AssertionError('GPU')):
            e.main(['--dry-run','--pack',str(path),'--expected-pack-sha',e.file_sha(path),'--output-dir',folder+'/dry'])
            self.assertEqual(json.loads(Path(folder+'/dry/synthetic_evaluation.json').read_text())['status'],'synthetic_wiring_check')
        book=json.loads((e.ROOT/'kaggle/group_context_restoration.ipynb').read_text())
        for cell in book['cells']:
            self.assertFalse(cell.get('outputs'));self.assertIsNone(cell.get('execution_count'))
            if cell['cell_type']=='code' and not ''.join(cell['source']).startswith('%'):compile(''.join(cell['source']),'notebook','exec')
        with tempfile.TemporaryDirectory() as folder:
            setup=''.join(book['cells'][3]['source']).replace('/kaggle/working/_group_context_runtime',folder+'/runtime')
            check=''.join(book['cells'][4]['source']).replace('list(Path("/kaggle/input").rglob("context_pack.json"))',repr([str(path)])).replace('/kaggle/working/group_context_preflight',folder+'/preflight')
            result=subprocess.run([sys.executable,'-I','-c',setup+'\n'+check],cwd=folder,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

if __name__=='__main__':unittest.main()
