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
from fama_style.rhetoric import minimal_antecedent as e

class MinimalAntecedentTests(unittest.TestCase):
    def setUp(self):
        patcher=patch.dict(e.EDIT_SPEC,{('synthetic-'+str(i),1):[('Firms','Local firms',['Firms'])] for i in range(3)})
        patcher.start();self.addCleanup(patcher.stop)

    def pack(self):
        p=e.parent.synthetic_pack()
        p['rows']=[{**r,'selected_group_indices':[1]} for r in p['rows']]
        return p

    def outputs(self,p):
        return [{**{k:j[k] for k in ('id','condition','seed','group_index','sampling_seed','max_new_tokens','prompt_sha256')},'output':'Synthetic response.','synthetic':True,'hit_output_limit':False} for j in e.jobs_for(p)]

    def test_matched_intervention_reference_exclusion(self):
        p=self.pack();jobs=e.jobs_for(p)
        old={e.job_key(j):j for j in e.previous.jobs_for(p)}
        for offset in range(0,len(jobs),3):
            a,b,c=sorted(jobs[offset:offset+3],key=lambda j:j['condition'])
            self.assertEqual(a,old[e.job_key(a)]);self.assertEqual(b,old[e.job_key(b)])
            for k in ('sampling_seed','max_new_tokens','seed','group_index'):self.assertEqual(a[k],c[k])
            self.assertEqual(a['prompt'].split('Assigned group:\n')[0],c['prompt'].split('Assigned group:\n')[0])
            self.assertNotIn('<context>',c['prompt'])
            self.assertNotIn('Investors save.',c['prompt'])
            self.assertIn('Local firms',c['prompt'])
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
                'local_group_verifier':{'nested':['evidence']},'full_context_to_output_nli':None,'expanded_group_sha256':e.expansion(r,o['group_index'])['expanded_group_sha256'],'expanded_group_verifier':{'nested':['expanded evidence']}}
        records,summary=e.evaluate(p,out,evidence)
        self.assertEqual(records[0]['diagnostics'],evidence[e.job_key(out[0])]);self.assertIsNone(summary['semantic_accuracy'])
        evidence[e.job_key(out[0])]['context_sha256']='wrong'
        with self.assertRaises(ValueError):e.evaluate(p,out,evidence)
        with self.assertRaises(ValueError):e.evaluate(p,out)

    def test_expansion_provenance_and_rejected_unanchored_edits(self):
        row=self.pack()['rows'][0];record=e.expansion(row,1)
        self.assertEqual(record['expanded_group'],'Local firms invest. Banks lend.')
        for edit in record['edits']:
            for support in edit['source_support']:
                self.assertEqual(row['source'][support['start']:support['end']],support['quote'])
        with patch.dict(e.EDIT_SPEC,{(row['id'],1):[('Firms','Firms 92',['Firms'])]}):
            with self.assertRaises(ValueError):e.expansion(row,1)
        with patch.dict(e.EDIT_SPEC,{(row['id'],1):[('Firms','Local firms',['Missing actor'])]}):
            with self.assertRaises(ValueError):e.expansion(row,1)

    def test_over_context_rejected(self):
        with self.assertRaises(ValueError):e.audit_tokens(e.jobs_for(self.pack())[0],[1]*4096)

    def test_prepared_real_pack_and_notebook_cpu_only(self):
        path=e.ROOT/'research_runs/minimal_antecedent_v1/input_pack/antecedent_pack.json'
        if not path.exists():self.skipTest('Private pack absent in public checkout')
        pack=e.load_pack(path,e.file_sha(path));self.assertEqual(len(e.jobs_for(pack)),27)
        self.assertEqual({(j['id'],j['group_index']) for j in e.jobs_for(pack)},set(e.SELECTION))
        with tempfile.TemporaryDirectory() as folder,patch.object(e,'secret_token',side_effect=AssertionError('secret')),patch.object(e,'run_training',side_effect=AssertionError('GPU')):
            e.main(['--dry-run','--pack',str(path),'--expected-pack-sha',e.file_sha(path),'--output-dir',folder+'/dry'])
            self.assertEqual(json.loads(Path(folder+'/dry/synthetic_evaluation.json').read_text())['status'],'synthetic_wiring_check')
        book=json.loads((e.ROOT/'kaggle/minimal_antecedent.ipynb').read_text())
        for cell in book['cells']:
            self.assertFalse(cell.get('outputs'));self.assertIsNone(cell.get('execution_count'))
            if cell['cell_type']=='code' and not ''.join(cell['source']).startswith('%'):compile(''.join(cell['source']),'notebook','exec')
        with tempfile.TemporaryDirectory() as folder:
            setup=''.join(book['cells'][3]['source']).replace('/kaggle/working/_minimal_antecedent_runtime',folder+'/runtime')
            check=''.join(book['cells'][4]['source']).replace('list(Path("/kaggle/input").rglob("antecedent_pack.json"))',repr([str(path)])).replace('/kaggle/working/minimal_antecedent_preflight',folder+'/preflight')
            result=subprocess.run([sys.executable,'-I','-c',setup+'\n'+check],cwd=folder,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

if __name__=='__main__':unittest.main()
