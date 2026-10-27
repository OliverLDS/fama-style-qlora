"""Grouping realization contracts and evidence retention, without model weights."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.rhetoric import bound_grouping as e


class BoundGroupingTests(unittest.TestCase):
    def outputs(self,pack,synthetic=True):
        rows={r['id']:r for r in pack['rows']}
        return [{**{k:j[k] for k in ('id','condition','seed','prompt_sha256')},'output':rows[j['id']]['source'],
                 'hit_output_limit':False,'synthetic':synthetic} for j in e.jobs_for(pack)]

    def test_matched_jobs_and_reference_obligation_exclusion(self):
        pack=e.synthetic_pack();jobs=e.jobs_for(pack)
        self.assertEqual(len(jobs),27)
        self.assertEqual(len({(j['id'],j['condition'],j['seed']) for j in jobs}),27)
        altered=copy.deepcopy(pack)
        for r in altered['rows']:
            r['reference']='REFERENCE_SENTINEL';r['semantic_obligations_evaluation_only']=['OBLIGATION_SENTINEL']
        self.assertEqual(jobs,e.jobs_for(altered))
        for j in jobs:
            self.assertEqual(j['oracle'],j['condition']=='C_oracle_groups')
            row=next(r for r in pack['rows'] if r['id']==j['id'])
            if j['condition']=='A_concrete_only':self.assertEqual(j['prompt'],e.CONCRETE_STYLE+e.COMMON_INSTRUCTION+'\n\nPassage:\n'+row['source'])
        self.assertEqual(e.DECODING,e.previous.DECODING)

    def test_B_C_share_everything_except_group_assignments(self):
        jobs=e.jobs_for(e.synthetic_pack())
        b=next(j for j in jobs if j['condition']=='B_identity_groups')
        c=next(j for j in jobs if j['condition']=='C_oracle_groups')
        bp,bj=b['prompt'].rsplit('\n',1);cp,cj=c['prompt'].rsplit('\n',1)
        self.assertEqual(bp,cp)
        left,right=json.loads(bj),json.loads(cj)
        self.assertEqual(left.pop('source_units'),right.pop('source_units'))
        self.assertEqual(set(left),{'sentence_groups'});self.assertEqual(set(right),{'sentence_groups'})
        self.assertNotEqual(left,right)

    def test_grouping_cannot_omit_duplicate_or_reorder_source(self):
        for groups in ([[1],[3]],[[1,1],[2,3]],[[2,1],[3]],[[True,2],[3]]):
            pack=e.synthetic_pack();pack['rows'][0]['groups']=groups
            with self.assertRaises(ValueError):e.jobs_for(pack)
        pack=e.synthetic_pack();pack['rows'][0]['split']='test'
        with self.assertRaises(ValueError):e.jobs_for(pack)
        pack=e.synthetic_pack();pack['rows'][0]['source_units'][0]['text']='Altered text.'
        with self.assertRaises(ValueError):e.jobs_for(pack)

    def test_count_match_triggers_group_evidence_but_is_never_semantic_proof(self):
        row=e.synthetic_pack()['rows'][0];out='Firms invest and banks lend. Investors save.'
        verdict={'status':'risk_signal','nested_evidence':{'binding':'synthetic'}}
        with patch.object(e.verifier,'verify',return_value=verdict) as verify:
            result=e.group_diagnostics(row,'C_oracle_groups',out,lambda a,b:None)
            self.assertTrue(result['count_adherence']);self.assertFalse(result['group_binding_verified'])
            self.assertEqual(verify.call_args_list[0].args[:2],('Firms invest. Banks lend.','Firms invest and banks lend.'))
            self.assertEqual(len(result['positional_group_checks']),2)
            self.assertEqual(result['positional_group_checks'][0]['verifier'],verdict)
        result=e.group_diagnostics(row,'C_oracle_groups',row['source'],lambda a,b:1/0)
        self.assertFalse(result['count_adherence']);self.assertEqual(result['positional_group_checks'],[])

    def test_parenthesis_boundaries_are_preserved_as_uncertainty(self):
        text='Firms invest. (Banks lend.) Investors save.'
        result=e.output_units(text)
        self.assertEqual(len(result['units']),3);self.assertTrue(result['ambiguous'])
        row=e.synthetic_pack()['rows'][0]
        result=e.group_diagnostics(row,'C_oracle_groups',text,lambda a,b:1/0)
        self.assertIsNone(result['count_adherence']);self.assertEqual(result['positional_group_checks'],[])
        self.assertTrue(e.output_units('The U.S. Investor saves.')['ambiguous'])

    def test_all_candidates_and_full_verdicts_are_retained_and_replayable(self):
        pack=e.synthetic_pack();outs=self.outputs(pack,False);rows={r['id']:r for r in pack['rows']}
        key=lambda r:(r['id'],r['condition'],r['seed']);verdict={'status':'risk_signal','nested':{'why':['synthetic']}}
        semantic={key(o):verdict for o in outs}
        with patch.object(e.verifier,'verify',return_value=verdict):
            grouping={key(o):e.group_diagnostics(rows[o['id']],o['condition'],o['output'],lambda a,b:None) for o in outs}
        records,summary=e.evaluate(pack,outs,semantic,group_checks=grouping)
        self.assertEqual(len(records),27);self.assertTrue(all(r['verifier']==verdict for r in records))
        self.assertEqual(summary['conditions']['B_identity_groups']['count_adherence'],9)
        self.assertEqual(summary['conditions']['C_oracle_groups']['count_adherence'],0)
        self.assertIsNone(summary['semantic_accuracy'])
        self.assertEqual(e.evaluate(pack,outs,{key(r):r['verifier'] for r in records},group_checks={key(r):r['group_diagnostics'] for r in records}),(records,summary))
        with self.assertRaises(ValueError):e.evaluate(pack,outs,semantic)
        bad=copy.deepcopy(grouping);bad[key(outs[0])]['output_sentence_count']=999
        with self.assertRaises(ValueError):e.evaluate(pack,outs,semantic,group_checks=bad)

    def test_missing_swapped_empty_and_cutoff_outputs_never_complete(self):
        pack=e.synthetic_pack();outs=self.outputs(pack)
        for mode in ('missing','duplicate','empty','cutoff'):
            changed=copy.deepcopy(outs)
            if mode=='missing':changed.pop()
            if mode=='duplicate':changed[-1]=changed[0]
            if mode=='empty':changed[0]['output']=' '
            if mode=='cutoff':changed[0]['hit_output_limit']=True
            with self.assertRaises(ValueError):e.evaluate(pack,changed,synthetic=True)

    def test_dry_run_and_token_audit_do_not_enter_runtime(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(e,'run_training',side_effect=AssertionError('runtime')),patch.object(e,'secret_token',side_effect=AssertionError('secret')):
            e.main(['--dry-run','--output-dir',str(Path(folder)/'dry')])
        job=e.jobs_for(e.synthetic_pack())[0]
        with self.assertRaises(ValueError):e.audit_tokens(job,[1]*4096)

    def test_notebook_embedded_runtime_runs_without_checkout_or_model(self):
        path=e.ROOT/'kaggle/bound_grouping.ipynb'
        if not path.exists():self.skipTest('Notebook not built yet')
        book=json.loads(path.read_text());self.assertTrue(all(not c.get('outputs') and c.get('execution_count') is None for c in book['cells']))
        for c in book['cells']:
            if c['cell_type']=='code' and not ''.join(c['source']).startswith('%'):compile(''.join(c['source']),'notebook','exec')
        with tempfile.TemporaryDirectory() as folder:
            setup=''.join(book['cells'][3]['source']).replace('/kaggle/working/_bound_grouping_runtime',folder+'/runtime')
            commands=setup+'\nfrom fama_style.rhetoric import bound_grouping as experiment\nexperiment.main(["--dry-run", "--output-dir", '+repr(folder+'/dry')+'])\n'
            packpath=e.ROOT/'research_runs/bound_grouping_realization_v1/input_pack/bound_grouping_pack.json'
            if packpath.exists():
                commands+=''.join(book['cells'][4]['source']).replace('list(Path("/kaggle/input").rglob("bound_grouping_pack.json"))',repr([str(packpath)])).replace('/kaggle/working/bound_grouping_preflight',folder+'/preflight')
            result=subprocess.run([sys.executable,'-c',commands],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertTrue((Path(folder)/'dry/synthetic_evaluation.json').exists())


if __name__=='__main__':unittest.main()
