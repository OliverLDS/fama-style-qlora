"""Groupwise calls, budgets, assembly and evidence contracts, without ML weights."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.rhetoric import groupwise_realization_v2 as e


class GroupwiseV2Tests(unittest.TestCase):
    def outputs(self,pack,synthetic=True):
        rows={r['id']:r for r in pack['rows']};outs=[]
        for j in e.jobs_for(pack):
            row=rows[j['id']];source={u['unit_id']:u['text'] for u in row['source_units']}
            text=row['source'] if j['condition']=='A_whole_paragraph' else ' '.join(source[i] for i in row['groups'][j['group_index']-1])
            outs.append({**{k:j[k] for k in ('id','condition','seed','group_index','sampling_seed','max_new_tokens','prompt_sha256')},
                'output':text,'synthetic':synthetic,'hit_output_limit':False})
        return outs

    def test_reference_exclusion_and_whole_control_exact_prompt(self):
        pack=e.synthetic_pack();jobs=e.jobs_for(pack)
        old={(j['id'],j['seed']):j for j in e.prior.jobs_for(pack) if j['condition']=='C_oracle_groups'}
        for j in jobs:
            if j['condition']=='A_whole_paragraph':
                self.assertEqual(j['prompt'],old[(j['id'],j['seed'])]['prompt'])
                self.assertEqual(j['sampling_seed'],j['seed'])
            else:
                row=next(r for r in pack['rows'] if r['id']==j['id'])
                units={u['unit_id']:u['text'] for u in row['source_units']}
                expected=' '.join(units[i] for i in row['groups'][j['group_index']-1])
                self.assertEqual(j['prompt'].split('Assigned group:\n',1)[1],expected)
                self.assertNotIn(row['source'],j['prompt'])
                self.assertNotIn('Return only one coherent paragraph',j['prompt'])
                self.assertNotIn('sentence_groups',j['prompt'])
        changed=copy.deepcopy(pack)
        for r in changed['rows']:r['reference']='REFERENCE_SENTINEL';r['semantic_obligations_evaluation_only']=['OBLIGATION_SENTINEL']
        self.assertEqual(jobs,e.jobs_for(changed))
        self.assertEqual(e.DECODING,e.prior.DECODING)

    def test_output_allowance_is_exactly_matched_and_group_seeds_are_stable(self):
        pack=e.synthetic_pack();jobs=e.jobs_for(pack)
        for row in pack['rows']:
            self.assertEqual(sum(e.budgets(row)),768)
            for seed in e.SEEDS:
                a=[j for j in jobs if j['id']==row['id'] and j['seed']==seed and j['condition']=='A_whole_paragraph']
                b=[j for j in jobs if j['id']==row['id'] and j['seed']==seed and j['condition']=='B_groupwise']
                self.assertEqual(sum(j['max_new_tokens'] for j in a),sum(j['max_new_tokens'] for j in b))
                self.assertEqual(len({j['sampling_seed'] for j in b}),len(b))
        self.assertEqual(jobs,e.jobs_for(pack))
        with self.assertRaises(ValueError):e.budgets(pack['rows'][0],1)

    def test_assembly_uses_group_order_even_if_calls_are_reordered(self):
        pack=e.synthetic_pack();outs=self.outputs(pack);a=e.assemble(pack,outs,True)
        self.assertEqual(a,e.assemble(pack,list(reversed(outs)),True))
        self.assertEqual(len(a),18)
        self.assertTrue(all(c['available'] for c in a))
        self.assertTrue(all(c['max_output_token_allowance']==768 for c in a))
        self.assertTrue(all(c['paragraph']==pack['rows'][0]['source'] for c in a))
        # Group 1 returned two sentences: it remains unchanged and is not a success.
        for c in a:
            if c['condition']=='B_groupwise':
                self.assertFalse(c['assembly_format_valid']);self.assertEqual(len(c['group_outputs']),2)

    def test_unpunctuated_or_multisentence_piece_is_kept_without_fallback(self):
        pack=e.synthetic_pack();outs=self.outputs(pack)
        out=next(o for o in outs if o['condition']=='B_groupwise' and o['group_index']==1)
        out['output']='Changed text without punctuation'
        candidates=e.assemble(pack,outs,True)
        c=next(c for c in candidates if e.candidate_key(c)==e.candidate_key(out))
        self.assertTrue(c['paragraph'].startswith(out['output']+' '));self.assertFalse(c['assembly_format_valid'])
        self.assertFalse(e.piece_format('Firms invest. (Banks lend.)')['single_sentence'])

    def test_missing_duplicated_or_changed_budget_calls_fail(self):
        pack=e.synthetic_pack();outs=self.outputs(pack)
        for mode in ('missing','duplicate','budget','seed'):
            changed=copy.deepcopy(outs)
            if mode=='missing':changed.pop()
            if mode=='duplicate':changed[-1]=changed[0]
            if mode=='empty':changed[0]['output']=''
            if mode=='cutoff':changed[0]['hit_output_limit']=True
            if mode=='budget':changed[0]['max_new_tokens']+=1
            if mode=='seed':changed[0]['sampling_seed']+=1
            with self.assertRaises(ValueError,msg=mode):e.assemble(pack,changed,True)

    def test_group_evidence_survives_piece_format_failure_and_replays(self):
        pack=e.synthetic_pack();outs=self.outputs(pack,False);candidates=e.assemble(pack,outs);rows={r['id']:r for r in pack['rows']}
        verdict={'status':'risk_signal','nested':{'evidence':['synthetic']}}
        semantic={e.candidate_key(c):copy.deepcopy(verdict) for c in candidates}
        with patch.object(e.verifier,'verify',return_value=verdict):
            binding={e.candidate_key(c):e.binding_diagnostics(rows[c['id']],c,lambda a,b:None) for c in candidates}
        records,summary=e.evaluate(pack,outs,semantic,binding)
        self.assertEqual(len(records),18);self.assertEqual(summary['conditions']['B_groupwise']['valid_assemblies_all_attempts'],0)
        b=next(r for r in records if r['condition']=='B_groupwise')
        self.assertEqual(len(b['group_diagnostics']['assigned_group_checks']),2)
        self.assertEqual(b['group_diagnostics']['assigned_group_checks'][0]['verifier'],verdict)
        self.assertEqual(json.loads(json.dumps(records))[0]['verifier'],verdict)
        self.assertEqual(e.evaluate(pack,outs,{e.candidate_key(r):r['verifier'] for r in records},{e.candidate_key(r):r['group_diagnostics'] for r in records}),(records,summary))
        self.assertIsNone(summary['semantic_accuracy'])
        altered=copy.deepcopy(binding);key=next(k for k in altered if k[1]=='B_groupwise')
        altered[key]['assigned_group_checks'][0]['output_piece_sha256']='incorrect'
        with self.assertRaises(ValueError):e.evaluate(pack,outs,semantic,altered)
        with self.assertRaises(ValueError):e.evaluate(pack,outs,semantic)

    def test_cutoff_or_empty_piece_stays_unavailable_without_aborting_other_candidates(self):
        for mode in ('cutoff','empty'):
            pack=e.synthetic_pack();outs=self.outputs(pack)
            out=next(o for o in outs if o['condition']=='B_groupwise')
            if mode=='cutoff':out['hit_output_limit']=True;out['output']='Partial text that'
            else:out['output']=''
            candidates=e.assemble(pack,outs,True)
            failed=next(c for c in candidates if e.candidate_key(c)==e.candidate_key(out))
            self.assertFalse(failed['available']);self.assertIsNone(failed['paragraph']);self.assertIsNone(failed['output_sha256'])
            self.assertEqual(len(candidates),18);self.assertEqual(sum(c['available'] for c in candidates),17)
            self.assertIn(out['output'],failed['group_outputs'])
            records,summary=e.evaluate(pack,outs,synthetic=True)
            b=summary['conditions']['B_groupwise']
            self.assertEqual(b['paragraph_attempts'],9);self.assertEqual(b['available_paragraphs'],8);self.assertEqual(b['unavailable_paragraphs'],1)
            self.assertEqual(b['failed_generation_calls'],1);self.assertFalse(summary['all_candidates_complete'])
            self.assertEqual(len(records),18)
            missing=next(r for r in records if not r['available'])
            self.assertIsNone(missing['verifier']);self.assertIsNone(missing['style_alignment_benefit_proxy'])
            # A real-mode execution can finish with a failed generation, but never
            # reports the affected paragraph as complete or substitutes source text.
            real=copy.deepcopy(outs)
            for item in real:item['synthetic']=False
            candidates=e.assemble(pack,real);row_by_id={r['id']:r for r in pack['rows']}
            verdict={'status':'risk_signal','nested':{'evidence':['synthetic test']}}
            available=[c for c in candidates if c['available']]
            with patch.object(e.verifier,'verify',return_value=verdict):
                evidence={e.candidate_key(c):e.binding_diagnostics(row_by_id[c['id']],c,lambda a,b:None) for c in available}
            _,real_summary=e.evaluate(pack,real,{e.candidate_key(c):verdict for c in available},evidence)
            self.assertEqual(real_summary['status'],'execution_complete_with_generation_failures')
            self.assertFalse(real_summary['all_candidates_complete'])

    def test_v1_control_seeds_and_budgets_are_unchanged(self):
        pack=e.synthetic_pack();old={e.job_key(j):j for j in e.failed_v1.jobs_for(pack)}
        for j in e.jobs_for(pack):
            original=old[e.job_key(j)]
            self.assertEqual(j['sampling_seed'],original['sampling_seed'])
            self.assertEqual(j['max_new_tokens'],original['max_new_tokens'])
            if j['condition']=='A_whole_paragraph':self.assertEqual(j['prompt'],original['prompt'])

    def test_context_and_dry_run_do_not_access_secret_or_runtime(self):
        pack=e.synthetic_pack();job=e.jobs_for(pack)[0]
        with self.assertRaises(ValueError):e.audit_tokens(job,[1]*4096)
        with tempfile.TemporaryDirectory() as folder,patch.object(e,'run_training',side_effect=AssertionError('GPU')),patch.object(e,'secret_token',side_effect=AssertionError('secret')):
            e.main(['--dry-run','--output-dir',str(Path(folder)/'dry')])
            result=json.loads((Path(folder)/'dry/synthetic_evaluation.json').read_text())
            self.assertEqual(result['status'],'synthetic_wiring_check')

    def test_embedded_notebook_runtime_without_checkout(self):
        path=e.ROOT/'kaggle/groupwise_realization_v2.ipynb'
        if not path.exists():self.skipTest('Notebook not built yet')
        book=json.loads(path.read_text());self.assertTrue(all(not c.get('outputs') and c.get('execution_count') is None for c in book['cells']))
        for c in book['cells']:
            if c['cell_type']=='code' and not ''.join(c['source']).startswith('%'):compile(''.join(c['source']),'notebook','exec')
        with tempfile.TemporaryDirectory() as folder:
            setup=''.join(book['cells'][3]['source']).replace('/kaggle/working/_groupwise_v2_runtime',folder+'/runtime')
            code=setup+'\nfrom fama_style.rhetoric import groupwise_realization_v2 as experiment\nexperiment.main(["--dry-run", "--output-dir", '+repr(folder+'/dry')+'])\n'
            packpath=e.ROOT/'research_runs/groupwise_realization_v2/input_pack/groupwise_pack.json'
            if packpath.exists():
                code+=''.join(book['cells'][4]['source']).replace('list(Path("/kaggle/input").rglob("groupwise_pack.json"))',repr([str(packpath)])).replace('/kaggle/working/groupwise_v2_preflight',folder+'/preflight')
            # Stream the unchanged program; do not put the runtime in argv.
            run=subprocess.run([sys.executable,'-'],input=code,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertTrue((Path(folder)/'dry/synthetic_evaluation.json').exists())


if __name__=='__main__':unittest.main()
