"""Candidate selection, semantic binding, source-only rendering and pilot gates."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.rhetoric import operation_expansion as e

class OperationExpansionTests(unittest.TestCase):
    def fixture(self):
        s='Demand may rise, so output may increase. The approximation may apply if costs are low.'
        t='Demand may rise. Thus output may increase. The approximation may apply if costs are low.'
        row={'plain_english':s,'completion':t};r={'case_id':'Xsynthetic','pair_sha256':e.pair_sha(row),'source':s,'reference':t,
            'source_units':[{'unit_id':'S1','text':'Demand may rise, so output may increase.'},{'unit_id':'S2','text':'The approximation may apply if costs are low.'}],
            'target_units':[{'unit_id':'T1','text':'Demand may rise.'},{'unit_id':'T2','text':'Thus output may increase.'},{'unit_id':'T3','text':'The approximation may apply if costs are low.'}]}
        a={'case_id':r['case_id'],'pair_sha256':r['pair_sha256'],'status':'supported','family':'premise_inference_scope','variant':'seed_exact',
            'claims':[{'claim_id':'c'+str(i),'epistemic_status':'hypothesis','composite':False,'source_evidence':[{'quote':sq,'occurrence':1}],'target_evidence':[{'quote':tq,'occurrence':1}],'protected':[]} for i,(sq,tq) in enumerate(zip(['Demand may rise','so output may increase','The approximation may apply if costs are low.'],['Demand may rise.','Thus output may increase.','The approximation may apply if costs are low.']),1)],
            'source_units':[{'unit_id':'S1','claim_refs':['c1','c2']},{'unit_id':'S2','claim_refs':['c3']}],
            'target_units':[{'unit_id':'T'+str(i),'claim_refs':['c'+str(i)]} for i in range(1,4)],
            'relations':[{'relation_id':'r1','kind':'inference','from_claim_refs':['c1'],'to_claim_refs':['c2'],'source_evidence':[{'quote':'so','occurrence':1}],'certainty':'explicit'}],
            'program':{'stages':[{'role':role,'claim_refs':['c'+str(i)],'target_evidence':[{'quote':q,'occurrence':1}]} for i,(role,q) in enumerate(zip(['premise','inference','scope'],['Demand may rise.','Thus output may increase.','The approximation may apply if costs are low.']),1)],'preserve_relation_refs':['r1'],'language_controls':['retain_modal_strength']},'notes':''}
        return r,a

    def test_evidence_compiler_and_generator_reference_boundary(self):
        r,a=self.fixture();c=e.validate_annotation(r,a);self.assertTrue(c['packing_or_order_contrast']);self.assertFalse(c['semantic_equivalence_verified'])
        p=e.render_pair(r,c);self.assertNotIn('Thus',p['operation_shared_evidence']);self.assertNotIn(r['reference'],p['operation_shared_evidence'])
        self.assertIn(r['source'],p['operation_shared_evidence'])
        before=p['direct_shared_evidence'].split('\nReturn only the rewritten paragraph.')[0]
        self.assertTrue(p['operation_shared_evidence'].startswith(before))

    def test_missing_claims_changed_anchors_and_reversed_relations_rejected(self):
        for mode in ['hash','source_span','unknown_claim','unit_missing','relation_missing','reverse','target_span','stage_order','extra_field']:
            r,a=self.fixture()
            if mode=='hash':a['pair_sha256']='0'*64
            if mode=='source_span':a['claims'][0]['source_evidence'][0]['quote']='not in source'
            if mode=='unknown_claim':a['program']['stages'][0]['claim_refs']=['c99']
            if mode=='unit_missing':a['source_units'].pop()
            if mode=='relation_missing':a['program']['preserve_relation_refs']=['r99']
            if mode=='reverse':a['relations'][0]['from_claim_refs']=['c2'];a['relations'][0]['to_claim_refs']=['c1']
            if mode=='target_span':a['program']['stages'][0]['target_evidence'][0]['quote']='not in target'
            if mode=='stage_order':a['program']['stages']=[a['program']['stages'][2]]+a['program']['stages'][:2]
            if mode=='extra_field':a['replacement_reference']='forbidden'
            with self.assertRaises(ValueError,msg=mode):e.validate_annotation(r,a)

    def test_uncertain_cases_are_records_not_forced_plans(self):
        r,a=self.fixture();a.update(status='uncertain',family='none',variant='none',claims=[],source_units=[],target_units=[],relations=[],program=None,notes='Claim alignment unresolved.')
        c=e.validate_annotation(r,a);self.assertFalse(c['packing_or_order_contrast'])
        a['notes']=''
        with self.assertRaises(ValueError):e.validate_annotation(r,a)

    def test_count_and_diversity_gate_never_authorizes_training(self):
        rows=[{'family':list(e.FAMILIES)[i%3],'paper_key':'paper-'+str(i%4)} for i in range(24)]
        gate=e.pilot_gates(rows,[{}]);self.assertTrue(gate['training_annotation_gate_passed']);self.assertFalse(gate['ready_to_train'])
        self.assertEqual(gate['related_variants_needing_realization_check'],1)
        self.assertFalse(e.pilot_gates(rows[:12],[])['training_annotation_gate_passed'])
        for row in rows:row['paper_key']='same'
        self.assertFalse(e.pilot_gates(rows,[])['training_annotation_gate_passed'])

    def test_audit_keeps_related_variant_out_of_primary_and_preserves_raw(self):
        r,a=self.fixture();a['variant']='related_variant'
        key={'records':[{'case_id':r['case_id'],'source_id':'synthetic','paper_key':'synthetic','selection_role':'cue_selected_pending_annotation','record':r}],
            'response_files':{r['case_id']:'annotations_01.jsonl'}}
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);package=folder/'package';package.mkdir();(package/'manifest.json').write_text('{}')
            responses=folder/'responses';responses.mkdir();f=responses/'annotations_01.jsonl';f.write_text(json.dumps(a)+'\n')
            with patch.object(e,'check_package',return_value=key):e.audit(package,responses,folder/'audit')
            self.assertEqual(e.read_lines(folder/'audit/primary_pairs.provisional.private.jsonl'),[])
            self.assertEqual(len(e.read_lines(folder/'audit/related_variants.private.jsonl')),1)
            gate=json.loads((folder/'audit/pilot_gate.json').read_text());self.assertFalse(gate['ready_to_train'])
            self.assertEqual(e.file_sha(f),e.file_sha(folder/'audit/raw/annotations_01.jsonl'))

    def test_nontraining_selection_rejected(self):
        with self.assertRaises(ValueError):e.candidate_pool([{'split':'test'}])

    def test_prepared_candidates_are_diverse_exact_and_unlabelled(self):
        path=e.ROOT/'research_runs/operation_expansion_v1/package_v1'
        if not path.exists():self.skipTest('Private prepared artifacts absent')
        key=e.check_package(path);self.assertEqual(len(key['records']),30)
        self.assertEqual(len(key['selection']['paper_counts']),7)
        self.assertTrue(all(n<=6 for n in key['selection']['paper_counts'].values()))
        self.assertEqual(sum(r['selection_role']=='cue_selected_pending_annotation' for r in key['records']),24)
        for p in (path/'batches').glob('*.jsonl'):
            for r in e.read_lines(p):self.assertNotIn('candidate_family_hint',r);self.assertNotIn('paper_key',r)
        train=e.read_lines(e.ROOT/'data/fama_corpus_v1/train.jsonl');a,b=e.select_candidates(train)
        self.assertEqual(b,key['selection'])

if __name__=='__main__':unittest.main()
