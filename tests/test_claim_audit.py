"""Synthetic regression and negative controls for bounded claim-risk detectors."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.research.claim_audit import claim_checks,numeric_atoms,audit_saved
from fama_style.rhetoric import claim_constrained as exp


def kinds(result):return {e['kind'] for e in result['events']}


class ClaimAuditTests(unittest.TestCase):
    def test_scoped_bound_and_internal_inconsistency(self):
        source='The approximation is useful when dependence does not exceed the threshold.'
        correct='The approximation is useful when dependence does not surpass the threshold.'
        reverse='The approximation is useful when dependence does not fall below the threshold.'
        self.assertNotIn('scoped_bound_reversal_candidate',kinds(claim_checks(source,correct)))
        self.assertIn('scoped_bound_reversal_candidate',kinds(claim_checks(source,reverse)))
        # A correct statement elsewhere does not cancel the conflicting one.
        self.assertIn('scoped_bound_reversal_candidate',kinds(claim_checks(source,reverse+' '+correct)))
        unrelated='Trading does not fall below the threshold.'
        self.assertNotIn('scoped_bound_reversal_candidate',kinds(claim_checks(source,unrelated)))

    def test_subject_property_scope_not_global_keyword_presence(self):
        source='Most firms are small. Their market share is small.'
        omitted='Their market share is small.'
        r=claim_checks(source,omitted)
        self.assertIn('quantified_property_unresolved',kinds(r))
        self.assertNotIn('quantified_property_unresolved',kinds(claim_checks(source,source)))
        changed=claim_checks('Most firms are small.','Some firms are small.')
        self.assertEqual(changed['obligations'][0]['status'],'different_quantifier_candidate')
        # Legitimate paraphrases can remain unresolved, rather than be rejected.
        paraphrase=claim_checks('Most firms are small.','A preponderance of firms have a small size.')
        self.assertFalse(paraphrase['semantic_proof']);self.assertNotEqual(paraphrase['status'],'rejected')

    def test_annual_scope_is_not_discharged_by_unrelated_sentence(self):
        source='New firms enter each year.'
        self.assertNotIn('annual_scope_unresolved',kinds(claim_checks(source,'New firms enter annually.')))
        self.assertIn('annual_scope_unresolved',kinds(claim_checks(source,'New firms enter. Banks report losses annually.')))

    def test_number_formatting_year_expansion_and_changed_values(self):
        source='Data cover 1963-98. Ten firms hold a 3.70% share.'
        same='Data cover 1963 to 1998. 10 firms hold a 3.7 percent share.'
        self.assertEqual(numeric_atoms(source),numeric_atoms(same))
        self.assertNotIn('source_quantity_missing_candidate',kinds(claim_checks(source,same)))
        changed=claim_checks(source,same.replace('3.7','8.2'))
        self.assertIn('source_quantity_missing_candidate',kinds(changed));self.assertIn('new_quantity_candidate',kinds(changed))
        # Equal atoms cannot certify roles or quantities bound to the right actor.
        role_swap=claim_checks('Firm A owns 3 units; firm B owns 5 units.','Firm A owns 5 units; firm B owns 3 units.')
        self.assertEqual(role_swap['status'],'unresolved');self.assertFalse(role_swap['semantic_proof'])

    def test_concept_novelty_is_warning_not_proof(self):
        source='The approximation does not describe observations exactly.'
        r=claim_checks(source,'The approximation limits predictive power.')
        self.assertIn('new_concept_candidate',kinds(r));self.assertFalse(r['semantic_proof'])
        self.assertNotIn('new_concept_candidate',kinds(claim_checks('Predictive power is limited.','Predictive accuracy is limited.')))

    def test_modality_strengthening_is_scoped_and_conservative(self):
        self.assertIn('modality_strengthening_candidate',kinds(claim_checks('Firms may reduce output.','Firms must reduce output.')))
        self.assertNotIn('modality_strengthening_candidate',kinds(claim_checks('Firms may reduce output.','Firms could reduce output.')))
        self.assertNotIn('modality_strengthening_candidate',kinds(claim_checks('Firms may reduce output.','Banks must disclose assets.')))

    def test_nli_pass_is_not_semantic_certification(self):
        nli={'status':'proxy_consistent','checks':[{'direction':'source_unit_1','probabilities':{'entailment':.99,'neutral':.01,'contradiction':0.}}]}
        r=claim_checks('Most firms are small.','Firms own a small share.',nli)
        self.assertEqual(r['saved_nli_status'],'proxy_consistent');self.assertEqual(r['status'],'risk_flagged')
        self.assertEqual(r['source_units'][0]['saved_nli_status'],'proxy_consistent')
        identity=claim_checks('Firms report sales.','Firms report sales.')
        self.assertEqual(identity['status'],'surface_replay_only');self.assertFalse(identity['semantic_proof'])
        unknown=claim_checks('Firms report sales.','Corporations disclose revenues.',nli)
        self.assertEqual(unknown['status'],'unresolved')

    def fixtures(self,root):
        pack=exp.synthetic_pack();pack['rows'][0]['source']='Most firms are small.'
        packpath=root/'pack.json';packpath.write_text(json.dumps(pack));result=root/'results';result.mkdir()
        rows={r['id']:r for r in pack['rows']};jobs=exp.jobs_for(pack);outputs=[]
        for j in jobs:
            ref=rows[j['id']]['reference'];text=json.dumps({'sentences':exp.sentences(ref)}) if j['condition'].endswith('_slots') else ref
            outputs.append({'id':j['id'],'condition':j['condition'],'prompt_sha256':j['prompt_sha256'],'output':text,'synthetic':False,'hit_output_limit':False})
        records,_=exp.evaluate(pack,jobs,outputs)
        for name,data in [('metrics.private.jsonl',records),('generations.private.jsonl',outputs)]:
            (result/name).write_text(''.join(json.dumps(r)+'\n' for r in data))
        (result/'run_status.json').write_text(json.dumps({'complete':True,'training_called':False}))
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        (result/'manifest.json').write_text(json.dumps({'input_pack_sha256':sha(packpath)}))
        verification=root/'verification.json';verification.write_text(json.dumps({'checks':{'previous_audit':True},'artifact_hashes':{p.name:sha(p) for p in result.iterdir()}}))
        return packpath,result,verification

    def test_saved_audit_reproducibility_and_no_overwrites(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);pack,results,verification=self.fixtures(root)
            a=audit_saved(pack,results,verification,root/'a');b=audit_saved(pack,results,verification,root/'b')
            self.assertEqual(a,b);self.assertEqual((root/'a/claims.private.jsonl').read_bytes(),(root/'b/claims.private.jsonl').read_bytes())
            self.assertEqual(a['model_calls'],0);self.assertFalse(a['all_claims_verified'])
            with self.assertRaises(ValueError):audit_saved(pack,results,verification,root/'a')
            (results/'metrics.private.jsonl').write_text('tampered')
            with self.assertRaises(ValueError):audit_saved(pack,results,verification,root/'c')


if __name__=='__main__':unittest.main()
