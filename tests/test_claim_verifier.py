"""Synthetic mechanics tests; no pretrained-model quality claims or downloads."""
import contextlib
import io
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.research import claim_verifier as v

ENTAIL={'entailment':.99,'neutral':.01,'contradiction':0.}
NEUTRAL={'entailment':.01,'neutral':.99,'contradiction':0.}


class ClaimVerifierTests(unittest.TestCase):
    def test_actor_and_quantity_roles_even_when_mock_nli_is_blind(self):
        for source,target,kind in (
            ('The broker pays the manager.','The manager pays the broker.','actor_role_conflict'),
            ('Alder owns 3 notes and Birch holds 8 notes.','Alder owns 8 notes and Birch holds 3 notes.','quantity_role_conflict')):
            verdict=v.verify(source,target,lambda a,b:ENTAIL)
            self.assertEqual(verdict['status'],'risk_signal')
            self.assertIn(kind,[c['kind'] for c in verdict['role_conflicts']])
        source='Alder owns 3 notes and Birch holds 8 notes.'
        target='Three notes are held by Alder; eight notes are held by Birch.'
        verdict=v.verify(source,target,lambda a,b:ENTAIL)
        self.assertFalse(verdict['role_conflicts']);self.assertEqual(verdict['status'],'model_consistent')
        self.assertFalse(verdict['semantic_proof'])

    def test_full_grammar_only_no_modal_or_pronoun_scope_guess(self):
        self.assertIsNone(v.role_fact('Alder may hold 3 notes.'))
        self.assertIsNone(v.role_fact('They own 3 notes.'))
        self.assertEqual(len(v.units('If demand falls and firms close, shops reduce inventory.')),1)
        self.assertEqual(len(v.units('Alder holds 3 notes and Birch holds 8 notes.')),2)
        self.assertEqual(len(v.units('Costs and profits rise.')),1)
        verdict=v.verify('Alder holds 3 notes.','Birch holds 3 notes.',lambda a,b:None)
        self.assertEqual(verdict['status'],'abstain')

    def test_both_coverage_and_output_support_are_checked(self):
        def predictor(premise,hypothesis):
            return NEUTRAL if 'Banks borrow' in hypothesis and 'Banks borrow' not in premise else ENTAIL
        result=v.verify('Firms invest.','Firms invest. Banks borrow.',predictor)
        self.assertEqual(result['status'],'risk_signal')
        self.assertEqual({a['direction'] for a in result['alignments']},{'source_coverage','output_support'})
        self.assertTrue(any(a['direction']=='output_support' and a['status']=='flagged_proxy' for a in result['alignments']))

    def test_abstention_budget_probability_validation_and_cache(self):
        calls=[]
        def predictor(a,b):calls.append((a,b));return ENTAIL
        result=v.verify('Firms invest.','Firms invest.',predictor)
        self.assertEqual(len(calls),1);self.assertEqual(result['status'],'model_consistent')
        self.assertEqual(v.verify('Firms invest.','Firms invest.',lambda a,b:None)['status'],'abstain')
        text=' '.join('Firms invest.' for i in range(v.MAX_UNITS+1))
        self.assertEqual(v.verify(text,text,lambda a,b:1/0)['status'],'abstain')
        with self.assertRaises(ValueError):v.verify('Firms invest.','Firms invest.',lambda a,b:{'entailment':float('nan'),'neutral':0.,'contradiction':0.})

    def test_finite_bound_guard_is_not_overridden_by_nli(self):
        source='The approximation works when noise does not exceed the threshold.'
        target='The approximation works when noise does not fall below the threshold.'
        result=v.verify(source,target,lambda a,b:ENTAIL)
        self.assertEqual(result['status'],'risk_signal')
        self.assertIn('scoped_bound_reversal_candidate',[c['kind'] for c in result['role_conflicts']])

    def test_abstain_all_reject_all_accept_all_fail_gate(self):
        criteria={'minimum_error_detection_rate':.8,'maximum_faithful_false_alarm_rate':.1,
            'minimum_nontrivial_faithful_acceptance_rate':.5,'maximum_changed_false_acceptance_rate':0.}
        for choice in ('abstain','risk_signal','model_consistent'):
            rows=[{'expected':e,'variant':'paraphrase' if e=='faithful' else 'mutation','verdict':{'status':choice}} for e in ('faithful','changed')]
            self.assertFalse(v.gate(v.score(rows),criteria)['passed'])

    def test_fresh_protocol_and_stub_cannot_claim_readiness(self):
        root=Path(__file__).resolve().parents[1]
        folder=root/'docs/claim_verifier_calibration_v2'
        protocol,cases=v.validate_protocol(folder)
        self.assertEqual(sum(c['split']=='held_out' for c in cases),48)
        self.assertEqual(sum(c['split']=='development' for c in cases),72)
        self.assertTrue(all(c['provenance']=='previous_v1_control_development_only' for c in cases if c['split']=='development'))
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);test_protocol=base/'protocol';test_protocol.mkdir()
            for split in ('development','held_out'):
                name=split+'.json';subset=[c for c in cases if c['split']==split][:3]
                path=test_protocol/name;path.write_text(json.dumps(subset))
                protocol['expected_counts'][split]=len(subset)
                protocol['fixture_files'][name]=hashlib.sha256(path.read_bytes()).hexdigest()
            (test_protocol/'protocol.json').write_text(json.dumps(protocol))
            with contextlib.redirect_stdout(io.StringIO()):summary=v.evaluate(test_protocol,base/'stub',lambda a,b:ENTAIL,'synthetic_stub')
            self.assertFalse(summary['readiness_gate']['passed'])
            self.assertEqual(summary['execution_kind'],'synthetic_stub')
            self.assertTrue(json.loads((base/'stub/run_status.json').read_text())['complete'])
            with self.assertRaises(ValueError):v.evaluate(test_protocol,base/'stub',lambda a,b:ENTAIL,'synthetic_stub')
            (test_protocol/'held_out.json').write_text('[]')
            with self.assertRaises(ValueError):v.validate_protocol(test_protocol)

    def test_standalone_notebook_preflight_has_no_model_access(self):
        root=Path(__file__).resolve().parents[1];book=json.loads((root/'kaggle/verifier_calibration.ipynb').read_text())
        self.assertTrue(all(not c.get('outputs') for c in book['cells']))
        setup=next(''.join(c['source']) for c in book['cells'] if c['cell_type']=='code' and 'files = json.loads' in ''.join(c['source']))
        with tempfile.TemporaryDirectory() as tmp:
            runtime=Path(tmp)/'runtime';ns={}
            exec(setup.replace('/kaggle/working/_verifier_runtime',str(runtime)),ns)
            result=subprocess.run([sys.executable,'-m','fama_style.research.claim_verifier','--output-dir',str(Path(tmp)/'preflight')],cwd=tmp,
                env={**os.environ,'PYTHONPATH':str(runtime/'src'),'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('No model download',result.stdout)
            self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main()
