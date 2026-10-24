"""Scoped regression/negative controls; fake NLI is not model-quality evidence."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.research import relation_guards as g
from fama_style.research import claim_verifier_v3 as v3

ENTAIL={'entailment':.99,'neutral':.01,'contradiction':0.}


class RelationGuardsTests(unittest.TestCase):
    def test_inclusive_and_strict_endpoints_both_directions(self):
        for a,b in [('at least 10','higher than 10'),('at most 5','less than 5'),('> 8','>= 8'),('below -2','at most -2')]:
            source='The score is '+a+'.';target='The score is '+b+'.'
            self.assertTrue(g.check(source,target)['conflicts'])
            self.assertTrue(g.check(target,source)['conflicts'])
            self.assertEqual(v3.verify(source,target,lambda a,b:ENTAIL)['status'],'risk_signal')

    def test_equivalent_comparators_values_and_units(self):
        for source,target in [('The score is at least ten.','The score is ten or higher.'),
                              ('The rate is at most 3.70 percent.','The rate is no greater than 3.7%.'),
                              ('Noise does not exceed six units.','Noise is at most 6 units.'),
                              ('The score is at least -ten.','The score is -10 or higher.')]:
            checks=g.check(source,target)
            self.assertFalse(checks['conflicts']);self.assertFalse(checks['unresolved'])

    def test_variable_unit_and_ambiguous_scope_are_not_guessed(self):
        for source,target in [('The score is at least 10.','The cost is higher than 10.'),
                              ('The rate is at least 5 percent.','The rate is higher than 5 units.'),
                              ('The score is at least 10. The score is at least 10.','The score is higher than 10.'),
                              ('The score may be at least 10.','The score might be at least 10.')]:
            checks=g.check(source,target)
            self.assertFalse(checks['conflicts']);self.assertTrue(checks['unresolved'])
            self.assertEqual(v3.verify(source,target,lambda a,b:ENTAIL)['status'],'abstain')
        # Integer-domain equivalence is not assumed.
        self.assertFalse(g.check('The score is at least 10.','The score is higher than 9.')['conflicts'])

    def test_association_cannot_be_promoted_in_either_direction(self):
        source='Costs are associated with losses.'
        for target in ('Costs cause losses.','Losses cause costs.','Losses are caused by costs.'):
            checks=g.check(source,target)
            self.assertIn('association_strengthened_to_causation',[r['kind'] for r in checks['conflicts']])
        self.assertFalse(g.check(source,'Losses and costs are correlated.')['conflicts'])
        self.assertFalse(g.check(source,'Losses and costs are correlated.')['unresolved'])

    def test_existing_causal_evidence_is_pair_and_direction_specific(self):
        source='Costs are associated with losses. Costs cause losses.'
        self.assertFalse(g.check(source,'Costs cause losses.')['conflicts'])
        self.assertTrue(g.check(source,'Losses cause costs.')['conflicts'])
        unrelated='Costs are associated with losses. Demand causes revenue.'
        self.assertTrue(g.check(unrelated,'Costs cause losses. Demand causes revenue.')['conflicts'])
        self.assertFalse(g.check(unrelated,'Costs are associated with losses. Revenue is caused by demand.')['conflicts'])
        conditional='Costs are associated with losses. Costs cause losses if demand falls.'
        self.assertTrue(g.check(conditional,'Costs cause losses.')['conflicts'])
        self.assertTrue(g.check('Costs cause losses if demand falls.','Costs cause losses when demand falls.')['unresolved'])

    def test_negation_modality_and_attribution_are_not_positive_causality(self):
        for target in ('Costs do not cause losses.','Costs may cause losses.','The authors report that costs cause losses.'):
            checks=g.check('Costs are associated with losses.',target)
            self.assertFalse(checks['conflicts']);self.assertTrue(checks['unresolved'])
        # An unresolved guard cannot erase an existing NLI risk signal.
        neutral=lambda a,b:{'entailment':.01,'neutral':.99,'contradiction':0.}
        self.assertEqual(v3.verify('Costs are associated with losses.','Costs may cause losses.',neutral)['status'],'risk_signal')

    def test_identity_is_not_a_style_success_or_semantic_proof(self):
        text='The authors report that costs may cause losses.'
        result=v3.verify(text,text,lambda a,b:ENTAIL)
        self.assertEqual(result['status'],'model_consistent')
        self.assertTrue(result['scoped_relation_guards']['identity_bypass'])
        self.assertFalse(result['semantic_proof'])

    def test_cached_replay_is_development_only_and_preserves_v2(self):
        pairs=[('The score is at least 10.','The score is higher than 10.'),
               ('Costs are associated with losses.','Costs cause losses.')]
        rows=[{'id':str(i),'split':'held_out','source':a,'output':b,'verdict':v3.base.verify(a,b,lambda a,b:ENTAIL)} for i,(a,b) in enumerate(pairs)]
        before=json.dumps(rows,sort_keys=True);replayed=v3.replay_v2(rows)
        self.assertEqual(before,json.dumps(rows,sort_keys=True))
        self.assertTrue(all(r['split']=='development' and r['original_v2_split']=='held_out' for r in replayed))
        self.assertTrue(all(r['verdict']['v2_status']=='model_consistent' and r['verdict']['status']=='risk_signal' for r in replayed))

    def test_frozen_v3_protocol_and_fresh_cases(self):
        root=Path(__file__).resolve().parents[1]
        protocol,cases=v3.validate_protocol(root/'docs/claim_verifier_calibration_v3')
        self.assertEqual(len(cases),168)
        dev=[c for c in cases if c['split']=='development'];held=[c for c in cases if c['split']=='held_out']
        self.assertEqual(len(dev),120);self.assertEqual(len(held),48)
        self.assertTrue(all(c['provenance']=='previously_exposed_v2_case_development_only' for c in dev))
        self.assertFalse({c['source'] for c in dev}&{c['source'] for c in held})
        self.assertFalse({c['scenario_group'] for c in dev}&{c['scenario_group'] for c in held})

    def test_standalone_v3_preflight(self):
        root=Path(__file__).resolve().parents[1]
        book=json.loads((root/'kaggle/verifier_calibration_v3.ipynb').read_text())
        setup=next(''.join(c['source']) for c in book['cells'] if c['cell_type']=='code' and 'files = json.loads' in ''.join(c['source']))
        self.assertTrue(all(not c.get('outputs') for c in book['cells']))
        with tempfile.TemporaryDirectory() as tmp:
            runtime=Path(tmp)/'runtime';ns={}
            exec(setup.replace('/kaggle/working/_verifier_v3_runtime',str(runtime)),ns)
            result=subprocess.run([sys.executable,'-m','fama_style.research.claim_verifier_v3','--output-dir',str(Path(tmp)/'preflight')],cwd=tmp,
                env={**os.environ,'PYTHONPATH':str(runtime/'src'),'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr);self.assertIn('No model download',result.stdout)


if __name__=='__main__':unittest.main()
