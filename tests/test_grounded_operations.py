"""Evidence, semantic-link and target-leakage boundaries, without ML dependencies."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.rhetoric import grounded_operations as e

class GroundedOperationTests(unittest.TestCase):
    def annotation(self,text,parts,groups,quotes,target=False):
        claims=[{'claim_id':'c'+str(i),'entity_refs':[],'epistemic_status':'hypothesis','quantities':[],'qualifications':[],
            'evidence':[{'quote':q,'occurrence':1}]} for i,q in enumerate(quotes,1)]
        return {'semantic_content':{'entities':[],'claims':claims,'relations':[{'relation_id':'r1','kind':'other',
            'source_claim_refs':['c1'],'target_claim_refs':['c2'],'direction':'source_to_target','explicitness':'explicit',
            'evidence':[{'quote':'Thus' if target else 'so','occurrence':1}]}]},
            'rhetorical_taxonomy':{'sentence_structure':[{'sentence_id':'s'+str(i),'sentence_text':p,'claim_refs':g,'features':[]} for i,(p,g) in enumerate(zip(parts,groups),1)],
                'paragraph_structure':{'moves':[]},'language_features':[{'label':{'name':'lexical_or_modal_hedge'}}]}}

    def fixture(self):
        source='Demand may rise, so output may increase. The approximation may apply if costs are low.'
        target='Demand may rise. Thus output may increase. For practical purposes, the approximation may apply as long as costs are low.'
        a=self.annotation(source,['Demand may rise, so output may increase.','The approximation may apply if costs are low.'],[['c1','c2'],['c3']],['Demand may rise','so output may increase','The approximation may apply if costs are low.'])
        b=self.annotation(target,['Demand may rise.','Thus output may increase.','For practical purposes, the approximation may apply as long as costs are low.'],[['c1'],['c2'],['c3']],['Demand may rise.','Thus output may increase.','For practical purposes, the approximation may apply as long as costs are low.'],True)
        row={'id':e.SPECS[0]['id'],'paper_key':'synthetic','split':'train','plain_english':source,'completion':target}
        return row,a,b

    def test_compiled_program_changes_presentation_without_new_edges(self):
        row,a,b=self.fixture();card=e.compile_card(row,a,b,e.SPECS[0]);e.validate_card(card)
        self.assertEqual(card['use'],'contrast_hypothesis');self.assertFalse(card['training_ready'])
        self.assertEqual(card['program']['preserve_source_relation_refs'],['r1'])
        self.assertEqual([s['source_claim_refs'] for s in card['program']['stages']],[['c1'],['c2'],['c3']])

    def test_missing_reference_realization_and_reversed_relation_rejected(self):
        for mode in ['missing_cue','wrong_target_packing','missing_source_relation','reverse_relation','uncovered_claim','nontraining']:
            row,a,b=self.fixture()
            if mode=='missing_cue':row['completion']=row['completion'].replace('Thus','Hence')
            if mode=='wrong_target_packing':b['rhetorical_taxonomy']['sentence_structure'][1]['claim_refs']=['c2','c3']
            if mode=='missing_source_relation':a['semantic_content']['relations']=[]
            if mode=='reverse_relation':a['semantic_content']['relations'][0]['direction']='target_to_source'
            if mode=='uncovered_claim':a['semantic_content']['claims'][0]['claim_id']='c99'
            if mode=='nontraining':row['split']='validation'
            with self.assertRaises(ValueError,msg=mode):e.compile_card(row,a,b,e.SPECS[0])

    def test_unknown_claim_relation_and_lexical_plan_fields_rejected(self):
        row,a,b=self.fixture();card=e.compile_card(row,a,b,e.SPECS[0])
        for mode in ['new_claim','new_relation','relation_endpoint','target_words','training_ready','wrong_control','span']:
            changed=copy.deepcopy(card)
            if mode=='new_claim':changed['program']['stages'][0]['source_claim_refs']=['c999']
            if mode=='new_relation':changed['program']['preserve_source_relation_refs'].append('r2')
            if mode=='relation_endpoint':changed['source_annotation']['semantic_content']['relations'][0]['source_claim_refs']=['c3']
            if mode=='target_words':changed['program']['reference_text']='PRIVATE_TARGET'
            if mode=='training_ready':changed['training_ready']=True
            if mode=='wrong_control':changed['use']='preservation_control'
            if mode=='span':changed['source_annotation']['claim_spans']['c1'][0]['start']+=1
            with self.assertRaises(ValueError,msg=mode):e.validate_card(changed)

    def test_payload_contains_only_source_evidence_and_closed_structural_controls(self):
        row,a,b=self.fixture();card=e.compile_card(row,a,b,e.SPECS[0]);payload=e.generation_payload(card)
        rendered=e.render_payload(payload)
        self.assertNotIn('Thus',rendered);self.assertNotIn('For practical purposes',rendered)
        self.assertNotIn(row['completion'],rendered);self.assertIn(row['plain_english'],rendered)
        self.assertNotIn('reference',payload);self.assertNotIn('target_annotation',payload)
        for claim in payload['source_claim_index']:
            for span in claim['source_spans']:self.assertEqual(row['plain_english'][span['start']:span['end']],span['quote'])
        with self.assertRaises(ValueError):e.render_payload({**payload,'reference':'PRIVATE_TARGET'})
        changed=copy.deepcopy(card);changed['target_annotation']['semantic_content']['unresolved_notes']=['PRIVATE_TARGET_NOTE']
        self.assertEqual(e.render_prompt(card),e.render_prompt(changed))

    def test_ambiguous_or_unresolved_evidence_fails_without_guessing(self):
        self.assertEqual(e.span('a a','a',2),{'start':2,'end':3,'quote':'a'})
        for quote,occurrence in [('a',None),('a',3),('missing',1)]:
            with self.assertRaises(ValueError):e.span('a a',quote,occurrence)

    def test_private_package_replays_and_control_not_mislabeled(self):
        path=e.ROOT/'research_runs/grounded_rhetorical_operations_v1/package_v3'
        if not path.exists():self.skipTest('Private annotation artifacts absent in public checkout')
        e.check(path);cards=e.read_lines(path/'operation_cards.private.jsonl')
        self.assertEqual(sum(c['use']=='contrast_hypothesis' for c in cards),4)
        self.assertEqual(sum(c['use']=='preservation_control' for c in cards),1)
        numeric=next(c for c in cards if c['quantity_role_bindings'])
        self.assertEqual(len(numeric['quantity_role_bindings']),7)
        changed=copy.deepcopy(numeric);changed['quantity_role_bindings'][0]['role']='actual_rate'
        with self.assertRaises(ValueError):e.validate_card(changed)
        count_card=copy.deepcopy(next(c for c in cards if c['source_fact_supplements']))
        count_card['source_fact_supplements'][0]['value']=4
        with self.assertRaises(ValueError):e.validate_card(count_card)
        for c in cards:
            e.validate_card(c)
            payload=e.generation_payload(c)
            self.assertFalse(c['training_ready']);self.assertEqual(payload['program'],e.compile_program(c))
        with tempfile.TemporaryDirectory() as folder,patch.object(e,'build_cards',side_effect=AssertionError('No dataset read needed')):
            # Existing artifact changes fail their manifest before rebuilding inputs.
            target=Path(folder)
            for name in ['manifest.json','operation_cards.private.jsonl','summary.json','prompt_preview.private.jsonl']:(target/name).write_bytes((path/name).read_bytes())
            (target/'operation_cards.private.jsonl').write_text('{}\n')
            with self.assertRaises(ValueError):e.check(target)

if __name__=='__main__':unittest.main()
