"""Grounding, control isolation and reproducibility checks; no model required."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.research import claim_bound_expression as experiment


def synthetic_card():
    source='Pay may stay. Future wages change. Managers gain a stake.'
    parts=source.split('. ');parts=[p if p.endswith('.') else p+'.' for p in parts]
    spans=[experiment.binding.anchored(source,p)[1] for p in parts]
    edge={'relation_id':'r1','kind':'causal','from_claim_refs':['c2'],'to_claim_refs':['c3'],
          'source_evidence':[{'quote':parts[1],'occurrence':1},{'quote':parts[2],'occurrence':1}],'certainty':'explicit'}
    card={'id':'synthetic','paper_key':'F1980','input_variant':'repaired','context_flags':[],'alignment_caveats':[],
          'record':{'source':source,'reference':'Private synthetic reference excluded from prompts.'},
          'evidence':{'source_units':[{'unit_id':f'S{i}','text':p} for i,p in enumerate(parts,1)],
              'claim_index':[{'claim_id':f'c{i}','composite':False,'epistemic_status':'assertion','source_spans':[span],'protected':[]} for i,span in enumerate(spans,1)],
              'source_relations':[edge]},
          'endpoint_bindings':[{'relation_id':'r1','from':{'claim_id':'c2','span':spans[1]},'to':{'claim_id':'c3','span':spans[2]}}],
          'plan':[{'op':'KEEP','source_unit_ids':['S1'],'output_slots':[{'claim_refs':['c1']}]},
                  {'op':'MERGE','source_unit_ids':['S2','S3'],'output_slots':[{'claim_refs':['c2','c3']}]}]}
    spec={'paper_key':'F1980','control_id':'FINITE_SUPPORTED_CONSEQUENCE','claim_refs':['c2','c3'],
          'relation_id':'r1','actor_claim':'c3','actor_quote':'Managers'}
    card['expression_operation']=experiment.compile_operation(card,spec)
    return card,spec


class ClaimBoundExpressionTests(unittest.TestCase):
    def test_explicit_consequence_grounded_in_same_slot(self):
        card,spec=synthetic_card(); operation=experiment.compile_operation(card,spec)
        self.assertEqual(operation['output_slot'],2)
        self.assertEqual(operation['actor_source_span']['quote'],'Managers')
        self.assertNotIn('reference',operation)

    def test_reject_invented_reversed_or_interpretive_edge(self):
        for mutation in ('certainty','direction','missing'):
            card,spec=synthetic_card()
            if mutation=='certainty':card['evidence']['source_relations'][0]['certainty']='interpretive'
            elif mutation=='direction':card['evidence']['source_relations'][0]['from_claim_refs']=['c3']
            else:card['evidence']['source_relations']=[]
            with self.assertRaises(ValueError):experiment.compile_operation(card,spec)

    def test_endpoint_actor_and_grouping_ownership(self):
        card,spec=synthetic_card();card['endpoint_bindings'][0]['from']['span']=card['endpoint_bindings'][0]['to']['span']
        with self.assertRaisesRegex(ValueError,'misbound_consequence_endpoint'):experiment.compile_operation(card,spec)
        card,spec=synthetic_card();spec['actor_quote']='Future wages'
        with self.assertRaisesRegex(ValueError,'owned_evidence'):experiment.compile_operation(card,spec)
        card,spec=synthetic_card();card['plan'][1]['output_slots']=[{'claim_refs':['c2']},{'claim_refs':['c3']}]
        with self.assertRaisesRegex(ValueError,'fixed_shared_slot'):experiment.compile_operation(card,spec)

    def test_theme_focus_requires_owning_claim_and_bounded_input(self):
        card,_=synthetic_card();card['paper_key']='FJ1983'
        spec={'paper_key':'FJ1983','control_id':'FOREGROUND_SUPPORTED_RELATION_OR_FUNCTION',
              'claim_refs':['c2','c3'],'focus_claim':'c2','focus_quote':'Future wages'}
        self.assertEqual(experiment.compile_operation(card,spec)['focus_source_span']['quote'],'Future wages')
        spec['focus_quote']='Managers'
        with self.assertRaisesRegex(ValueError,'owned_evidence'):experiment.compile_operation(card,spec)
        card['context_flags']=['unresolved']
        with self.assertRaisesRegex(ValueError,'bounded_case'):experiment.compile_operation(card,spec)

    def test_primary_prompts_differ_only_by_operation_block(self):
        card,_=synthetic_card();control=experiment.prompt_for(card,'grouping_control')
        guided=experiment.prompt_for(card,'selective_expression');tail='\nReturn only the rewritten paragraph.'
        self.assertTrue(guided.startswith(control[:-len(tail)]+'\nClaim-bound expression operation'))
        ordinary=experiment.prompt_for(card,'ordinary')
        self.assertNotIn('Sentence-packing edit plan:',ordinary)
        for text in (control,guided,ordinary):
            self.assertIn(experiment.SEMANTIC_GUARDS[card['paper_key']],text)
            self.assertNotIn(card['record']['reference'],text)

    def test_six_jobs_cover_every_paper_condition_once(self):
        card,_=synthetic_card();second=copy.deepcopy(card);second.update(paper_key='FJ1983',id='second')
        batches,mapping=experiment.build_jobs([card,second])
        self.assertEqual(len(mapping),6);self.assertEqual(len({j['job_id'] for j in mapping}),6)
        self.assertEqual((batches,mapping),experiment.build_jobs([second,card]))
        for paper in ('F1980','FJ1983'):
            self.assertEqual({j['condition'] for j in mapping if j['paper_key']==paper},set(experiment.CONDITIONS))
        for i in range(1,4):
            self.assertEqual(len({j['paper_key'] for j in mapping if j['batch_file']==f'jobs_{i:02d}.jsonl'}),2)

    @unittest.skipUnless((experiment.ROOT/'research_runs/claim_bound_expression_v1/specification_v1/operations.private.json').exists(),'Private parent artifacts unavailable')
    def test_private_package_reproduction_and_tampering(self):
        spec=experiment.ROOT/'research_runs/claim_bound_expression_v1/specification_v1/operations.private.json'
        with tempfile.TemporaryDirectory() as tmp:
            package=Path(tmp)/'package';experiment.prepare(spec,package)
            key=experiment.check_package(package);self.assertEqual(len(key['jobs']),6)
            for card in key['rows']:
                self.assertFalse(card['context_flags']);self.assertFalse(card['alignment_caveats'])
            self.assertEqual(json.loads((package/'summary.json').read_text())['model_calls'],0)
            p=package/'batches/jobs_01.jsonl';p.write_text(p.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'parent_artifact_changed'):experiment.check_package(package)


if __name__=='__main__':unittest.main()
