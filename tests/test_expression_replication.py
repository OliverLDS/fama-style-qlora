"""Scope-preserving replication design checks, with optional private integration."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.research import expression_replication as replication


def fixture():
    source='Costs are uncertain. Wages may change. Risk persists.'
    texts=['Costs are uncertain.','Wages may change.','Risk persists.']
    spans=[replication.binding.anchored(source,t)[1] for t in texts]
    claims=[{'claim_id':f'c{i}','composite':False,'epistemic_status':'assertion','source_spans':[s],'protected':[]} for i,s in enumerate(spans,1)]
    edge={'relation_id':'r1','kind':'causal','certainty':'explicit','from_claim_refs':['c1'],'to_claim_refs':['c2'],
          'source_evidence':[{'quote':texts[0],'occurrence':1},{'quote':texts[1],'occurrence':1}]}
    card={'id':'synthetic','paper_key':'VL2018','record':{'source':source,'reference':'A private synthetic reference excluded from prompts.'},
          'semantic_guard':'Retain the stated uncertainty and risk.',
          'evidence':{'source_units':[{'unit_id':f'S{i}','text':t} for i,t in enumerate(texts,1)],'claim_index':claims,'source_relations':[edge]},
          'endpoint_bindings':[{'relation_id':'r1','from':{'claim_id':'c1','span':spans[0]},'to':{'claim_id':'c2','span':spans[1]}}],
          'plan':[{'op':'MERGE','source_unit_ids':['S1','S2','S3'],'output_slots':[{'claim_refs':['c1','c2','c3']}]}]}
    spec={'control_id':'FINITE_SUPPORTED_CONSEQUENCE','claim_refs':['c1','c2'],'relation_id':'r1','actor_claim':'c2','actor_quote':'Wages'}
    card['expression_operation']=replication.compile_operation(card,spec)
    return card,spec


class ReplicationTests(unittest.TestCase):
    def test_operation_subset_retains_third_claim(self):
        card,spec=fixture();op=replication.compile_operation(card,spec)
        self.assertEqual(op['claim_refs'],['c1','c2'])
        self.assertEqual(op['slot_claim_refs'],['c1','c2','c3'])
        self.assertEqual(op['untargeted_slot_claim_refs'],['c3'])
        self.assertEqual(op['actor_source_span']['quote'],'Wages')

    def test_reject_split_scope_and_reversed_or_interpretive_edge(self):
        card,spec=fixture();card['plan'][0]['output_slots']=[{'claim_refs':['c1']},{'claim_refs':['c2','c3']}]
        with self.assertRaisesRegex(ValueError,'share_one_slot'):replication.compile_operation(card,spec)
        for mutation in ('certainty','direction'):
            card,spec=fixture();edge=card['evidence']['source_relations'][0]
            if mutation=='certainty':edge['certainty']='interpretive'
            else:edge['from_claim_refs'],edge['to_claim_refs']=['c2'],['c1']
            with self.assertRaisesRegex(ValueError,'explicit_source_causal'):replication.compile_operation(card,spec)

    def test_actors_and_endpoints_owned_by_declared_claims(self):
        card,spec=fixture();spec['actor_quote']='Risk'
        with self.assertRaisesRegex(ValueError,'owned_evidence'):replication.compile_operation(card,spec)
        card,spec=fixture();card['endpoint_bindings'][0]['from']['span']=card['endpoint_bindings'][0]['to']['span']
        with self.assertRaises(ValueError):replication.compile_operation(card,spec)

    def test_theme_singleton_slot_and_no_invented_focus(self):
        card,_=fixture();card['plan']=[{'op':'KEEP','source_unit_ids':[f'S{i}'],'output_slots':[{'claim_refs':[f'c{i}']}]} for i in range(1,4)]
        spec={'control_id':'FOREGROUND_SUPPORTED_RELATION_OR_FUNCTION','claim_refs':['c1'],'focus_claim':'c1','focus_quote':'Costs'}
        self.assertEqual(replication.compile_operation(card,spec)['slot_claim_refs'],['c1'])
        spec['focus_quote']='Unstated mechanism'
        with self.assertRaisesRegex(ValueError,'owned_evidence'):replication.compile_operation(card,spec)

    def test_primary_control_and_guidance_only_differ_by_operation(self):
        card,_=fixture();control=replication.prompt_for(card,'grouping_control');guided=replication.prompt_for(card,'selective_expression')
        tail='\nReturn only the rewritten paragraph.'
        self.assertTrue(guided.startswith(control[:-len(tail)]+'\nClaim-bound expression operation'))
        ordinary=replication.prompt_for(card,'ordinary');self.assertNotIn('Sentence-packing edit plan:',ordinary)
        for p in (control,guided,ordinary):self.assertIn(card['semantic_guard'],p);self.assertNotIn(card['record']['reference'],p)

    def test_four_paper_twelve_job_design(self):
        card,_=fixture();cards=[]
        for paper in ('VL2018','FF2000','F1965','ME1997'):
            c=copy.deepcopy(card);c.update(id=paper,paper_key=paper);cards.append(c)
        batches,mapping=replication.build_jobs(cards)
        self.assertEqual(len(mapping),12);self.assertEqual(len({j['job_id'] for j in mapping}),12)
        self.assertEqual((batches,mapping),replication.build_jobs(list(reversed(cards))))
        for paper in ('VL2018','FF2000','F1965','ME1997'):
            self.assertEqual({j['condition'] for j in mapping if j['paper_key']==paper},set(replication.CONDITIONS))
        for i in range(1,4):
            rows=[j for j in mapping if j['batch_file']==f'jobs_{i:02d}.jsonl']
            self.assertEqual(len({j['paper_key'] for j in rows}),4)
        with self.assertRaises(ValueError):replication.build_jobs(cards[:-1])

    def test_validation_and_reserved_test_parent_rejected(self):
        for split in ('validation','test'):
            with self.assertRaisesRegex(ValueError,'wrong_parent_or_split'):replication.build_card({'split':split},{},{})

    @unittest.skipUnless((replication.ROOT/'research_runs/expression_replication_v1/specification_v1/selection.private.json').exists(),'Private replication inputs unavailable')
    def test_private_identity_scoped_operation_and_rebuild(self):
        root=replication.ROOT/'research_runs/expression_replication_v1';spec=root/'specification_v1/selection.private.json'
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'package';replication.prepare(spec,output);key=replication.check_package(output)
            self.assertEqual(len(key['jobs']),12)
            cards={c['paper_key']:c for c in key['rows']}
            self.assertEqual(cards['F1965']['grouping_origin'],'source_identity_checked_against_reference')
            self.assertTrue(all(p['op']=='KEEP' for p in cards['F1965']['plan']))
            self.assertEqual(cards['VL2018']['expression_operation']['untargeted_slot_claim_refs'],['c4'])
            self.assertTrue(all(c['split']=='train' and not c['training_ready'] for c in cards.values()))
            p=output/'batches/jobs_01.jsonl';p.write_text(p.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'parent_artifact_changed'):replication.check_package(output)


if __name__=='__main__':unittest.main()
