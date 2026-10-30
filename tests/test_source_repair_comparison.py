"""Balanced, source-owned preparation checks without model weights or GPUs."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.research import source_repair_comparison as comparison
from fama_style.prompting.economics import make_prompt


def synthetic_cards():
    cards = []
    for paper in ('P1', 'P2', 'P3', 'P4'):
        for variant in ('original', 'repaired'):
            text = 'Costs rise. Demand falls.' if variant == 'original' else 'Costs rise. Demand often falls.'
            first, second = text.split('. ', 1); first += '.'
            units = [{'unit_id':'S1','text': first}, {'unit_id':'S2','text':second}]
            claims = []
            for i,u in enumerate(units,1):
                start = text.index(u['text'])
                claims.append({'claim_id':f'c{i}','composite':False,'epistemic_status':'assertion',
                    'source_spans':[{'start':start,'end':start+len(u['text']),'quote':u['text']}],'protected':[]})
            cards.append({'paper_key':paper,'input_variant':variant,'id':paper+'_'+variant,
                'record':{'source':text,'reference':'A synthetic reference that is excluded from prompts.'},
                'evidence':{'source_units':units,'claim_index':claims,'source_relations':[]},
                'plan':[{'op':'MERGE','source_unit_ids':['S1','S2'],'output_slots':[{'claim_refs':['c1','c2']}]}]})
    return cards


class SourceRepairComparisonTests(unittest.TestCase):
    def test_balanced_complete_reproducible_design(self):
        cards = synthetic_cards(); before = copy.deepcopy(cards)
        batches, mapping = comparison.build_jobs(cards)
        self.assertEqual(cards, before)
        self.assertEqual((batches,mapping),comparison.build_jobs(cards))
        self.assertEqual(len(mapping),16)
        self.assertEqual(len({r['job_id'] for r in mapping}),16)
        for i,batch in enumerate(batches,1):
            rows = [r for r in mapping if r['batch_file']==f'jobs_{i:02d}.jsonl']
            self.assertEqual(len(batch),4)
            self.assertEqual(len({r['paper_key'] for r in rows}),4)
            self.assertEqual(len({r['condition'] for r in rows}),4)
        for paper in ('P1','P2','P3','P4'):
            self.assertEqual(len({r['condition'] for r in mapping if r['paper_key']==paper}),4)

    def test_only_plan_changes_within_input(self):
        card = synthetic_cards()[0]
        ordinary = comparison.prompt_for(card,'ordinary')
        planned = comparison.prompt_for(card,'planned')
        tail = '\nReturn only the rewritten paragraph.'
        self.assertTrue(planned.startswith(ordinary[:-len(tail)]+'\nSentence-packing edit plan:'))
        self.assertNotIn('Sentence-packing edit plan:',ordinary)
        self.assertNotIn(card['record']['reference'],planned)
        self.assertEqual(comparison.base.sha(planned),comparison.base.sha(comparison.prompt_for(card,'planned')))

    def test_reference_guard(self):
        cards = synthetic_cards(); cards[0]['record']['reference'] = cards[0]['record']['source']
        with self.assertRaisesRegex(ValueError,'reference_leakage'): comparison.build_jobs(cards)

    def test_reject_incomplete_design(self):
        with self.assertRaises(ValueError): comparison.build_jobs(synthetic_cards()[:-1])

    def test_original_spec_does_not_insert_repaired_text(self):
        spec = {'paper_key':'FJ1983','claims':[{'claim_id':'c1','source_quote':'An augmented definition.'}],
            'relations':[], 'source_facets':[{'quote':'Only repaired text'}], 'changes_reviewed':[],
            'intra_claim_constraints':[{'parts':[{'quote':'An augmented definition.'}],'rule':'reviewed'}]}
        original = comparison.original_spec(spec, {'claims':[{'claim_id':'c1','source_evidence':[{'quote':'Approval happens.','occurrence':1}]}]},
                                              'Approval happens.','Private target.')
        self.assertEqual(original['claims'][0]['source_quote'],'Approval happens.')
        self.assertEqual(original['intra_claim_constraints'][0]['parts'][0]['quote'],'approval')
        self.assertEqual(original['source_facets'],[])
        self.assertEqual(spec['claims'][0]['source_quote'],'An augmented definition.')

    @unittest.skipUnless((comparison.ROOT/'research_runs/repaired_source_bindings_v1/release_v1/manifest.json').exists(),'Private repaired bindings unavailable')
    def test_private_original_repaired_controls(self):
        cards,_ = comparison.load_cards()
        self.assertEqual(len(cards),8)
        for paper in {c['paper_key'] for c in cards}:
            pair = {c['input_variant']:c for c in cards if c['paper_key']==paper}
            a,b = pair['original'],pair['repaired']
            self.assertNotEqual(a['source_sha256'],b['source_sha256'])
            self.assertEqual(a['target_sha256'],b['target_sha256'])
            self.assertEqual(a['plan'],b['plan'])
            self.assertEqual([(c['claim_id'],c['epistemic_status'],[p['kind'] for p in c['protected']]) for c in a['annotation']['claims']],
                             [(c['claim_id'],c['epistemic_status'],[p['kind'] for p in c['protected']]) for c in b['annotation']['claims']])
            for card in pair.values():
                self.assertNotIn('r2',[r['relation_id'] for r in card['evidence']['source_relations']] if paper=='ME1997' else [])
                for relation in card['endpoint_bindings']:
                    for side in ('from','to'):
                        endpoint=relation[side]; owner=next(c for c in card['evidence']['claim_index'] if c['claim_id']==endpoint['claim_id'])
                        span=endpoint['span']
                        self.assertTrue(any(o['start']<=span['start']<span['end']<=o['end'] for o in owner['source_spans']))

    @unittest.skipUnless((comparison.ROOT/'research_runs/repaired_source_bindings_v1/release_v1/manifest.json').exists(),'Private repaired bindings unavailable')
    def test_private_prepare_check_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'package'; comparison.prepare(output)
            key=comparison.check_package(output)
            self.assertEqual(len(key['jobs']),16)
            self.assertEqual(json.loads((output/'summary.json').read_text())['model_calls'],0)
            path=output/'batches/jobs_01.jsonl'; path.write_text(path.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'parent_artifact_changed'): comparison.check_package(output)


if __name__ == '__main__': unittest.main()
