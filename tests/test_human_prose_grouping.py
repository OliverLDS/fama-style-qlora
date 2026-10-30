"""Nonidentity selection, isolation, matched payloads and balanced job allocation."""
import copy
from collections import Counter
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.research import human_prose_grouping as module


class HumanProseGroupingTests(unittest.TestCase):
    def annotation(self):
        source=['Firms borrow.','Banks lend.','Costs fall.']
        return {'plain_english':' '.join(source), 'original_paragraph':'Firms borrow while banks lend. Costs fall.',
            'semantic_ledger':{'claims':[{'claim_id':'C'+str(i+1),'alignment_status':'preserved',
                'proposition':'REFERENCE_ONLY_CONTENT','original_evidence':[{'quote':'REFERENCE_ONLY_CONTENT'}],
                'rewrite_evidence':[{'quote':quote,'occurrence':1}]} for i,quote in enumerate(source)]},
            'source_schema':{'sentences':[{'sentence_id':'S'+str(i+1),'exact_text':quote,'claim_ids':['C'+str(i+1)]}
                for i,quote in enumerate(source)]},
            'target_schema':{'sentences':[{'sentence_id':'S1','exact_text':'Firms borrow while banks lend.','claim_ids':['C1','C2']},
                {'sentence_id':'S2','exact_text':'Costs fall.','claim_ids':['C3']}]},
            'source_quality':{'needs_external_context':False},'needs_review':False}

    def records(self):
        return [{'paragraph_id':str(i),'paper_id':str(i%3),'split':'development','schema_valid':True,
            'audit':{'errors':[],'program_errors':[]},'annotation':self.annotation()} for i in range(5)]

    def test_reserved_and_previously_used_rows_are_skipped_before_reading_annotation(self):
        r=self.records()+[{'split':'reserved_test','paragraph_id':'reserved'},
                          {'split':'training','paragraph_id':'training'},
                          {'split':'development','paragraph_id':'used'}]
        selected=module.select_records(r,{'used'})
        self.assertEqual(len(selected),5)
        self.assertTrue(all(x['split']=='development' for x in selected))
        with self.assertRaises(ValueError):module.select_records(self.records(),{'0'})

    def test_identity_and_duplicate_selection_rejected(self):
        rows=self.records()
        rows[0]['annotation']['target_schema']=copy.deepcopy(rows[0]['annotation']['source_schema'])
        with self.assertRaises(ValueError):module.select_records(rows,set())
        rows=self.records();rows[-1]['paragraph_id']=rows[0]['paragraph_id']
        with self.assertRaises(ValueError):module.select_records(rows,set())

    def test_matched_inputs_and_reference_prose_omission(self):
        a=self.annotation();before=copy.deepcopy(a)
        conditions={c:module.payload(a,c) for c in module.CONDITIONS}
        plain=conditions['ordinary']
        self.assertNotIn('sentence_plan',plain)
        for condition,p in conditions.items():
            self.assertEqual({k:v for k,v in p.items() if k!='sentence_plan'},plain)
            self.assertNotIn('REFERENCE_ONLY_CONTENT',str(p))
            self.assertNotIn(a['original_paragraph'],str(p))
        self.assertEqual(conditions['source_group_plan']['sentence_plan']['sentence_claim_groups'],[['C1'],['C2'],['C3']])
        self.assertEqual(conditions['oracle_target_group_plan']['sentence_plan']['sentence_claim_groups'],[['C1','C2'],['C3']])
        self.assertEqual(a,before)

    def test_exact_evidence_full_coverage_and_unknown_group_ids(self):
        module.validate_annotation(self.annotation())
        for issue in ('evidence','unknown','incomplete','sentence_coverage','uncertain'):
            a=self.annotation()
            if issue=='evidence':a['semantic_ledger']['claims'][0]['rewrite_evidence'][0]['quote']='Invented claim.'
            elif issue=='unknown':a['target_schema']['sentences'][0]['claim_ids'].append('C9')
            elif issue=='incomplete':a['target_schema']['sentences'][1]['claim_ids']=['C2']
            elif issue=='sentence_coverage':a['plain_english']='Changed source.'
            else:a['semantic_ledger']['claims'][0]['alignment_status']='uncertain'
            with self.assertRaises(ValueError):module.validate_annotation(a)

    def test_claim_components_may_span_multiple_sentences(self):
        a=self.annotation();a['source_schema']['sentences'][1]['claim_ids'].append('C1')
        module.validate_annotation(a)
        a['source_schema']['sentences'][1]['claim_ids'].append('C1')
        with self.assertRaises(ValueError):module.validate_annotation(a)

    def test_fifteen_unique_jobs_counterbalanced_without_within_batch_pair(self):
        selected=module.select_records(self.records(),set())
        batches,key=module.make_jobs(selected)
        self.assertEqual([len(b) for b in batches],[5,5,5])
        self.assertEqual(len({j['job_id'] for j in key}),15)
        self.assertEqual(Counter(j['condition'] for j in key),{c:5 for c in module.CONDITIONS})
        for batch in (1,2,3):
            subset=[j for j in key if j['batch']==batch]
            self.assertEqual(len({j['paragraph_id'] for j in subset}),5)
            self.assertEqual(sorted(Counter(j['condition'] for j in subset).values()),[1,2,2])
        self.assertEqual(module.make_jobs(selected),(batches,key))
