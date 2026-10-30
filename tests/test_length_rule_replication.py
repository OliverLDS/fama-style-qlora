"""Frozen cutoff, nonreserved selection and blinded two-stage payload tests."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.research import length_rule_replication as module
from fama_style.research.author_control_replication import verify_files


class LengthRuleReplicationTests(unittest.TestCase):
    def fixture(self,root):
        audit=root/'audit';selector=root/'selector';audit.mkdir();selector.mkdir()
        papers=[{'paper_id':'p'+str(i),'title':'Synthetic Study '+chr(65+i),
            'authors':[chr(65+i)+'ster Writer'],'publication_year':1900+i,
            'publication_evidence_url':'https://example.org/p'+str(i),'split':'training'} for i in range(4)]
        rows=[]
        for p in papers:
            for i in range(5):
                sentences=['Firms may invest.','Investors supply funds.',
                    'Researchers study '+('economic relationships under specified conditions '*7)+'carefully.',
                    'Conditions can change.']
                a={'plain_english':' '.join(sentences),'rewrite_status':'simplified',
                    'source_schema':{'sentences':[{'exact_text':s} for s in sentences]},
                    'target_schema':'POISON_TARGET','semantic_ledger':'POISON_LEDGER','transformations':'POISON_LABEL',
                    'source_quality':{'status':'uncertain' if i else 'usable','issues':['Context unavailable.'] if i else [],'needs_external_context':bool(i)},
                    'needs_review':bool(i)}
                rows.append({'paragraph_id':p['paper_id']+'_'+str(i),'paper_id':p['paper_id'],'split':'training',
                    'schema_valid':True,'audit':{'errors':[],'program_errors':[]},'annotation':a})
        # Deliberately malformed excluded bodies prove they are not deserialized.
        (audit/'audits.private.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows)+
            '{"split":"reserved_test","annotation": UNREADABLE_RESERVED_BODY}\n'+
            '{"split":"development","annotation": UNREADABLE_DEVELOPMENT_BODY}\n')
        module.file_manifest(audit)
        old={'papers':[{'paper_id':'old'+str(i),'authors':['Prior '+chr(65+i)]} for i in range(4)]}
        prior=root/'prior.json';prior.write_text(json.dumps(old))
        registry=root/'registry.json';registry.write_text(json.dumps({'papers':papers}))
        (selector/'summary.json').write_text(json.dumps({'selected_thresholds':{p['paper_id']:{'length_only':30} for p in old['papers']},
            'per_paper':{p['paper_id']:{} for p in old['papers']}}))
        module.file_manifest(selector)
        return audit,registry,selector,prior

    def test_nonreserved_balanced_selection_has_no_labels_or_originals_in_payload(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);audit,registry,selector,prior=self.fixture(root)
            summary=module.build(audit,registry,selector,prior,root/'out')
            self.assertEqual(summary['strata'],{'at_most_30':16,'over_30':16})
            self.assertEqual(summary['items'],32);self.assertFalse(summary['verification_inputs_created'])
            seen=[]
            for file in sorted((root/'out/generation').glob('*/items.jsonl')):
                rows=module.read_rows(file);self.assertEqual(len(rows),4)
                self.assertTrue(all(set(r)=={'item_id','source_paragraph','left_sentence','right_sentence'} for r in rows))
                self.assertNotIn('POISON',file.read_text());seen.extend(r['item_id'] for r in rows)
            self.assertEqual(len(set(seen)),32)
            self.assertFalse((root/'out/verification').exists())
            verify_files(root/'out');(root/'out/SESSION_NOTES.md').write_text('Effort uncertain.');verify_files(root/'out')
            with self.assertRaises(ValueError):module.build(audit,registry,selector,prior,root/'out')

    def test_rule_and_sampling_are_fixed_and_do_not_use_paired_judgments(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);audit,registry,_,prior=self.fixture(root)
            registry=json.loads(registry.read_text());prior=json.loads(prior.read_text())
            pool,_=module.source_pool(audit,registry);first,_=module.select(pool,registry,prior)
            second,_=module.select(list(reversed(pool)),registry,prior)
            self.assertEqual(first,second)
            self.assertTrue(all(r['rule_merge']==(r['pair_words']<=30) for r in first))
            self.assertNotIn('POISON',str(first))
            poisoned=copy.deepcopy(prior);poisoned['papers'][0]['paper_id']='p0'
            with self.assertRaises(ValueError):module.select(pool,registry,poisoned)
            poisoned=copy.deepcopy(prior);poisoned['papers'][0]['authors']=registry['papers'][0]['authors']
            with self.assertRaises(ValueError):module.select(pool,registry,poisoned)

    def test_generation_abstention_exact_context_and_later_projection(self):
        expected={'item_id':'x','source_paragraph':'Firms may invest. Costs fall.',
                  'left_sentence':'Firms may invest.','right_sentence':'Costs fall.'}
        row={**expected,'annotation_version':module.PROTOCOL+'_generation','candidate':'Firms may invest, and costs fall.',
            'generation_status':'generated','abstention_reason':'','source_context_evidence':[],
            'provenance':{'visible_model_label':None,'thinking_setting':None,'earlier_judgments_supplied':False,'independently_verified':False}}
        self.assertEqual(module.check_generation(row,expected),[])
        projected=module.verification_payload({**row,'extra_prior_label':'POISON'})
        self.assertEqual(set(projected),{'item_id','source_paragraph','left_sentence','right_sentence','candidate'})
        self.assertNotIn('POISON',str(projected))
        bad=copy.deepcopy(row);bad['source_context_evidence']=[{'quote':'Absent','occurrence':1}]
        self.assertTrue(module.check_generation(bad,expected))
        bad=copy.deepcopy(row);bad['left_sentence']='Changed';self.assertTrue(module.check_generation(bad,expected))
        row.update(generation_status='abstained',candidate=None,abstention_reason='Cannot safely merge.')
        self.assertEqual(module.check_generation(row,expected),[])
        with self.assertRaises(ValueError):module.verification_payload(row)
        row['candidate']='Invented candidate';self.assertTrue(module.check_generation(row,expected))

    def test_changed_cutoff_or_upstream_bytes_stop_before_output(self):
        for mode in ('cutoff','tamper'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);audit,registry,selector,prior=self.fixture(root)
                if mode=='cutoff':
                    path=selector/'summary.json';s=json.loads(path.read_text());s['selected_thresholds']['old0']['length_only']=40;path.write_text(json.dumps(s));module.file_manifest(selector)
                else:(audit/'audits.private.jsonl').write_text('tampered')
                with self.assertRaises(ValueError):module.build(audit,registry,selector,prior,root/'out')
                self.assertFalse((root/'out').exists())
