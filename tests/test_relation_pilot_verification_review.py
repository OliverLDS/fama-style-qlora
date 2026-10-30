"""Fixed-text/evidence integrity and separation of model judgments from approval."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.research import relation_pilot_verification as preparation
from fama_style.research import relation_pilot_verification_review as review
from fama_style.research.author_control_replication import verify_files


def fixture_row(key='x'):
    left='Firms may invest.';right='Investors provide funds.';candidate='Firms may invest, and investors provide funds.'
    row={'annotation_version':preparation.PROTOCOL,'item_id':key,'source_paragraph':left+' '+right,
        'left_sentence':left,'right_sentence':right,'candidate':candidate,'source_units':[],
        'added_content':[],'context_evidence':[],'fidelity_verdict':'faithful','fidelity_rationale':'Both claims retain their force.',
        'issue_categories':[],'readability_in_context':'merged_better','readability_rationale':'Both actors remain clear.',
        'readability_evidence':[{'text_origin':'source','quote':left,'occurrence':1},
                                {'text_origin':'candidate','quote':candidate,'occurrence':1}],
        'candidate_single_sentence':'yes','needs_external_context':False,'review_notes':[],
        'provenance':{'visible_model_label':None,'thinking_setting':None,
                      'earlier_judgments_supplied':False,'independently_verified':False}}
    for i,(source,target) in enumerate([(left,'Firms may invest'),(right,'investors provide funds')],1):
        row['source_units'].append({'unit_id':'V'+str(i),'proposition':source,
            **{k:[] for k in ('actors_and_roles','conditions','quantities','negation','modality','attribution')},
            'source_evidence':[{'quote':source,'occurrence':1}],
            'candidate_evidence':[{'quote':target,'occurrence':1}],'alignment':'preserved','notes':''})
    return row


class VerificationReviewTests(unittest.TestCase):
    def test_fixed_text_quotes_and_unit_sequences_fail_closed(self):
        original=fixture_row();schema=preparation.output_schema()
        self.assertEqual(review.check_row(original,original,schema),[])
        for mode in ('candidate','source','quote','occurrence','unit','missing'):
            row=copy.deepcopy(original)
            if mode=='candidate':row['candidate']+=' Added.'
            if mode=='source':row['source_paragraph']+=' Added.'
            if mode=='quote':row['source_units'][0]['candidate_evidence'][0]['quote']='Absent quote'
            if mode=='occurrence':row['readability_evidence'][0]['occurrence']=2
            if mode=='unit':row['source_units'][1]['unit_id']='V1'
            if mode=='missing':row['source_units'][0]['source_evidence']=[]
            with self.subTest(mode=mode):self.assertTrue(review.check_row(row,original,schema))

    def test_faithful_overall_cannot_hide_omission_or_added_content(self):
        row=fixture_row();schema=preparation.output_schema()
        row['source_units'][0]['alignment']='omitted';row['source_units'][0]['candidate_evidence']=[]
        self.assertTrue(review.check_row(row,row,schema))
        row['fidelity_verdict']='local_shift'
        self.assertEqual(review.check_row(row,row,schema),[])
        row=fixture_row();row['added_content']=[{'description':'Possible added relation',
            'candidate_evidence':[{'quote':'and','occurrence':1}],'source_support':'uncertain','explanation':'Requires checking.'}]
        self.assertTrue(review.check_row(row,row,schema))

    def packet(self,root):
        packet=root/'packet';(packet/'common').mkdir(parents=True);(packet/'batches/batch_01').mkdir(parents=True);(packet/'responses').mkdir()
        (packet/'common/verification_output.schema.json').write_text(json.dumps(preparation.output_schema()))
        rows=[fixture_row('x'+str(i)) for i in range(4)]
        rows[1]['readability_in_context']='separated_better'
        rows[2]['fidelity_verdict']='local_shift';rows[2]['readability_in_context']='comparable'
        inputs=[{k:r[k] for k in ('item_id',*review.TEXT_FIELDS)} for r in rows]
        (packet/'batches/batch_01/items.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in inputs))
        (packet/'SESSION_NOTES.md').write_text('Blank form.')
        (packet/'PRIVATE_MAPPING.json').write_text(json.dumps([{'item_id':r['item_id'],'paragraph_id':'p'+str(i),
            'paper_id':str(i%2),'boundary_id':'B1','risk_flags':['guard_inventory_changed'] if i==0 else []} for i,r in enumerate(rows)]))
        review.file_manifest(packet)
        (packet/'responses/responses_01.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        session=root/'session.json';session.write_text(json.dumps({'visible_model_label':'GPT-6','thinking_setting':'High'}))
        return packet,session

    def test_snapshot_summary_does_not_approve_training_or_overwrite_raw_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);packet,session=self.packet(root);result=review.audit(packet,root/'audit',session)
            self.assertEqual(result['mechanically_valid_records'],4)
            self.assertEqual(result['faithful_merged_better'],2)
            self.assertEqual(result['valid_model_judgments']['fidelity'],{'faithful':3,'local_shift':1})
            self.assertEqual(result['approved_training_rows'],0)
            self.assertFalse(result['judgments_independently_verified'])
            raw=root/'audit/raw_responses/responses_01.jsonl'
            self.assertEqual(raw.read_bytes(),(packet/'responses/responses_01.jsonl').read_bytes())
            self.assertIsNone(review.read_rows(raw)[0]['provenance']['visible_model_label'])
            verify_files(root/'audit')
            with self.assertRaises(ValueError):review.audit(packet,root/'audit',session)

    def test_missing_duplicate_ids_and_changed_packet_do_not_create_audit(self):
        for mode in ('duplicate','missing','packet'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);packet,session=self.packet(root);path=packet/'responses/responses_01.jsonl'
                if mode=='duplicate':
                    rows=review.read_rows(path);rows[1]=rows[0];path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                if mode=='missing':path.unlink()
                if mode=='packet':(packet/'common/verification_output.schema.json').write_text('{}')
                with self.assertRaises(ValueError):review.audit(packet,root/'audit',session)
                self.assertFalse((root/'audit').exists())
