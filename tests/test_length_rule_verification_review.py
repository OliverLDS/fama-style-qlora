"""Frozen-rule scoring, abstention coverage and replication-aware audit checks."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.research import length_rule_verification_review as module
from fama_style.research import relation_pilot_verification as verification
from fama_style.research.author_control_replication import verify_files


class LengthRuleVerificationReviewTests(unittest.TestCase):
    def data(self):
        selection={};generation={};judgments={}
        cases=[(30,'faithful','merged_better',True),(30,'faithful','separated_better',True),
               (31,'faithful','separated_better',True),(31,'faithful','comparable',True),
               (30,'local_shift','separated_better',True),(30,None,None,False)]
        for i,(n,fidelity,readability,available) in enumerate(cases):
            key='x'+str(i);left=' '.join(['Firms']*(n-1))+'.';right='Costs.'
            selection[key]={'item_id':key,'left_sentence':left,'right_sentence':right,'pair_words':n,
                'stratum':'at_most_30' if n<=30 else 'over_30','rule_merge':n<=30,
                'paragraph_id':'p'+str(i),'paper_id':str(i%2)}
            generation[key]={'generation_status':'generated' if available else 'abstained'}
            if available:judgments[key]={'fidelity_verdict':fidelity,'readability_in_context':readability}
        return selection,generation,judgments

    def test_exact_cutoff_unavailable_candidates_and_separate_side_cohorts(self):
        selection,generation,judgments=self.data();rows=module.policy_rows(selection,generation,judgments)
        self.assertEqual(len(rows),18)
        fixed=module.policy_score([r for r in rows if r['method']=='fixed_30_available'])
        self.assertEqual((fixed['tp'],fixed['fp'],fixed['fn'],fixed['tn']),(1,1,0,1))
        self.assertEqual(fixed['primary_items'],3)
        self.assertEqual(fixed['side_cohorts']['claim_shift']['selected_merges'],1)
        self.assertEqual(fixed['side_cohorts']['faithful_comparable']['selected_merges'],0)
        self.assertEqual(fixed['side_cohorts']['generation_abstention'],{'items':1,'selected_merges':0})
        self.assertTrue(all(r['prediction']==0 for r in rows if r['item_id']=='x5'))

    def test_frozen_word_count_and_stratum_cannot_be_changed(self):
        selection,generation,judgments=self.data();selection['x0']['rule_merge']=False
        with self.assertRaisesRegex(ValueError,'frozen_rule'):module.policy_rows(selection,generation,judgments)

    def fixture(self,root):
        initial=root/'initial';gen=root/'generation';packet=root/'verification'
        for folder in (initial,gen,packet):folder.mkdir()
        (initial/'protocol.json').write_text(json.dumps(module.preparation.CONFIG))
        selections=[];generations=[];inputs=[];responses=[]
        for i in range(2):
            key='x'+str(i);left='Firms may invest.';right='Costs fall.';source=left+' '+right;candidate='Firms may invest, and costs fall.'
            selections.append({'item_id':key,'paper_id':'p'+str(i),'paragraph_id':'paragraph'+str(i),
                'pair_words':5,'stratum':'at_most_30','rule_merge':True,'left_sentence':left,'right_sentence':right})
            inp={'item_id':key,'source_paragraph':source,'left_sentence':left,'right_sentence':right,'candidate':candidate}
            inputs.append(inp);generations.append({**inp,'generation_status':'generated'})
            responses.append({**inp,'annotation_version':verification.PROTOCOL,
                'source_units':[{'unit_id':'V1','proposition':'Firms may invest and costs fall.',
                    **{k:[] for k in ('actors_and_roles','conditions','quantities','negation','modality','attribution')},
                    'source_evidence':[{'quote':source,'occurrence':1}],
                    'candidate_evidence':[{'quote':candidate,'occurrence':1}],'alignment':'preserved','notes':''}],
                'added_content':[],'context_evidence':[],'fidelity_verdict':'faithful','fidelity_rationale':'Retains both statements.',
                'issue_categories':[],'readability_in_context':'merged_better' if i==0 else 'separated_better',
                'readability_rationale':'Comparing the sentence organizations.',
                'readability_evidence':[{'text_origin':'candidate','quote':candidate,'occurrence':1}],
                'candidate_single_sentence':'yes','needs_external_context':False,'review_notes':[],
                'provenance':{'visible_model_label':None,'thinking_setting':None,
                    'earlier_judgments_supplied':False,'independently_verified':False}})
        (initial/'PRIVATE_SELECTION.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selections));module.file_manifest(initial)
        (gen/'annotations.private.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in generations))
        (gen/'summary.json').write_text(json.dumps({'invalid_records':0}))
        module.file_manifest(gen,input_packet_manifest_sha256=module.file_sha(initial/'manifest.json'))
        (packet/'common').mkdir();(packet/'batches/batch_01').mkdir(parents=True);(packet/'responses').mkdir()
        (packet/'common/verification_output.schema.json').write_text(json.dumps(verification.output_schema()))
        (packet/'batches/batch_01/items.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in inputs))
        (packet/'PRIVATE_MAPPING.json').write_text(json.dumps([{'item_id':r['item_id']} for r in inputs]))
        (packet/'SESSION_NOTES.md').write_text('Original blank form.')
        module.file_manifest(packet,generation_audit_manifest_sha256=module.file_sha(gen/'manifest.json'),initial_packet_manifest_sha256=module.file_sha(initial/'manifest.json'))
        (packet/'responses/responses_01.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in responses))
        session=root/'session.json';session.write_text(json.dumps({'visible_model_label':'GPT-6','requested_thinking_setting':'High','effective_thinking_setting':'not confirmed'}))
        return initial,gen,packet,session

    def test_raw_snapshot_and_replication_metadata_do_not_claim_positive_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);initial,gen,packet,session=self.fixture(root)
            s=module.run(initial,gen,packet,session,root/'out')
            self.assertEqual(s['mechanically_valid_records'],2);self.assertFalse(s['selected_preferred_subset'])
            self.assertFalse(s['cutoff_refitted']);self.assertEqual(s['approved_training_rows'],0)
            self.assertEqual(s['methods']['fixed_30_available']['fp'],1)
            self.assertEqual((root/'out/raw_responses/responses_01.jsonl').read_bytes(),(packet/'responses/responses_01.jsonl').read_bytes())
            self.assertIsNone(module.read_rows(root/'out/annotations.private.jsonl')[0]['provenance']['visible_model_label'])
            verify_files(root/'out')
            with self.assertRaises(ValueError):module.run(initial,gen,packet,session,root/'out')

    def test_changed_candidate_is_preserved_but_blocks_scoring(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);initial,gen,packet,session=self.fixture(root)
            path=packet/'responses/responses_01.jsonl';rows=module.read_rows(path);rows[0]['candidate']='Changed.'
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            s=module.run(initial,gen,packet,session,root/'out')
            self.assertEqual(s['status'],'blocked_mechanical_issues');self.assertEqual(s['invalid_records'],1)
            self.assertNotIn('methods',s);self.assertFalse((root/'out/policy_predictions.private.jsonl').exists())

    def test_duplicate_or_wrong_candidate_batch_stops_before_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);initial,gen,packet,session=self.fixture(root)
            path=packet/'responses/responses_01.jsonl';rows=module.read_rows(path);rows[1]=rows[0]
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError,'duplicate'):module.run(initial,gen,packet,session,root/'out')
            self.assertFalse((root/'out').exists())
