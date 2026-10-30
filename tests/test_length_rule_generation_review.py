"""Raw generation preservation, availability and cutoff-blind verification."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.research import length_rule_generation_review as module
from fama_style.research import length_rule_replication as preparation
from fama_style.research.author_control_replication import verify_files


class LengthRuleGenerationReviewTests(unittest.TestCase):
    def fixture(self,root):
        packet=root/'packet';(packet/'common').mkdir(parents=True);(packet/'responses/generation').mkdir(parents=True)
        (packet/'common/generation_output.schema.json').write_text(json.dumps(preparation.generation_schema()))
        (packet/'common/verification_output.schema.json').write_text(json.dumps(preparation.verification_schema()))
        (packet/'common/VERIFICATION_PROMPT_SPECIFICATION.md').write_text(preparation.VERIFICATION_PROMPT)
        (packet/'SESSION_NOTES.md').write_text('Original blank form.')
        selection=[]
        for batch in (1,2):
            folder=packet/('generation/batch_%02d'%batch);folder.mkdir(parents=True)
            inputs=[];responses=[]
            for j in range(2):
                key='x%d_%d'%(batch,j);inp={'item_id':key,'source_paragraph':'Firms may invest. Costs fall.',
                    'left_sentence':'Firms may invest.','right_sentence':'Costs fall.'}
                status='abstained' if batch==2 and j==1 else 'generated'
                responses.append({**inp,'annotation_version':preparation.PROTOCOL+'_generation',
                    'candidate':None if status=='abstained' else 'Firms may invest because costs fall.',
                    'generation_status':status,'abstention_reason':'Cannot safely merge.' if status=='abstained' else '',
                    'source_context_evidence':[],
                    'provenance':{'visible_model_label':None,'thinking_setting':None,'earlier_judgments_supplied':False,'independently_verified':False}})
                inputs.append(inp);selection.append({**inp,'paragraph_id':'p'+key,'paper_id':'paper'+str(j),
                    'boundary_id':'B1','stratum':'at_most_30' if batch==1 else 'over_30',
                    'rule_merge':batch==1,'pair_words':20 if batch==1 else 40})
            (folder/'items.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in inputs))
            (packet/('responses/generation/responses_%02d.jsonl'%batch)).write_text(''.join(json.dumps(r)+'\n' for r in responses))
        (packet/'PRIVATE_SELECTION.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selection))
        module.file_manifest(packet)
        manifest=json.loads((packet/'manifest.json').read_text())
        manifest['files']={k:v for k,v in manifest['files'].items() if not k.startswith('responses/') and k!='SESSION_NOTES.md'}
        (packet/'manifest.json').write_text(json.dumps(manifest))
        session=root/'session.json';session.write_text(json.dumps({'visible_model_label':'GPT-6',
            'requested_thinking_setting':'High','effective_thinking_setting':'not confirmed'}))
        return packet,session

    def test_snapshot_and_verification_include_both_strata_not_abstentions_or_hints(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);packet,session=self.fixture(root)
            summary=module.audit(packet,root/'audit',session)
            self.assertEqual(summary['generation_status'],{'generated':3,'abstained':1})
            self.assertFalse(summary['verification_outcomes_available'])
            raw=root/'audit/raw_generation/responses_01.jsonl'
            self.assertEqual(raw.read_bytes(),(packet/'responses/generation/responses_01.jsonl').read_bytes())
            self.assertIsNone(module.read_rows(raw)[0]['provenance']['visible_model_label'])
            report=module.prepare_verification(packet,root/'audit',root/'verification')
            self.assertEqual(report['items'],3);self.assertEqual(report['unavailable_generation_abstentions'],1)
            self.assertEqual(report['strata'],{'at_most_30':2,'over_30':1})
            for file in (root/'verification/batches').glob('*/items.jsonl'):
                for row in module.read_rows(file):
                    self.assertEqual(set(row),{'item_id','source_paragraph','left_sentence','right_sentence','candidate'})
                    self.assertNotEqual(row['item_id'],'x2_1')
            verify_files(root/'audit');verify_files(root/'verification')
            (root/'verification/SESSION_NOTES.md').write_text('Effective effort unknown.');verify_files(root/'verification')
            with self.assertRaises(ValueError):module.prepare_verification(packet,root/'audit',root/'verification')

    def test_invalid_fixed_text_is_retained_but_cannot_be_forwarded(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);packet,session=self.fixture(root)
            path=packet/'responses/generation/responses_01.jsonl';rows=module.read_rows(path)
            rows[0]['left_sentence']='Changed source.';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            result=module.audit(packet,root/'audit',session)
            self.assertEqual(result['invalid_records'],1)
            with self.assertRaisesRegex(ValueError,'invalid_generation_records'):
                module.prepare_verification(packet,root/'audit',root/'verification')
            self.assertFalse((root/'verification').exists())

    def test_missing_duplicate_wrong_batch_and_tampered_packet_stop_before_output(self):
        for mode in ('missing','duplicate','wrong_batch','packet'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);packet,session=self.fixture(root);path=packet/'responses/generation/responses_01.jsonl'
                if mode=='missing':path.unlink()
                if mode in ('duplicate','wrong_batch'):
                    rows=module.read_rows(path)
                    if mode=='duplicate':rows[1]=rows[0]
                    else:rows[1]['item_id']='other_batch'
                    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                if mode=='packet':(packet/'common/generation_output.schema.json').write_text('{}')
                with self.assertRaises(ValueError):module.audit(packet,root/'audit',session)
                self.assertFalse((root/'audit').exists())
