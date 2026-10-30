"""Prior-judgment isolation, immutable candidate selection and schema checks."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.research import relation_pilot_verification as module
from fama_style.research.human_prose_review import schema_errors
from fama_style.research.author_control_replication import verify_files


def source(pid='p'):
    return {'paragraph_id': pid, 'source_paragraph': 'Firms may invest. Investors provide funds.',
        'source_sentences': [{'sentence_id': 'S1', 'exact_text': 'Firms may invest.'},
                             {'sentence_id': 'S2', 'exact_text': 'Investors provide funds.'}],
        'original_paragraph': 'PRIVATE_REFERENCE', 'source_claims': ['PRIVATE_CLAIMS'],
        'boundaries': [{'boundary_id': 'B1', 'left_sentence_id': 'S1', 'right_sentence_id': 'S2',
            'merge_candidate': 'Firms may invest, and investors provide funds.',
            'presentation_preference': 'MERGE', 'permissible': 'BOTH', 'rationale': 'PRIVATE_PRIOR_JUDGMENT'}]}


class RelationPilotVerificationTests(unittest.TestCase):
    def test_projection_excludes_reference_claims_labels_and_warning_hints(self):
        row = source(); result = module.payload(row, row['boundaries'][0])
        self.assertEqual(set(result), {'item_id', 'source_paragraph', 'left_sentence', 'right_sentence', 'candidate'})
        self.assertNotIn('PRIVATE', str(result))
        self.assertEqual(result['candidate'], row['boundaries'][0]['merge_candidate'])
        row['boundaries'][0]['boundary_id'] = 'B2'
        self.assertNotEqual(result['item_id'], module.payload(row, row['boundaries'][0])['item_id'])

    def test_batch_allocation_is_deterministic_and_keeps_repeated_paragraphs_apart(self):
        jobs = [{'paragraph_id': 'p'+str(i), 'paper_id': str(i%4),
                 'payload': {'item_id': str(i)+'_'+str(j)}} for i in range(13) for j in range(4)]
        first = module.allocate(jobs); second = module.allocate(list(reversed(jobs)))
        self.assertEqual(first, second); self.assertEqual([len(b) for b in first], [4]*13)
        self.assertTrue(all(len({j['paragraph_id'] for j in b})==len(b) for b in first))

    def test_schema_separates_fidelity_readability_and_uncertain_effort(self):
        schema = module.output_schema(); properties = schema['properties']
        self.assertIn('local_shift', properties['fidelity_verdict']['enum'])
        self.assertIn('separated_better', properties['readability_in_context']['enum'])
        self.assertIn('unresolved', properties['fidelity_verdict']['enum'])
        self.assertEqual(schema_errors(None, properties['provenance']['properties']['thinking_setting']), [])
        self.assertTrue(schema_errors(True, properties['provenance']['properties']['independently_verified']))
        self.assertTrue(schema_errors('KEEP', properties['fidelity_verdict']))
        self.assertFalse(schema['additionalProperties'])

    def fixture(self, root):
        audit = root/'audit'; audit.mkdir(); rows = [source('p'+str(i)) for i in range(4)]
        (audit/'annotations.private.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (audit/'row_audit.private.jsonl').write_text(''.join(json.dumps({'paragraph_id':r['paragraph_id'],
            'mechanically_valid':True,'errors':[]})+'\n' for r in rows))
        (audit/'candidate_diagnostics.private.jsonl').write_text(''.join(json.dumps({'paragraph_id':r['paragraph_id'],
            'boundary_id':'B1','risk_flags':['guard_inventory_changed'] if i==0 else []})+'\n' for i,r in enumerate(rows)))
        (audit/'summary.json').write_text(json.dumps({'invalid_records':0,'records':4,'mechanically_valid_records':4}))
        module.file_manifest(audit)
        selection = root/'selection.json'; selection.write_text(json.dumps([{'paragraph_id':r['paragraph_id'],
            'paper_id':str(i)} for i,r in enumerate(rows)]))
        return audit, selection

    def test_whole_packet_pins_texts_preserves_flagged_jobs_and_exports_no_prior_judgments(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); audit,selection=self.fixture(root)
            summary=module.build(audit,selection,root/'out',expected_merges=4)
            self.assertEqual(summary['items'],4); self.assertEqual(summary['items_with_surface_flags'],1)
            path=root/'out/batches/batch_01/items.jsonl'; rows=module.read_rows(path)
            self.assertEqual(len(rows),4); self.assertNotIn('PRIVATE',path.read_text())
            self.assertEqual(summary['responses_received'],0)
            self.assertFalse(summary['automatic_training_acceptance'])
            verify_files(root/'out')
            (root/'out/SESSION_NOTES.md').write_text('Effective effort uncertain.')
            verify_files(root/'out')
            with self.assertRaises(ValueError):module.build(audit,selection,root/'out',expected_merges=4)

    def test_stale_audit_or_wrong_selection_count_stops_before_output(self):
        for mode in ('tamper','count'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                root=Path(temp);audit,selection=self.fixture(root)
                if mode=='tamper':(audit/'annotations.private.jsonl').write_text('changed')
                with self.assertRaises(ValueError): module.build(audit,selection,root/'out',expected_merges=52)
                self.assertFalse((root/'out').exists())
