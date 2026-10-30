"""Source integrity, boundary completeness and unverified candidate semantics."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.research import relation_pilot as preparation
from fama_style.research import relation_pilot_stage2_review as review
from fama_style.research.author_control_replication import verify_files


def annotation(pid='p'):
    texts = ['Firms may invest if costs fall.', 'Investors provide funds.']
    claims = []
    for i, text in enumerate(texts, 1):
        claims.append({'claim_id': 'D'+str(i), 'proposition': text,
            **{k: [] for k in ('actors_and_roles', 'conditions', 'quantities', 'negation', 'modality', 'attribution')},
            'source_evidence': [{'quote': text, 'occurrence': 1}]})
    return {'annotation_version': 'relation_pilot_v1_stage2', 'paragraph_id': pid,
        'source_paragraph': ' '.join(texts), 'source_claims': claims,
        'source_sentences': [{'sentence_id': 'S'+str(i), 'exact_text': text,
            'claim_ids': ['D'+str(i)], 'reference_only_claim_ids': []} for i, text in enumerate(texts, 1)],
        'boundaries': [{'boundary_id': 'B1', 'left_sentence_id': 'S1', 'right_sentence_id': 'S2',
            'relation': 'parallel_actions', 'source_evidence': [{'quote': t, 'occurrence': 1} for t in texts],
            'restrictions': ['Retain condition and modality.'], 'permissible': 'BOTH',
            'presentation_preference': 'MERGE', 'rationale': 'Related actions can be combined.',
            'confidence': 'medium', 'merge_candidate': 'Firms may invest if costs fall, and investors provide funds.',
            'candidate_quality': 'unverified'}], 'needs_review': True, 'needs_external_context': False,
        'review_notes': ['Candidate not independently checked.'], 'provenance': {'visible_model_label': 'GPT-6',
            'thinking_setting': None, 'reference_supplied': False, 'independently_verified': False}}


class RelationPilotStage2ReviewTests(unittest.TestCase):
    def check(self, row):
        expected = {k: annotation()[k] for k in ('paragraph_id', 'source_paragraph')}
        return review.check_row(row, expected, preparation.output_schemas()[1])

    def test_valid_and_corrupted_sources_claim_evidence_and_bindings(self):
        self.assertEqual(self.check(annotation()), [])
        for mode in ('source', 'sentence', 'evidence', 'binding', 'sequence', 'empty_claim'):
            row = annotation()
            if mode == 'source': row['source_paragraph'] += ' Extra.'
            if mode == 'sentence': row['source_sentences'][0]['exact_text'] = 'Changed.'
            if mode == 'evidence': row['source_claims'][0]['source_evidence'][0]['occurrence'] = 2
            if mode == 'binding': row['source_sentences'][0]['claim_ids'] = ['D2']
            if mode == 'sequence': row['source_claims'][1]['claim_id'] = 'D1'
            if mode == 'empty_claim': row['source_claims'][0]['proposition'] = ''
            with self.subTest(mode=mode): self.assertTrue(self.check(row))

    def test_boundary_coverage_adjacency_and_evidence_from_both_sides(self):
        for mode in ('missing', 'duplicate', 'adjacency', 'one_side'):
            row = annotation(); b = row['boundaries'][0]
            if mode == 'missing': row['boundaries'] = []
            if mode == 'duplicate': row['boundaries'].append(copy.deepcopy(b))
            if mode == 'adjacency': b['right_sentence_id'] = 'S1'
            if mode == 'one_side': b['source_evidence'].pop()
            with self.subTest(mode=mode): self.assertTrue(self.check(row))

    def test_verdict_candidate_and_preference_consistency(self):
        row = annotation(); row['boundaries'][0]['merge_candidate'] = None
        self.assertTrue(self.check(row))
        row = annotation(); row['boundaries'][0]['permissible'] = 'KEEP'
        self.assertTrue(self.check(row))
        row['boundaries'][0]['presentation_preference'] = 'KEEP'
        self.assertEqual(self.check(row), [])
        row['boundaries'][0]['permissible'] = 'UNRESOLVED'
        row['boundaries'][0]['presentation_preference'] = 'UNRESOLVED'
        row['boundaries'][0]['merge_candidate'] = None
        self.assertEqual(self.check(row), [])

    def test_surface_flags_never_certify_semantics_and_handle_abbreviations(self):
        result = review.candidate_diagnostics('Firms may invest.', 'Costs fall.', 'Firms invest because costs fall.')
        self.assertIn('new_explicit_causal_cue', result['risk_flags'])
        self.assertIn('guard_inventory_changed', result['risk_flags'])
        result = review.candidate_diagnostics('See note 2.', 'See Sec. V.', 'See note 2 and Sec. V.')
        self.assertNotIn('possible_multiple_sentences', result['risk_flags'])
        self.assertEqual(result['semantic_status'], 'unverified')
        result = review.candidate_diagnostics('Costs are 100.', 'Returns are 240.', 'Costs are 100. Returns are 200.')
        self.assertIn('number_inventory_changed', result['risk_flags'])
        self.assertIn('possible_multiple_sentences', result['risk_flags'])

    def fixture(self, root):
        packet = root/'packet'; (packet/'common').mkdir(parents=True); (packet/'responses').mkdir()
        (packet/'common/stage2_output.schema.json').write_text(json.dumps(preparation.output_schemas()[1]))
        for batch in (1, 2):
            folder = packet/('batches/batch_%02d' % batch); folder.mkdir(parents=True)
            row = annotation('p'+str(batch))
            (folder/'paragraphs.jsonl').write_text(json.dumps({k: row[k] for k in ('paragraph_id', 'source_paragraph')})+'\n')
            (packet/('responses/responses_%02d.jsonl' % batch)).write_text(json.dumps(row)+'\n')
        files = {str(p.relative_to(packet)): review.file_sha(p) for p in packet.rglob('*') if p.is_file() and 'responses' not in p.parts}
        (packet/'manifest.json').write_text(json.dumps({'files': files}))
        session = root/'session.json'; session.write_text(json.dumps({'visible_model_label': 'GPT-6',
            'requested_thinking_setting': 'High', 'effective_thinking_setting': 'uncertain High or Medium'}))
        return packet, session

    def test_snapshot_unknown_effort_and_no_semantic_or_training_certification(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); packet, session = self.fixture(root)
            result = review.audit(packet, root/'audit', session)
            self.assertEqual(result['mechanically_valid_records'], 2)
            self.assertEqual(result['effective_effort'], 'uncertain High or Medium')
            self.assertTrue(result['all_keep_permitted_under_raw_judgments'])
            self.assertFalse(result['semantic_fidelity_certified'])
            self.assertEqual(result['approved_training_rows'], 0)
            raw = root/'audit/raw_stage2/responses_01.jsonl'
            self.assertEqual(raw.read_bytes(), (packet/'responses/responses_01.jsonl').read_bytes())
            self.assertIsNone(review.read_rows(raw)[0]['provenance']['thinking_setting'])
            verify_files(root/'audit')
            with self.assertRaises(ValueError): review.audit(packet, root/'audit', session)

    def test_wrong_batch_missing_response_and_changed_packet_rejected(self):
        for mode in ('id', 'missing', 'packet'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); packet, session = self.fixture(root)
                path = packet/'responses/responses_01.jsonl'
                if mode == 'id':
                    row = annotation('wrong_batch'); path.write_text(json.dumps(row)+'\n')
                if mode == 'missing': path.unlink()
                if mode == 'packet': (packet/'common/stage2_output.schema.json').write_text('{}')
                with self.assertRaises(ValueError): review.audit(packet, root/'audit', session)
                self.assertFalse((root/'audit').exists())
