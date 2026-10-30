"""Mechanical corruption, abstention and source-only packet isolation checks."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fama_style.research import relation_pilot as preparation
from fama_style.research import relation_pilot_review as review
from fama_style.research.author_control_replication import verify_files


def annotation(pid='p'):
    text = 'Firms may invest if costs fall. Investors provide funds.'
    claims = []
    for i, quote in enumerate(text.split('. ') , 1):
        claims.append({'claim_id': 'C'+str(i), 'proposition': quote,
            **{k: [] for k in ('actors_and_roles', 'conditions', 'quantities', 'negation', 'modality', 'attribution')},
            'original_evidence': [{'quote': quote, 'occurrence': 1}],
            'neutral_evidence': [{'quote': quote, 'occurrence': 1}], 'alignment': 'preserved'})
    sentences = lambda prefix: [{'sentence_id': prefix+str(i), 'exact_text': s,
        'claim_ids': ['C'+str(i)], 'reference_only_claim_ids': []} for i, s in enumerate(
            ['Firms may invest if costs fall.', 'Investors provide funds.'], 1)]
    return {'annotation_version': 'relation_pilot_v1_stage1', 'paragraph_id': pid,
        'original_paragraph': text, 'plain_english': text, 'rewrite_status': 'already_clear',
        'rewrite_notes': 'Already clear.', 'claims': claims,
        'original_sentences': sentences('O'), 'neutral_sentences': sentences('S'),
        'needs_external_context': False, 'needs_review': False, 'review_notes': [],
        'provenance': {'visible_model_label': None, 'thinking_setting': None,
                       'paired_context': True, 'independently_verified': False}}


class RelationPilotReviewTests(unittest.TestCase):
    def test_corrupt_original_evidence_binding_and_sentence_coverage_rejected(self):
        good = annotation(); schema = preparation.output_schemas()[0]
        self.assertEqual(review.check_row(good, good, schema), [])
        for mode in ('original', 'evidence', 'binding', 'coverage', 'claim_sequence', 'back_reference'):
            row = copy.deepcopy(good)
            if mode == 'original': row['original_paragraph'] += ' Unsupported.'
            if mode == 'evidence': row['claims'][0]['neutral_evidence'][0]['occurrence'] = 2
            if mode == 'binding': row['neutral_sentences'][0]['claim_ids'] = ['C99']
            if mode == 'coverage': row['neutral_sentences'].pop()
            if mode == 'claim_sequence': row['claims'][1]['claim_id'] = 'C1'
            if mode == 'back_reference': row['neutral_sentences'][0]['reference_only_claim_ids'] = ['C1']
            with self.subTest(mode=mode): self.assertTrue(review.check_row(row, good, schema))

    def test_pending_changed_missing_and_mechanical_failures_not_exported(self):
        for mode, expected in [('pending', 'pending_rewrite'), ('changed', 'nonpreserved_claim_annotation'),
                               ('empty', 'no_neutral_text'), ('error', 'mechanical_errors')]:
            row = annotation(); errors = []
            if mode == 'pending': row['rewrite_status'] = 'pending_review'
            if mode == 'changed': row['claims'][0]['alignment'] = 'changed'
            if mode == 'empty': row['plain_english'] = None
            if mode == 'error': errors = [{'code': 'original_changed'}]
            self.assertEqual(review.eligibility(row, errors), expected)
        row = annotation(); row['needs_review'] = row['needs_external_context'] = True
        self.assertTrue(review.eligibility(row, []).startswith('eligible_'))

    def fixture(self, root):
        packet = root / 'packet'; (packet / 'common').mkdir(parents=True)
        (packet / 'responses/stage1').mkdir(parents=True)
        for i, schema in enumerate(preparation.output_schemas(), 1):
            (packet / ('common/stage%d_output.schema.json' % i)).write_text(json.dumps(schema))
        (packet / 'common/STAGE2_PROMPT.md').write_text(preparation.STAGE2_PROMPT)
        (packet / 'SESSION_NOTES.md').write_text('Original blank form.')
        selection = []
        for batch in range(1, 9):
            folder = packet / ('stage1/batch_%02d' % batch); folder.mkdir(parents=True)
            rows = [annotation('p%d_%d' % (batch, i)) for i in range(5)]
            if batch == 1: rows[0]['rewrite_status'] = 'pending_review'
            source = [{k: r[k] for k in ('paragraph_id', 'original_paragraph')} for r in rows]
            (folder / 'paragraphs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in source))
            (packet / ('responses/stage1/responses_%02d.jsonl' % batch)).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            selection.extend({'paragraph_id': r['paragraph_id'], 'paper_id': str((batch-1)//2)} for r in rows)
        (packet / 'PRIVATE_SELECTION.json').write_text(json.dumps(selection))
        # Initial packet manifest excludes user-supplied responses/session notes.
        files = {str(p.relative_to(packet)): review.file_sha(p) for p in packet.rglob('*')
                 if p.is_file() and 'responses' not in p.parts and p.name != 'SESSION_NOTES.md'}
        (packet / 'manifest.json').write_text(json.dumps({'files': files}))
        session = root / 'session.json'
        session.write_text(json.dumps({'visible_model_label': 'GPT-6', 'thinking_setting': 'High thinking effort'}))
        return packet, session

    def test_end_to_end_snapshot_pending_hold_and_reference_exclusion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); packet, session = self.fixture(root)
            summary = review.audit(packet, root/'audit', session)
            self.assertEqual(summary['mechanically_valid_records'], 40)
            raw = Path('raw_stage1/responses_01.jsonl')
            self.assertEqual((root/'audit'/raw).read_bytes(), (packet/'responses/stage1/responses_01.jsonl').read_bytes())
            result = review.prepare_stage2(packet, root/'audit', root/'stage2')
            self.assertEqual((result['paragraphs'], result['held_back']), (39, 1))
            exported = []
            for f in sorted((root/'stage2/batches').glob('*/paragraphs.jsonl')):
                rows = review.read_rows(f); self.assertTrue(4 <= len(rows) <= 5)
                self.assertTrue(all(set(r) == {'paragraph_id', 'source_paragraph'} for r in rows))
                exported.extend(r['paragraph_id'] for r in rows)
            self.assertEqual(len(set(exported)), 39); self.assertNotIn('p1_0', exported)
            verify_files(root/'audit'); verify_files(root/'stage2')
            with self.assertRaises((ValueError, FileExistsError)):
                review.prepare_stage2(packet, root/'audit', root/'stage2')
            (root/'audit/annotations.private.jsonl').write_text('tampered')
            with self.assertRaises(ValueError): review.prepare_stage2(packet, root/'audit', root/'bad')

    def test_duplicate_or_wrong_batch_ids_stop_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); packet, session = self.fixture(root)
            path = packet/'responses/stage1/responses_01.jsonl'
            rows = review.read_rows(path); rows[1] = rows[0]
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError, 'wrong_batch_ids'):
                review.audit(packet, root/'audit', session)
            self.assertFalse((root/'audit').exists())
