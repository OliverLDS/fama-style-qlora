from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.research.author_pair_expansion import prepare, select, shingles, validate_roles
from fama_style.research.author_curation import screen


class PairExpansionTests(unittest.TestCase):
    def row(self, pid='P1', text=None, **overrides):
        row = {'paragraph_id':pid, 'original_paragraph':text or
               'Trading costs affect investment choices. Investors retain their uncertain expectations.',
               'word_count':120, 'status':'annotation_candidate', 'hold_reasons':[],
               'quality_flags':[], 'pdf_page_start':3, 'duplicate_of':None}
        row.update(overrides)
        return row

    def test_holds_and_context_never_promoted(self):
        rows = [self.row('bad', hold_reasons=['corrupt_text']),
                self.row('context', quality_flags=['context_dependent_opening'])]
        chosen, _ = select(rows, set(), [])
        self.assertEqual(chosen, [])

    def test_prior_targets_and_overlay_excluded(self):
        original = self.row('seen')
        chosen, rejected = select([original, self.row('held')], {'held'},
                                  [shingles(original['original_paragraph'])])
        self.assertEqual(chosen, [])
        self.assertEqual(rejected, ['seen'])

    def test_cross_split_duplicate_and_shortfall(self):
        seen = []
        first, _ = select([self.row()], set(), seen)
        second, _ = select([self.row('P2')], set(), seen)
        self.assertEqual(len(first), 1)
        self.assertEqual(second, [])

    def test_roles_preserve_past_and_reserved(self):
        for roles, previous, sealed in [({'P':'development'}, {'P':'training'}, set()),
                                        ({'P':'training'}, {'P':'development'}, set()),
                                        ({'P':'training'}, {}, {'P'})]:
            with self.assertRaises(ValueError):
                validate_roles(roles, previous, sealed)
        validate_roles({'P':'training', 'N':'development'}, {'P':'training'}, {'T'})

    def test_supplementary_font_glyph_is_held(self):
        block = {'raw_extraction':'Investors retain an uncertain expectation \U0010fc03. Trading costs affect choices.',
                 'ocr_confidences':[]}
        lexicon = {'investors','retain','uncertain','expectation','trading','costs','affect','choices'}
        result = screen(block, lexicon, reject_supplementary_private_use=True)
        self.assertIn('unresolved_corrupt_glyph', result[3])
        self.assertNotIn('unresolved_corrupt_glyph', screen(block, lexicon)[3])

    def test_packet_preserves_original_and_skips_reserved_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def write(relative, value, raw=False):
                path = root/relative; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(value if raw else json.dumps(value))
                return path
            write('research_runs/human_prose_annotation_v1/package_v2/SPLIT_MANIFEST.json',
                  {'papers':[{'paper_id':'P1','split':'training'}, {'paper_id':'SEALED','split':'reserved_test'}]})
            write('data/fama_corpus_v1/split_manifest.json', {'paper_assignments':{}})
            write('research_runs/neutral_reconstruction_v1/package_v1/paper_role_commitments.private.json', [])
            write('docs/author_curation_v1/post_annotation_holds_v1.json', {'holds':[]})
            for relative in ['data/fama_corpus_v1/train.jsonl','data/fama_corpus_v1/validation.jsonl',
                             'data/teacher_pair_pilot_v2/training.private.jsonl',
                             'data/teacher_pair_pilot_v2/development.private.jsonl',
                             'research_runs/neutral_reconstruction_v1/package_v1/private_key/selection.private.jsonl']:
                write(relative, '', raw=True)
            write('data/fama_corpus_v1/test.jsonl', 'RESERVED BODY IS NOT JSON', raw=True)
            write('docs/human_prose_annotation_v1/ANNOTATION_PROMPT.md', 'Preserve all claims.', raw=True)
            catalogue = write('catalogue.json', {'taxonomy_version':'synthetic', 'taxonomy_sha256':'test',
                                               'canonical_items':[], 'status':'test_only'})
            row = self.row()
            row.update(paper_id='P1', original_sha256=hashlib.sha256(row['original_paragraph'].encode()).hexdigest(),
                       paper_pdf_sha256='synthetic_pdf', page_location={'pdf_page_start':3},
                       line_refs=[], normalization_events=[])
            write('data/author_style_corpus_v1/paragraphs.private.jsonl', json.dumps(row)+'\n', raw=True)
            plan = write('plan.json', {'papers':[{'paper_id':'P1','style_group':'synthetic','role':'training'}],
                                      'candidate_files':['data/author_style_corpus_v1/paragraphs.private.jsonl']})
            first, second = root/'first/packet', root/'second/packet'
            with patch('fama_style.research.author_pair_expansion.CATALOG', catalogue):
                summary = prepare(root, plan, first)
                prepare(root, plan, second)
            self.assertEqual(summary['training_pairs_created'], 0)
            self.assertEqual(summary['paragraphs'], 1)
            payload = json.loads((first/'batches/batch_01/paragraphs.jsonl').read_text())
            self.assertEqual(payload['text'], row['original_paragraph'])
            for file in first.rglob('*'):
                if file.is_file():
                    self.assertEqual(file.read_bytes(), (second/file.relative_to(first)).read_bytes())


if __name__ == '__main__':
    unittest.main()
