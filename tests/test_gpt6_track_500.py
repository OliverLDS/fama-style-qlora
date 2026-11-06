"""Prevent teacher mislabeling and double counting of reviewed pair versions."""
import contextlib
import importlib.util
import io
import itertools
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('gpt6_packet', ROOT / 'scripts/data/prepare_gpt6_track_500.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class GPT6ReuseTests(unittest.TestCase):
    def test_wholly_codex_input_is_not_relabeled_gpt6(self):
        row = dict(provenance=dict(new_neutral_input_author='Codex', user_reported_generation='GPT-6 High'),
                   target_sha256='human', source='Codex replacement')
        self.assertEqual(m.classify(row, {}), 'missing_GPT6_wholly_Codex_input')

    def test_reviewed_teacher_reuse_requires_exact_source_and_provenance(self):
        row = dict(provenance={}, target_sha256='human', source='Reviewed source')
        old = dict(source='Reviewed source', annotation_origin='paired_GPT6_not_independent')
        self.assertEqual(m.classify(row, {'human': old}), 'reuse_prior_known_GPT6')
        for bad in [dict(old, source='Different teacher input'), dict(old, annotation_origin='unknown')]:
            with self.assertRaises(ValueError):
                m.classify(row, {'human': bad})

    def test_historical_model_confirmation_does_not_make_unknown_teacher_reusable(self):
        row = dict(provenance=dict(parent_file=m.POOLS['early_fama']), target_sha256='human')
        self.assertEqual(m.classify(row, {}), 'reuse_user_confirmed_historical_GPT6')
        row['provenance'] = dict(generator='unknown provider')
        with self.assertRaises(ValueError):
            m.classify(row, {})

    def test_repaired_target_remains_linked_to_its_original_parent(self):
        pools = dict(early_fama=[], early_20_batches=[dict(example_id='old', target='Same  paragraph.')],
                     reviewed_teacher=[dict(example_id='reviewed', target='same paragraph.')],
                     later_54_batches=[dict(paragraph_id='parent', annotation=dict(original_paragraph='An OCR error.', plain_english='Neutral.'))],
                     reviewed_salvage=[dict(id='child', parent_id='parent', original_paragraph='A corrected paragraph.', plain_english='Neutral.')])
        exact, lineage, _ = m.overlap_audit(pools)
        self.assertEqual(next(x['shared_normalized_targets'] for x in exact if x['left'] == 'early_20_batches' and x['right'] == 'reviewed_teacher'), 1)
        repaired = next(x for x in lineage if x['relationship'] == 'reviewed_or_repaired_version')
        self.assertEqual(repaired['parent_id'], 'parent')
        self.assertFalse(repaired['target_unchanged_normalized'])
        self.assertTrue(repaired['source_unchanged'])

    def test_fuzzy_screen_catches_small_target_edits(self):
        a = 'Higher trading costs can reduce trading frequency when investors face binding constraints.'
        b = 'Higher trading costs may reduce trading frequency when investors face binding constraints.'
        self.assertEqual(len(m.fuzzy_flags([dict(id='a', target=a)], [dict(id='b', target=b)])), 1)


@unittest.skipUnless((m.DEFAULT_OUT / 'manifest.json').exists(), 'private preparation is not distributed')
class PreparedGPT6PacketTests(unittest.TestCase):
    def test_private_inventory_and_target_bindings(self):
        self.assertEqual(m.validate(), dict(reused=273, missing=227, batches=8))
        parent = m.lines(ROOT / m.MASTER)
        index = m.lines(m.DEFAULT_OUT / 'target_lineage.private.jsonl')
        self.assertEqual(len({r['parent_target_id'] for r in index}), 500)
        self.assertEqual(len({r['target_sha256'] for r in index}), 500)
        self.assertEqual([r['target_sha256'] for r in parent], [r['target_sha256'] for r in index])
        for job in m.read(m.DEFAULT_OUT / 'jobs.json'):
            for row in m.lines(m.DEFAULT_OUT / 'manual_batches_v1' / f"inputs_{job['batch']:02d}.jsonl"):
                self.assertEqual(set(row), {'id', 'target_sha256', 'original_paragraph'})
                self.assertEqual(m.sha(row['original_paragraph']), row['target_sha256'])
                self.assertEqual(row['original_paragraph'], parent[int(row['id'][1:])-1]['target'])
        papers = m.read(m.DEFAULT_OUT / 'paper_identity_audit.json')
        self.assertEqual(papers['canonical_pdf_count'], 48)
        self.assertEqual(len(papers['alias_groups']), 5)
        self.assertEqual(papers['training_development_pdf_overlap'], 0)

    def test_rebuild_is_deterministic_and_overwrite_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'packet'
            with contextlib.redirect_stdout(io.StringIO()):
                m.prepare(out)
            self.assertEqual((out / 'manifest.json').read_bytes(), (m.DEFAULT_OUT / 'manifest.json').read_bytes())
            with self.assertRaisesRegex(ValueError, 'refuse_overwrite'):
                m.prepare(out)
            with (out / 'manual_batches_v1/batch_01.md').open('a') as handle:
                handle.write('Changed instructions')
            with self.assertRaisesRegex(ValueError, 'packet_changed'):
                m.validate(out)

    def test_same_pdf_aliases_do_not_hide_near_duplicate_selected_targets(self):
        groups = {}
        for entry, row in zip(m.lines(m.DEFAULT_OUT / 'target_lineage.private.jsonl'), m.lines(ROOT / m.MASTER)):
            groups.setdefault(entry['canonical_pdf_sha256'], []).append(row)
        count = 0
        for rows in groups.values():
            for a, b in itertools.combinations(rows, 2):
                count += 1
                self.assertEqual(m.fuzzy_flags([a], [b]), [])
        self.assertEqual(count, 3305)


if __name__ == '__main__':
    unittest.main()
