"""Identity, repair replay and immutable-target checks for fresh reconstructions."""
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('output_review',ROOT/'scripts/data/review_gpt6_contrast_outputs.py')
m=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class RewriteReviewTests(unittest.TestCase):
    def test_identity_and_nonempty_prose_are_required_but_minor_shift_is_not_a_schema_failure(self):
        expected=dict(id='P002',target_sha256='a'*64)
        row=dict(expected,plain_english='A newly composed explanation.',rewrite_status='rewritten',review_notes=[])
        m.validate_response(row,expected)
        for changed in [dict(row,id='P003'),dict(row,target_sha256='b'*64),dict(row,plain_english=' '),dict(row,extra='reference prose')]:
            with self.assertRaises(ValueError):m.validate_response(changed,expected)
        with self.assertRaises(ValueError):m.validate_response(dict(row,rewrite_status='source_problem'),expected)
        m.validate_response(dict(row,rewrite_status='source_problem',review_notes=['Inherited ambiguity.']),expected)

    def test_local_patches_reject_stale_anchors_and_full_replacement(self):
        source='The return was compared with a benchmark. The rest of the paragraph stays fresh.'
        patch=dict(before='a benchmark',after='the market index',reason='Clarify comparison.')
        self.assertEqual(m.apply_patches(source,[patch]),source.replace('a benchmark','the market index'))
        for bad in [dict(patch,before='missing'),dict(patch,before=source),dict(patch,reason='')]:
            with self.assertRaises(ValueError):m.apply_patches(source,[bad])


@unittest.skipUnless((m.OUT/'manifest.json').exists(),'private rewrite review is not distributed')
class PrivateRewriteReviewTests(unittest.TestCase):
    def test_every_reconstruction_improves_contrast_without_automatic_semantic_promotion(self):
        self.assertEqual(m.check(),dict(rewrites=175,candidates=500,training_ready=False,raw_sources_unchanged=True))
        summary=m.m.read(m.OUT/'summary.json')
        self.assertEqual(summary['decision_counts'],dict(accept=163,repair=10,source_concern=2))
        self.assertEqual(summary['lower_word_similarity'],175)
        self.assertEqual(summary['new_inputs_staged'],173)
        self.assertEqual(summary['source_versions_discarded'],0)
        self.assertEqual(summary['still_flagged_by_contrast_rules'],0)
        self.assertFalse(summary['training_ready_release'])

    def test_sources_change_only_for_173_admitted_replacements_and_targets_never_change(self):
        parent=m.m.lines(m.m.PARENT/'candidates_0500.private.jsonl')
        current=m.m.lines(m.OUT/'candidates_0500.private.jsonl')
        changed=[]
        for i,(a,b) in enumerate(zip(parent,current),1):
            self.assertEqual(a['target'],b['target']);self.assertEqual(a['target_sha256'],b['target_sha256'])
            self.assertEqual(a['parent_target_id'],b['parent_target_id']);self.assertFalse(b['training_ready'])
            if a['source']!=b['source']:changed.append(b['id'])
        self.assertEqual(len(changed),173)
        self.assertNotIn('P134',changed);self.assertNotIn('P139',changed)
        self.assertFalse(any('hope' in b['source'].casefold() for b in current if b['id']=='P002'))
        self.assertTrue((m.OUT/'candidates_0500.private.jsonl').read_bytes().startswith((m.OUT/'candidates_0200.private.jsonl').read_bytes()))
        with self.assertRaisesRegex(ValueError,'overwrite'):m.prepare()


if __name__=='__main__':unittest.main()
