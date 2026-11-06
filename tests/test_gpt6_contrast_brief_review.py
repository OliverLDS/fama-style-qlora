"""Verify local brief edits and the actual brief-only Stage 2 export."""
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('brief_review',ROOT/'scripts/data/review_gpt6_contrast_briefs.py')
m=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class BriefEditTests(unittest.TestCase):
    def row(self):
        return dict(id='P002',target_sha256='a'*64,brief=dict(task='Explain an economic result.',key_points=['A neutral fact.'],anchors=[]),status='ready',notes=[])

    def test_edits_replay_without_mutating_raw_or_identity(self):
        row=self.row();before=copy.deepcopy(row)
        edits=[dict(path=['brief','key_points',1],operation='append',before=None,after='A supporting fact.',reason='Retain an omitted mechanism.')]
        final=m.apply_edits(row,edits)
        self.assertEqual(row,before)
        self.assertEqual(final['id'],row['id']);self.assertEqual(final['target_sha256'],row['target_sha256'])
        self.assertEqual(final['brief']['key_points'],['A neutral fact.','A supporting fact.'])

    def test_wrong_before_text_and_identity_edits_are_rejected(self):
        row=self.row()
        for edit in [dict(path=['status'],before='source_problem',after='ready',reason='Not bound.'),
                     dict(path=['id'],before='P002',after='P003',reason='Change identity.'),
                     dict(path=['brief','key_points',4],operation='append',before=None,after='Missing.',reason='Skip indices.')]:
            with self.subTest(edit=edit),self.assertRaises(ValueError):m.apply_edits(row,[edit])

    def test_response_hash_tracks_content_not_json_key_order(self):
        row=self.row()
        self.assertEqual(m.row_hash(row),m.row_hash(dict(reversed(list(row.items())))))
        self.assertNotEqual(m.row_hash(row),m.row_hash(dict(row,status='source_problem')))


@unittest.skipUnless((m.AUDIT/'manifest.json').exists(),'private Stage 2 packet not distributed')
class Stage2BriefReviewTests(unittest.TestCase):
    def test_actual_export_keeps_175_items_and_carries_only_two_source_concerns(self):
        self.assertEqual(m.check(),dict(records=175,batches=6,original_prose_fields_forwarded=0,source_files_modified=0))
        reviewed=m.m.lines(m.AUDIT/'reviewed_briefs.private.jsonl')
        self.assertEqual({r['id'] for r in reviewed if r['effective_response']['status']=='source_problem'},{'P134','P139'})
        self.assertEqual(next(r for r in reviewed if r['id']=='P282')['effective_response']['status'],'ready')
        p2=next(r for r in reviewed if r['id']=='P002')['effective_response']['brief']
        self.assertNotIn('hope',' '.join([p2['task']]+p2['key_points']+p2['anchors']).lower())

    def test_attachments_do_not_contain_the_original_or_previous_rewrite(self):
        selected={r['id']:r for r in m.m.lines(m.m.OUT/'selected_targets.private.jsonl')}
        for job in m.m.read(m.STAGE2/'jobs.json'):
            text=(m.STAGE2/job['attachment']).read_text()
            for rid in job['ids']:
                self.assertNotIn(selected[rid]['target'],text)
                self.assertNotIn(selected[rid]['source'],text)
        with self.assertRaisesRegex(ValueError,'overwrite'):m.prepare()


if __name__=='__main__':unittest.main()
