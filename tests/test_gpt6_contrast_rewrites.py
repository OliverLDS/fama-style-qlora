"""Meaning-aware contrast triage and genuinely brief-only reconstruction packets."""
import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('contrast_rewrites',ROOT/'scripts/data/prepare_gpt6_contrast_rewrites.py')
m=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class ContrastTests(unittest.TestCase):
    def test_close_copy_requires_regeneration(self):
        target='The researchers combine household accounts to estimate monthly dividends. They match each security to an external database and calculate its payment. Finally, they sum the payments over the whole portfolio.'
        source=target.replace('combine','pool').replace('Finally','Lastly')
        metrics=m.contrast(source,target)
        self.assertEqual(m.classify(metrics)[0],'regenerate_brief_first')
        self.assertGreater(metrics['ordinary_retention'],.85)

    def test_fixed_anchors_are_exempt_and_fresh_expression_can_be_retained(self):
        text='We calculate CRSP codes 1232 and 1272 with η_t and book-to-market ratios.'
        tokens=m.tokenize(text);mask=m.exempt_mask(text,tokens)
        flagged={r[1]:x for r,x in zip(tokens,mask)}
        self.assertTrue(flagged['CRSP']);self.assertTrue(flagged['1232']);self.assertTrue(flagged['η_t'])
        self.assertFalse(flagged['calculate'])
        metrics=m.contrast('A 5% fee could discourage transactions. New information may consequently take longer to affect prices.',
                           'Higher trading charges of 5% may reduce trading frequency and slow the incorporation of information into prices.')
        self.assertEqual(m.classify(metrics)[0],'retain_current')

    def test_sentence_splitting_does_not_hide_copied_clauses(self):
        metrics=dict(word_similarity=.72,ordinary_retention=.72,ordinary_token_count=60,
            ordinary_shared_fivegram_coverage=.48,longest_ordinary_words=12,
            sentence_layout_echo=False,ordinary_bag_reuse=.78)
        decision,reasons=m.classify(metrics)
        self.assertEqual(decision,'regenerate_brief_first')
        self.assertTrue(any('splitting sentences' in s for s in reasons))

    def test_reordering_same_inventory_is_not_proof_of_rewriting(self):
        metrics=dict(word_similarity=.55,ordinary_retention=.55,ordinary_token_count=60,
            ordinary_shared_fivegram_coverage=.35,longest_ordinary_words=10,
            sentence_layout_echo=False,ordinary_bag_reuse=.9)
        decision,reasons=m.classify(metrics)
        self.assertEqual(decision,'regenerate_brief_first')
        self.assertTrue(any('clause reordering' in s for s in reasons))
        self.assertEqual(m.classify(metrics,target_issue=True)[0],'target_cleanup_first')

    def test_briefs_do_not_have_to_preserve_stylistic_wording(self):
        expected=dict(id='P002',target_sha256='a'*64,target='The hopes were unrealistic.')
        row=dict(id='P002',target_sha256='a'*64,
            brief=dict(task='Explain why voluntary coverage was inadequate.',key_points=['Voluntary groups reached few people.'],anchors=[]),status='ready',notes=[])
        m.validate_brief(row,expected)
        for bad in [dict(row,id='P003'),dict(row,target_sha256='b'*64),dict(row,original_paragraph=expected['target'])]:
            with self.assertRaises(ValueError):m.validate_brief(bad,expected)
        disguised=copy.deepcopy(row);disguised['brief']['key_points']=[expected['target']]
        with self.assertRaisesRegex(ValueError,'disguised'):m.validate_brief(disguised,expected)


@unittest.skipUnless((m.OUT/'manifest.json').exists(),'private contrast packet not distributed')
class PrivateContrastTests(unittest.TestCase):
    def test_all_500_have_a_text_bound_decision_and_175_are_requested_once(self):
        m.check();rows=m.lines(m.OUT/'audit.private.jsonl');selected=m.lines(m.OUT/'selected_targets.private.jsonl')
        self.assertEqual(len(rows),500);self.assertEqual(len(selected),175)
        self.assertEqual({r['id'] for r in selected}&{'P250','P264','P291','P458','P484'},set())
        self.assertEqual(next(r for r in rows if r['id']=='P001')['decision'],'retain_current')
        self.assertTrue({'P002','P024'}<={r['id'] for r in selected})
        jobs=m.read(m.OUT/'stage_1_brief_batches/jobs.json');ids=[rid for j in jobs for rid in j['ids']]
        self.assertEqual(len(ids),len(set(ids)));self.assertEqual(set(ids),{r['id'] for r in selected})
        self.assertEqual([j['records'] for j in jobs],[30,30,30,30,30,25])
        parents=m.lines(m.PARENT/'candidates_0500.private.jsonl')
        for a,b in zip(rows,parents):
            self.assertEqual(a['source_sha256'],b['source_sha256']);self.assertEqual(a['target_sha256'],b['target_sha256'])

    def test_rebuild_is_identical_and_stage_2_forwards_only_actual_briefs(self):
        real=m.OUT
        try:
            with tempfile.TemporaryDirectory() as temp:
                m.OUT=Path(temp)/'packet'
                with contextlib.redirect_stdout(io.StringIO()):m.prepare()
                self.assertEqual((m.OUT/'manifest.json').read_bytes(),(real/'manifest.json').read_bytes())
                with self.assertRaisesRegex(ValueError,'overwrite'):m.prepare()
                selected={r['id']:r for r in m.lines(m.OUT/'selected_targets.private.jsonl')}
                for job in m.read(m.OUT/'stage_1_brief_batches/jobs.json'):
                    # Synthetic fixtures, never labeled real generated/qualified data.
                    rs=[dict(id=rid,target_sha256=selected[rid]['target_sha256'],
                        brief=dict(task='Synthetic test task.',key_points=['A synthetic economic fact.'],anchors=[]),status='ready',notes=[]) for rid in job['ids']]
                    (m.OUT/'responses/stage_1_briefs'/job['response']).write_text(m.jsonl(rs))
                with contextlib.redirect_stdout(io.StringIO()):m.prepare_reconstruction()
                folder=m.OUT/'stage_2_reconstruction_batches';all_ids=[]
                for job in m.read(folder/'jobs.json'):
                    inputs=m.lines(folder/f"inputs_{job['batch']:02d}.jsonl")
                    for row in inputs:
                        self.assertEqual(set(row),{'id','target_sha256','brief','brief_status','brief_notes'})
                        self.assertEqual(row['brief']['key_points'],['A synthetic economic fact.'])
                        all_ids.append(row['id'])
                    text=(folder/job['attachment']).read_text()
                    for rid in job['ids']:self.assertNotIn(selected[rid]['target'],text)
                self.assertEqual(set(all_ids),set(selected))
                with self.assertRaisesRegex(ValueError,'overwrite'):m.prepare_reconstruction()
        finally:m.OUT=real


if __name__=='__main__':unittest.main()
