import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


def load(name,file):
    s=importlib.util.spec_from_file_location(name,ROOT/file)
    module=importlib.util.module_from_spec(s);s.loader.exec_module(module);return module


review=load('purpose_result','scripts/data/review_gpt6_purpose_500_result.py')
audit=load('purpose_behavior','scripts/data/audit_gpt6_purpose_500_behavior.py')


class PurposeResultTests(unittest.TestCase):
    def test_wrong_version_rejected_before_reports(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)
            review.write(p/'download_manifest.json',dict(slug='oliverlds/gpt6-purpose-500-4b',version=2,
                version_explicitly_verified_via_SDK=True,files={}))
            with self.assertRaisesRegex(ValueError,'exact_version_required'):review.verify(p,p/'unused')

    def test_download_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'raw').mkdir();f=p/'raw/log.txt';f.write_text('Original log')
            review.write(p/'download_manifest.json',dict(slug='oliverlds/gpt6-purpose-500-4b',version=1,
                version_explicitly_verified_via_SDK=True,files={'log.txt':review.sha(f)}))
            f.write_text('Tampered log')
            with self.assertRaisesRegex(ValueError,'download_changed'):review.verify(p,p/'unused')

    def test_copy_counts_do_not_become_semantic_or_style_verdicts(self):
        r=dict(empty=False,output_limited=False,source_metrics=dict(word_sequence_overlap=1.,longest_shared_word_run=20,word_token_copy=True),
               reference_metrics=dict(word_sequence_overlap=.9))
        a=review.aggregate([r])
        self.assertEqual(a['exact_word_copies'],1)
        self.assertNotIn('material_claim_issues_identified',a)

    def test_repetition_proxy_distinguishes_sentence_repetition(self):
        sentence='The same sentence is repeated without providing any additional information.'
        self.assertEqual(audit.repetition(sentence+' '+sentence),1)
        self.assertEqual(audit.repetition(sentence+' A different sentence supplies a separate argument about prices.'),0)


if __name__=='__main__':unittest.main()
