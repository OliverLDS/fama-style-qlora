import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('prompt_input_review',ROOT/'scripts/data/review_purpose_500_prompt_input.py')
review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)


class PromptInputReviewTests(unittest.TestCase):
    def test_wrong_notebook_version_rejected_before_reading_prose(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            review.runtime.write_json(p/'download_manifest.json',dict(slug='oliverlds/purpose-500-prompt-input-4b',
                version=2,version_explicitly_verified_via_SDK=True,files={}))
            with self.assertRaisesRegex(ValueError,'exact_version_required'):review.verify(p)

    def test_tampered_download_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'raw').mkdir();file=p/'raw/log.txt';file.write_text('original')
            review.runtime.write_json(p/'download_manifest.json',dict(slug='oliverlds/purpose-500-prompt-input-4b',
                version=1,version_explicitly_verified_via_SDK=True,files={'log.txt':review.runtime.file_sha(file)}))
            file.write_text('changed')
            with self.assertRaisesRegex(ValueError,'download_changed'):review.verify(p)

    def test_refusal_does_not_become_a_success_from_low_copying(self):
        source='A substantive source passage discusses economic observations and their implications.'
        text='Please provide the original passage.'
        r=dict(empty=False,output_limited=False,output=text,
               source_metrics=review.metrics(source,text),reference_metrics=review.metrics(source,text))
        a=review.aggregate([r])
        self.assertEqual(a['exact_word_copies'],0)
        self.assertNotIn('quality_pass',a)
        self.assertNotIn('style_improved',a)

    def test_capped_repetition_is_retained_and_counted(self):
        text='This is an exact repeated long sentence about a changing price level. '
        r=dict(empty=False,output_limited=True,output=text*3,
               source_metrics=review.metrics('A short source.',text*3),reference_metrics=review.metrics('Reference.',text*3))
        a=review.aggregate([r]);self.assertEqual(a['output_limited'],1)
        self.assertEqual(a['repeated_long_sentence_outputs'],1)


if __name__=='__main__':unittest.main()
