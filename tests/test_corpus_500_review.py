"""Matched detector controls preserve full texts and retain unfavorable outputs."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('review500',ROOT/'scripts/data/review_corpus_500_result.py')
review=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(review)


class Corpus500Review(unittest.TestCase):
    def examples(self):
        return [dict(example_id=rid,snapshot=s,source=('Source '+rid+' ')*35,reference='Human '+rid,
                     content_id=rid,input_condition='gemini_input',output=('Output <sub>t</sub> '+rid+' ')*25,
                     assessment=dict(material_claim_change=s=='epoch_1'))
                for rid in ['one','two'] for s in ['base','epoch_1']]

    def test_input_base_and_trained_are_kept_even_with_material_errors(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)/'scans';summary=review.scan_packet(self.examples(),['one','two'],out)
            items=review.rows(out/'PRIVATE_KEY.jsonl')
            self.assertEqual(summary['texts'],6)
            self.assertEqual({r['condition'] for r in items},{'input','base','epoch_1'})
            for r in items:
                self.assertEqual((out/'texts'/(r['scan_id']+'.txt')).read_text(),r['text'])
                self.assertEqual(review.runtime.sha(r['text']),r['text_sha256'])
            self.assertIn('\\u003csub\\u003e',(out/'scan_round_1.html').read_text())
            with self.assertRaises(ValueError):review.scan_packet(self.examples(),['one','two'],out)

    def test_changed_input_pair_cannot_be_scanned_as_matched(self):
        items=self.examples();items[1]['source']='Different input.'
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaisesRegex(ValueError,'matched_Gemini'):
                review.scan_packet(items,['one','two'],Path(t)/'scans')

    def test_small_style_notes_do_not_automatically_become_material_errors(self):
        row=dict(empty=False,output_limited=False,source_metrics=dict(word_sequence_overlap=.5,longest_shared_word_run=5,word_token_copy=False),
                 reference_metrics=dict(word_sequence_overlap=.4),assessment=dict(material_claim_change=False,minor_notes=['Slight evaluative tone change.'],readable=True))
        result=review.aggregate([row])
        self.assertEqual(result['material_claim_issues_identified'],0)
        self.assertEqual(result['outputs_with_minor_shift_notes'],1)

    def test_tampered_download_is_rejected_before_loading_models_or_reports(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'raw').mkdir();f=p/'raw/evaluation.json';f.write_text('{}')
            review.write(p/'download_manifest.json',dict(slug='oliverlds/corpus-500-pilot-4b',version=1,version_explicitly_verified_via_SDK=True,files={'evaluation.json':review.sha(f)}))
            f.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'download_changed'):
                review.verify(p,p/'unused_packet')


if __name__=='__main__':unittest.main()
