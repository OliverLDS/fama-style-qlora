"""Triplet comparisons retain mixed labels, display floors and negative results."""
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('pangram500',ROOT/'scripts/data/import_corpus_500_pangram.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)


class Corpus500Pangram(unittest.TestCase):
    def observations(self):
        result=[]
        for rid,group,values in [('one','Fama_coauthors',[.1,.1,100.]),('two','Shiller',[67.,.1,.1])]:
            for c,h in zip(['input','base','epoch_1'],values):
                result.append(dict(example_id=rid,condition=c,style_group=group,paper_id='paper-'+rid,
                    provider_label='Human Written' if h==100 else 'AI Assisted',entirely_human=h==100,
                    human_written_document_percent=h,ai_generated_document_percent=.1,
                    ai_assisted_document_percent=.1 if h==100 else 100.,provider_word_count=100,
                    material_output_issue_identified=None if c=='input' else rid=='two',source_word_sequence_overlap=.5))
        return result

    def test_benefit_and_deterioration_are_not_collapsed_into_success(self):
        report=m.compare(self.observations())
        self.assertEqual(report['higher_human_share_than_input'],1)
        self.assertEqual(report['lower_human_share_than_input'],1)
        self.assertEqual(report['entirely_human_counts'],dict(input=0,base=0,epoch_1=1))
        self.assertAlmostEqual(report['pairs'][1]['change_human_share_vs_input'],-66.9)
        self.assertEqual(report['pairs'][0]['conditions']['input']['human_written_document_percent'],.1)
        self.assertIn('not document AI probabilities',report['score_semantics'])

    def test_duplicate_or_missing_reports_are_not_silently_averaged(self):
        rows=self.observations()
        with self.assertRaisesRegex(ValueError,'repeat_report'):m.compare(rows+[rows[0]])
        with self.assertRaisesRegex(ValueError,'complete_triplets'):m.compare(rows[:-1])


if __name__=='__main__':unittest.main()
