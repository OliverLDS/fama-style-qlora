"""Require the full prespecified panel and retain detector floors/native labels."""
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('full500',ROOT/'scripts/data/import_corpus_500_full_pangram.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)


class Full500Pangram(unittest.TestCase):
    def rows(self):
        ids=[f'case{i}' for i in range(8)];rows=[]
        # One conversion, one already-human input retained, four deteriorations,
        # and two unchanged AI-related inputs: an aggregate can hide the pattern.
        inputs=[.1,100.,67.,.1,1.,20.,4.,.1]
        outputs=[100.,100.,.1,.1,.1,.1,.1,.1]
        for i,rid in enumerate(ids):
            for c,h in [('input',inputs[i]),('base',.1),('epoch_1',outputs[i])]:
                rows.append(dict(example_id=rid,condition=c,style_group=f'group{i//2}',paper_id=f'paper{i//2}',
                    provider_label='Human Written' if h==100 else 'AI Assisted',entirely_human=h==100,
                    human_written_document_percent=h,ai_generated_document_percent=.1,ai_assisted_document_percent=.1 if h==100 else 100.,
                    provider_word_count=100,material_output_issue_identified=None if c=='input' else False,
                    source_word_sequence_overlap=1. if c=='input' else .5,provider_version='4.1',report_date='2026-11-05',binding_method='raw_whitespace_exact'))
        return rows,ids

    def test_retained_human_input_is_not_counted_as_new_conversion(self):
        rows,ids=self.rows();r=m.compare(rows,ids)
        self.assertEqual(r['conditions']['epoch_1']['entirely_human'],2)
        self.assertEqual(r['AI_related_inputs_available'],7)
        self.assertEqual(r['AI_related_inputs_becoming_entirely_human'],1)
        self.assertEqual(r['entirely_human_inputs_staying_entirely_human'],1)
        self.assertEqual((r['higher_human_share_than_input'],r['lower_human_share_than_input'],r['unchanged_human_share_vs_input']),(1,4,3))

    def test_missing_duplicate_and_unplanned_cases_are_rejected(self):
        rows,ids=self.rows()
        for changed in [rows[:-1],rows+[rows[0]],[dict(x,example_id='unexpected') if x['example_id']=='case7' else x for x in rows]]:
            with self.assertRaises(ValueError):m.compare(changed,ids)

    def test_new_native_summaries_keep_exact_displayed_values(self):
        for summary,label in m.EXTRA_SUMMARIES.items():
            text='AI Detection Report\nNov 5, 2026 132 Words Pangram 4.1\n'+summary+'\n47 %\nof the document\n52 %\nof the document\n1 %\nof the document\n'
            result=m.parse_page(text)
            self.assertEqual(result['provider_summary'],summary)
            self.assertEqual(result['provider_label'],label)
            self.assertEqual([result[k] for k in ['ai_generated_document_percent','ai_assisted_document_percent','human_written_document_percent']],[47.,52.,1.])
            self.assertFalse(result['percentages_rescaled']);self.assertFalse(result['entirely_human'])
        with self.assertRaises(ValueError):m.parse_page(text.replace(summary,'We believe something unknown.'))


if __name__=='__main__':unittest.main()
