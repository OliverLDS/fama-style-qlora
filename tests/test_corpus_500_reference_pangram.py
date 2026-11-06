"""Human controls retain unexpected results rather than forcing a human label."""
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('reference_results500',ROOT/'scripts/data/import_corpus_500_reference_pangram.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)


class Reference500Pangram(unittest.TestCase):
    def data(self):
        refs=[];pairs=[]
        for i in range(8):
            refs.append(dict(example_id=str(i),paper_id='paper',style_group='group',provider_label='Human Written',entirely_human=True,
                human_written_document_percent=100.,ai_generated_document_percent=.1,ai_assisted_document_percent=.1,text_sha256='reference'+str(i)))
            pairs.append(dict(example_id=str(i),paper_id='paper',style_group='group',conditions={'epoch_1':dict(human_written_document_percent=.1)}))
        return refs,dict(pairs=pairs,conditions={},score_semantics='Displayed shares, not AI probabilities.')

    def test_reference_outcome_is_recorded_without_changing_model_findings(self):
        refs,full=self.data();refs[0].update(provider_label='AI Assisted',entirely_human=False,human_written_document_percent=.1)
        result=m.combine(refs,full)
        self.assertEqual(result['human_reference_entirely_human'],7)
        self.assertEqual(result['human_reference_native_labels']['AI Assisted'],1)
        self.assertEqual(result['prior_conditions'],full['conditions'])
        self.assertEqual(result['pairs'][0]['conditions'],full['pairs'][0]['conditions'])

    def test_missing_or_mismatched_reference_does_not_enter_panel(self):
        refs,full=self.data()
        with self.assertRaisesRegex(ValueError,'eight_matched'):m.combine(refs[:-1],full)
        refs[0]['paper_id']='wrong'
        with self.assertRaisesRegex(ValueError,'lineage_changed'):m.combine(refs,full)


if __name__=='__main__':unittest.main()
