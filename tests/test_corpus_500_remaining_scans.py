"""Keep the unscanned frozen cases, including unfavorable outputs, unchanged."""
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('remaining500',ROOT/'scripts/data/prepare_corpus_500_remaining_scans.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)


class Remaining500Scans(unittest.TestCase):
    def inputs(self):
        ids=[f'case{i}' for i in range(8)]
        config=dict(detector_case_ids=ids,detector_priority_ids=ids[:2])
        old=[dict(example_id=rid,condition=c) for rid in ids[:2] for c in ['input','base','epoch_1']]
        outputs=[]
        for rid in ids:
            for snap in ['base','epoch_1']:
                text=('Base ' if snap=='base' else 'Trained ')*70
                outputs.append(dict(example_id=rid,snapshot=snap,content_id=rid,input_condition='gemini_input',
                    source='Neutral '*70,reference='Human reference.',style_group='group',paper_id='paper',
                    output=text,output_sha256=m.r.runtime.sha(text),assessment=dict(material_claim_change=True)))
        return config,outputs,old

    def test_exact_remainder_and_full_triplets_are_not_quality_filtered(self):
        config,outputs,old=self.inputs();cards=m.select(config,outputs,old)
        self.assertEqual(len(cards),18)
        self.assertEqual({x['example_id'] for x in cards},set(config['detector_case_ids'][2:]))
        self.assertEqual({x['condition'] for x in cards},{'input','base','epoch_1'})
        for c in cards:self.assertEqual(m.r.runtime.sha(c['text']),c['text_sha256'])
        self.assertEqual([sum(x['round']==n for x in cards) for n in [2,3]],[9,9])

    def test_incomplete_initial_reports_and_modified_output_are_rejected(self):
        c,o,p=self.inputs()
        with self.assertRaisesRegex(ValueError,'completed_initial_triplets'):m.select(c,o,p[:-1])
        o[-1]['output']+='Edited.'
        with self.assertRaisesRegex(ValueError,'output_text_changed'):m.select(c,o,p)

    def test_budget_overrun_does_not_silently_shorten_or_drop_text(self):
        c,o,p=self.inputs()
        for item in o:item['source']='Unshortened '*1100
        with self.assertRaisesRegex(ValueError,'daily_budget'):m.select(c,o,p)


if __name__=='__main__':unittest.main()
