"""Reference controls must use frozen held-out targets, never model prose."""
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('references500',ROOT/'scripts/data/prepare_corpus_500_reference_scans.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)


class Reference500Scans(unittest.TestCase):
    def inputs(self):
        ids=[f'case{i}' for i in range(8)];rows=[]
        for rid in ids:
            text=('Original '+rid+' ')*30
            rows.append(dict(example_id=rid,content_id=rid,parent_id='parent-'+rid,paper_id='paper',style_group='group',authors=['Author'],
                split='development',input_condition='gemini_input',source='AI-generated input.',target=text,target_sha256=m.r.runtime.sha(text)))
        return dict(detector_case_ids=ids),rows

    def test_exact_target_and_frozen_content_order_are_preserved(self):
        config,rows=self.inputs();cards=m.select(config,list(reversed(rows)))
        self.assertEqual([x['example_id'] for x in cards],config['detector_case_ids'])
        self.assertEqual([x['text'] for x in cards],[x['target'] for x in rows])
        self.assertEqual({x['condition'] for x in cards},{'human_reference'})
        self.assertTrue(all(x['text_sha256']==rows[i]['target_sha256'] for i,x in enumerate(cards)))

    def test_edited_targets_training_rows_and_incomplete_panels_are_rejected(self):
        config,rows=self.inputs();rows[0]['target']+=' Correction.'
        with self.assertRaisesRegex(ValueError,'reference_changed'):m.select(config,rows)
        config,rows=self.inputs();rows[0]['split']='training'
        with self.assertRaisesRegex(ValueError,'held_out_reference'):m.select(config,rows)
        config,rows=self.inputs();config['detector_case_ids']=config['detector_case_ids'][:-1]
        with self.assertRaisesRegex(ValueError,'eight_frozen'):m.select(config,rows)


if __name__=='__main__':unittest.main()
