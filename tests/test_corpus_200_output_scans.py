"""Protect preselected model-output scans, full-text binding and quota bounds."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('corpus200_output_scans',ROOT/'scripts/data/prepare_corpus_200_output_scans.py')
scans=importlib.util.module_from_spec(spec);spec.loader.exec_module(scans)


class Corpus200OutputScans(unittest.TestCase):
    def fixture(self,root):
        download=root/'download';review=download/'review_v1';review.mkdir(parents=True)
        packet=root/'packet';(packet/'upload').mkdir(parents=True)
        (download/'raw/corpus_200_pilot_v1').mkdir(parents=True)
        ids=[g+str(i) for g in ['Fama','Shiller','Holmstrom','Baker'] for i in [1,2]]
        config={'detector_case_ids':ids};items=[]
        for rid in ids:
            for stage in ['base','epoch_1']:
                text=rid+' '+stage+' '+('complete unchanged paragraph words '*16)
                items.append({'example_id':rid,'paper_id':'paper_'+rid[:-1],'style_group':rid[:-1],
                    'snapshot':stage,'output':text,'output_sha256':scans.sha(text),
                    'assessment':{'material_claim_change':rid=='Fama1','minor_semantic_shifts':0}})
        joined=review/'outputs_and_assessments.private.jsonl'
        joined.write_text(''.join(json.dumps(r)+'\n' for r in items))
        (review/'manifest.json').write_text(json.dumps({'files':{joined.name:scans.file_sha(joined)}}))
        (packet/'upload/corpus_200_config.json').write_text(json.dumps(config))
        (download/'raw/corpus_200_pilot_v1/evaluation.json').write_text(json.dumps({'config':config}))
        return download,packet,items

    def test_keeps_all_preselected_outputs_including_error_flags_and_exact_text(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);download,packet,items=self.fixture(root);out=root/'scans'
            summary=scans.build(download,packet,out)
            self.assertEqual(summary['texts'],16)
            self.assertTrue(summary['selection_not_filtered_by_quality_or_scores'])
            self.assertEqual(summary['known_quality_flags_in_trained_selected_outputs']['material_claim_change'],1)
            self.assertEqual([r['texts'] for r in summary['rounds']],[8,8])
            self.assertTrue(all(r['estimated_credits']<=20 for r in summary['rounds']))
            key=[json.loads(l) for l in (out/'PRIVATE_KEY.jsonl').read_text().splitlines()]
            expected={(r['example_id'],r['snapshot']):r['output'] for r in items}
            for row in key:
                self.assertEqual(row['text'],expected[row['example_id'],row['condition']])
                self.assertEqual((out/'texts'/f"{row['scan_id']}.txt").read_text(),row['text'])
            self.assertEqual(summary['new_detector_calls'],0)
            self.assertFalse(summary['dataset_inputs_or_human_references_rescanned'])

    def test_rejects_reselected_cases_and_changed_review_bytes(self):
        for mutation in ['config','review']:
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);download,packet,_=self.fixture(root)
                if mutation=='config':
                    p=packet/'upload/corpus_200_config.json';config=json.loads(p.read_text());config['detector_case_ids'].reverse();p.write_text(json.dumps(config))
                else:
                    p=download/'review_v1/outputs_and_assessments.private.jsonl';p.write_text(p.read_text()+'\n')
                with self.assertRaises(ValueError):scans.build(download,packet,root/'scans')


if __name__=='__main__':unittest.main()
