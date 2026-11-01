"""PDF text binding, missing metadata and conflicting scans without a service."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fama_style.research import checkpoint_detector_pdfs as m
from fama_style.evaluation.milestone_checkpoint_4b import sha


class DetectorPDFs(unittest.TestCase):
    def item(self,index=0):
        text=f'Firms in sector {index} may invest if the tax rate falls.'
        return {'scan_id':f'SCAN_{index}','round':1,'text':text,'text_sha256':sha(text)}

    def page(self,item,score=12,metadata=True):
        text=('Version 2026-10-07-baseTitle - 11/01/2026\n' if metadata else '')
        text+='We predict this text is\nHuman Generated\nAI Probability\n'+str(score)+'%\n'+item['text'].replace(' may ','\nmay\n')
        return {'page_extracted':1,'text':text}

    def test_native_values_and_unknown_metadata(self):
        item=self.item();result=m.bind_page(self.page(item),[item])
        self.assertEqual(result['scan_id'],item['scan_id']);self.assertEqual(result['ai_probability_percent'],12)
        self.assertEqual(result['report_date'],'2026-11-01');self.assertEqual(result['provider_version'],'2026-10-07-base')
        result=m.bind_page(self.page(item,metadata=False),[item]);self.assertIsNone(result['report_date'])
        self.assertIsNone(result['provider_version']);self.assertIsNone(result['numeric_decision_threshold'])

    def test_partial_wrong_ambiguous_or_unsupported_report_never_binds(self):
        item=self.item();page=self.page(item);page['text']=page['text'].replace(' tax rate falls.',' tax rate rises.')
        with self.assertRaisesRegex(ValueError,'no_complete'):m.bind_page(page,[item])
        with self.assertRaisesRegex(ValueError,'ambiguous'):m.bind_page(self.page(item),[item,item|{'scan_id':'another'}])
        page=self.page(item);page['page_extracted']=2
        with self.assertRaisesRegex(ValueError,'page_one'):m.bind_page(page,[item])
        page=self.page(item);page['text']=page['text'].replace('Human Generated','Unsupported Label')
        with self.assertRaisesRegex(ValueError,'unrecognized'):m.bind_page(page,[item])

    def test_conflicting_repeat_reports_are_not_cherry_picked(self):
        item=self.item();a=m.bind_page(self.page(item,4),[item])|{'status':'parsed','pdf_sha256':'first'}
        b=m.bind_page(self.page(item,90),[item])|{'status':'parsed','pdf_sha256':'second'}
        result=m.results_table([item],[a,b])[0]
        self.assertEqual(result['status'],'conflicting_repeat_reports');self.assertEqual(result['score'],'')
        self.assertEqual(m.results_table([item],[])[0]['status'],'not_tested')

    def test_arbitrary_filenames_match_text_and_duplicate_bytes_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);exported=root/'export';(exported/'detector').mkdir(parents=True)
            items=[self.item(i) for i in range(20)]
            path=exported/'detector/PRIVATE_KEY.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in items))
            (exported/'manifest.json').write_text(json.dumps({'files':{'detector/PRIVATE_KEY.jsonl':m.file_sha(path)}}))
            inputs=root/'pdfs';inputs.mkdir()
            for name in ('Default browser filename.pdf','Browser duplicate (1).PDF'):(inputs/name).write_bytes(b'Synthetic same PDF bytes')
            before=path.read_bytes()
            def extract(pdf,target):
                page=self.page(items[7]);target.write_text(json.dumps(page));return page
            with patch.object(m,'extract_first_page',side_effect=extract):
                summary=m.build(exported,inputs,root/'output')
            self.assertEqual(summary['scored_texts'],1);self.assertEqual(summary['missing_texts'],19)
            self.assertEqual(summary['identical_duplicate_files'],1);self.assertEqual(summary['external_calls'],0)
            self.assertFalse(summary['user_renaming_required']);self.assertEqual(path.read_bytes(),before)
            with self.assertRaisesRegex(ValueError,'overwrite'):m.build(exported,inputs,root/'output')

if __name__=='__main__':unittest.main()
