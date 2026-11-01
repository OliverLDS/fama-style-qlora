"""Offline salvage keeps transport normalization distinct from schema validity."""
import json
import unittest
from fama_style.research.reconstruction_4b_review import recover


class Reconstruction4BReview(unittest.TestCase):
    def fixture(self):
        job={'job_id':'SYN_direct','record_id':'SYN','method':'direct_rewrite','prompt_sha256':'hash'}
        obj={'record_id':'SYN','method':'direct_rewrite','status':'generated','text':'Synthetic text.','reason':''}
        row=job|{'raw_output':'```json\n'+json.dumps(obj)+'\n```',
                 'parsed':{'status':'format_error','text':None,'reason':'Original fence rejection.'},
                 'hit_output_limit':False,'output_tokens':30}
        return job,obj,row

    def test_fence_normalization_preserves_raw_text_and_original_error(self):
        job,obj,row=self.fixture();result=recover(row,job)
        self.assertTrue(result['schema_conforming'])
        self.assertEqual(result['text'],obj['text'])
        self.assertEqual(result['raw_output'],row['raw_output'])
        self.assertEqual(result['original_parser_result']['status'],'format_error')
        self.assertEqual(result['transport'],'single_markdown_fence')

    def test_missing_fields_remain_errors_but_saved_text_is_assessable(self):
        job,obj,row=self.fixture();del obj['status'];del obj['reason']
        row['raw_output']='```json\n'+json.dumps(obj)+'\n```'
        result=recover(row,job)
        self.assertFalse(result['schema_conforming'])
        self.assertEqual(result['missing_fields'],['reason','status'])
        self.assertEqual(result['text'],obj['text'])
        self.assertEqual(result['normalized_parser_result']['status'],'format_error')
        self.assertFalse(result['training_ready'])

    def test_job_binding_mismatch_is_not_salvaged(self):
        job,obj,row=self.fixture();row['prompt_sha256']='changed'
        with self.assertRaises(ValueError):recover(row,job)
        job,obj,row=self.fixture();obj['record_id']='other';row['raw_output']=json.dumps(obj)
        with self.assertRaises(ValueError):recover(row,job)


if __name__=='__main__':unittest.main()
