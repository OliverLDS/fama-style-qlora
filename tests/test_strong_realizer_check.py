"""Strict manual-response import, source-only prompts and CPU package replay."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.rhetoric import strong_realizer_check as e

class StrongRealizerCheckTests(unittest.TestCase):
    def fixture(self,folder):
        jobs=[{'job_id':'J'+str(i)*16,'prompt_sha256':str(i)*64,'response_file':f'responses_{i:02d}.jsonl'} for i in [1,2]]
        for j in jobs:(folder/j['response_file']).write_text(json.dumps({'job_id':j['job_id'],'prompt_sha256':j['prompt_sha256'],'output':'Synthetic returned text.'})+'\n')
        return {'jobs':jobs}

    def test_valid_response_coverage_preserves_original_text(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);key=self.fixture(folder)
            p=folder/'responses_01.jsonl';r=json.loads(p.read_text());r['output']='  Original whitespace.\nSecond sentence.  ';p.write_text(json.dumps(r)+'\n')
            valid,audit=e.parse_responses(key,folder)
            self.assertTrue(audit['complete']);self.assertEqual(len(valid),2)
            self.assertEqual(valid[0]['output'],r['output']);self.assertTrue(audit['no_repairs_or_best_of_selection'])

    def test_duplicates_exclude_every_copy_instead_of_selecting_first(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);key=self.fixture(folder);p=folder/'responses_01.jsonl';p.write_text(p.read_text()*2)
            valid,audit=e.parse_responses(key,folder)
            self.assertFalse(audit['complete']);self.assertEqual(len(valid),1)
            self.assertIn('J'+'1'*16,audit['missing_or_rejected_job_ids'])
            self.assertIn('duplicate_job_all_copies_excluded',{a['code'] for a in audit['errors']})

    def test_wrong_batch_hash_unknown_fields_and_empty_outputs_fail(self):
        for mode in ['batch','hash','fields','empty','unknown','duplicate_keys','malformed','utf8','missing']:
            with tempfile.TemporaryDirectory() as name:
                folder=Path(name);key=self.fixture(folder);p=folder/'responses_01.jsonl';r=json.loads(p.read_text())
                if mode=='batch':r['job_id']=key['jobs'][1]['job_id'];r['prompt_sha256']=key['jobs'][1]['prompt_sha256']
                if mode=='hash':r['prompt_sha256']='f'*64
                if mode=='fields':r['reference']='Forbidden extra field'
                if mode=='empty':r['output']='  '
                if mode=='unknown':r['job_id']='J'+'9'*16
                p.write_text(json.dumps(r)+'\n')
                if mode=='duplicate_keys':p.write_text('{"job_id":"one","job_id":"two"}\n')
                if mode=='malformed':p.write_text('```json\n')
                if mode=='utf8':p.write_bytes(b'\xff')
                if mode=='missing':p.unlink()
                valid,audit=e.parse_responses(key,folder)
                self.assertFalse(audit['complete'],mode);self.assertLess(len(valid),2,mode)

    def test_citation_alert_is_a_warning_not_a_semantic_verdict(self):
        source='Roll (1986) discusses mergers; Mitchell and Stafford (1997) discuss acquirers.'
        alerts=e.citation_context_alerts(source,'Roll (1986) explains proxy contests.')
        self.assertEqual(len(alerts),1);self.assertFalse(alerts[0]['semantic_error_verified'])
        self.assertEqual(e.citation_context_alerts(source,'Roll (1986) discusses an investment decision.'),[])

    def test_response_schema_has_no_assessment_or_reference_fields(self):
        self.assertEqual(set(e.RESPONSE_SCHEMA['properties']),{'job_id','prompt_sha256','output'})
        self.assertFalse(e.RESPONSE_SCHEMA['additionalProperties'])
        self.assertIn('separate new conversation',e.START_HERE)
        self.assertIn('do not prove',e.START_HERE)
        self.assertNotIn('clear memory',e.TASK_PROMPT)

    def test_prepared_batches_are_exact_and_uploads_exclude_reference_metadata(self):
        package=e.ROOT/'research_runs/strong_realizer_check_v1/package_v1'
        if not package.exists():self.skipTest('Private prepared package absent in public checkout')
        key=e.check_package(package);self.assertEqual(len(key['jobs']),10)
        for f in (package/'batches').glob('*.jsonl'):
            for r in e.read_lines(f):self.assertEqual(set(r),{'job_id','prompt_sha256','prompt'})
        expected={r['job_id']:r for r in key['jobs']}
        for job in key['jobs']:
            self.assertIn(job['condition'],e.CONDITIONS)
            self.assertNotIn(job['use'],json.dumps(next(r for r in e.read_lines(package/'batches'/job['batch_file']) if r['job_id']==job['job_id'])))
        # Import synthetic returned texts without requiring a Secret, network or ML.
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);responses=folder/'responses';responses.mkdir()
            for filename in ['responses_01.jsonl','responses_02.jsonl']:
                e.write_lines(responses/filename,[{'job_id':j['job_id'],'prompt_sha256':j['prompt_sha256'],'output':'Synthetic placeholder sentence.'} for j in key['jobs'] if j['response_file']==filename])
            e.audit_responses(package,responses,folder/'audit')
            summary=json.loads((folder/'audit/summary.json').read_text())
            self.assertEqual(summary['accepted_outputs'],10);self.assertIsNone(summary['semantic_accuracy']);self.assertEqual(summary['model_calls'],0)
            self.assertEqual(sum(g['accepted'] for g in summary['conditions']['preservation_control'].values()),2)
            self.assertTrue(all(e.file_sha(responses/f)==e.file_sha(folder/'audit/raw'/f) for f in ['responses_01.jsonl','responses_02.jsonl']))

if __name__=='__main__':unittest.main()
