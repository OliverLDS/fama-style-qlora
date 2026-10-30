"""Closed transport, complete claim evidence and development isolation checks."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.research.human_prose_realization_review import parse_responses, validate_assessment


class HumanProseRealizationReviewTests(unittest.TestCase):
    def annotation(self):
        return {'semantic_ledger':{'claims':[{'claim_id':'C1'},{'claim_id':'C2'}]}}

    def assessment(self):
        return {'output_claim_groups':[['C1'],['C2']], 'claims':[
            {'claim_id':'C1','status':'minor_shift','note':'Qualifier weakened.',
                'output_evidence':[{'quote':'Firms invest.','occurrence':1}]},
            {'claim_id':'C2','status':'no_clear_change_identified','note':'No clear change noticed.',
                'output_evidence':[{'quote':'Banks lend.','occurrence':1}]}]}

    def test_exact_evidence_and_complete_claim_coverage_required(self):
        text='Firms invest. Banks lend.'
        self.assertTrue(validate_assessment(self.annotation(),text,self.assessment()))
        for mutate in ('invented_quote','missing_claim','wrong_sentence','unknown_status'):
            a=self.assessment()
            if mutate=='invented_quote':a['claims'][0]['output_evidence'][0]['quote']='Firms may invest.'
            elif mutate=='missing_claim':a['claims'].pop()
            elif mutate=='wrong_sentence':a['output_claim_groups']=[['C2'],['C1']]
            else:a['claims'][0]['status']='verified'
            with self.assertRaises(ValueError):validate_assessment(self.annotation(),text,a)

    def packet(self,root):
        package=root/'package';responses=root/'responses';package.mkdir();responses.mkdir()
        jobs=[{'job_id':str(i),'batch':1 if i<8 else 2} for i in range(16)]
        (package/'private_key.json').write_text(json.dumps(jobs))
        from fama_style.research.author_comparison_corpus import file_sha
        (package/'manifest.json').write_text(json.dumps({'files':{'private_key.json':file_sha(package/'private_key.json')}}))
        rows=[[{'job_id':str(i),'output':'A complete sentence.'} for i in range(8)],
              [{'job_id':str(i),'output':'Another complete sentence.'} for i in range(8,16)]]
        self.write(responses,rows)
        return package,responses,rows

    def write(self,path,batches):
        for i,rows in enumerate(batches,1):
            (path/('responses_%02d.jsonl'%i)).write_text('\n'.join(json.dumps(r) for r in rows)+'\n')

    def test_transport_covers_16_and_preserves_output_bytes_as_text(self):
        with tempfile.TemporaryDirectory() as directory:
            package,responses,rows=self.packet(Path(directory))
            jobs,accepted,hashes=parse_responses(package,responses)
            self.assertEqual(len(jobs),16);self.assertEqual(len(accepted),16);self.assertEqual(len(hashes),2)
            self.assertEqual(accepted['0']['output'],rows[0][0]['output'])

    def test_transport_rejects_duplicate_missing_extra_empty_and_wrong_batch(self):
        for issue in ('duplicate','missing','extra','empty','wrong_batch','unhashable_id'):
            with self.subTest(issue=issue), tempfile.TemporaryDirectory() as directory:
                package,responses,rows=self.packet(Path(directory))
                if issue=='duplicate':rows[0].append(copy.deepcopy(rows[0][0]))
                elif issue=='missing':rows[1].pop()
                elif issue=='extra':rows[0][0]['commentary']='extra'
                elif issue=='empty':rows[0][0]['output']='  '
                elif issue=='wrong_batch':rows[0][0]['job_id']='8'
                else:rows[0][0]['job_id']=[]
                self.write(responses,rows)
                with self.assertRaises(ValueError):parse_responses(package,responses)

    def test_frozen_key_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            package,responses,rows=self.packet(Path(directory))
            (package/'private_key.json').write_text('[]')
            with self.assertRaisesRegex(ValueError,'frozen_file_changed'):parse_responses(package,responses)
