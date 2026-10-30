"""Response provenance, plan-deviation reports and composite claim abstention."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.research import human_prose_grouping_review as module
from fama_style.research.human_prose_grouping import CONDITIONS,PLAN_STATUSES
from fama_style.research.author_comparison_corpus import file_sha


class HumanProseGroupingReviewTests(unittest.TestCase):
    def fixture(self,root):
        p=root/'package';r=root/'responses';p.mkdir();r.mkdir()
        jobs=[];batches=[[],[],[]]
        for i in range(5):
            for c,condition in enumerate(CONDITIONS):
                job_id=format(i*3+c,'016x');batch=(i+c)%3
                jobs.append({'job_id':job_id,'batch':batch+1,'condition':condition,'split':'development'})
                batches[batch].append({'job_id':job_id,'output':'Firms invest.',
                    'plan_status':'not_requested' if c==0 else 'followed','plan_note':''})
        schema={'type':'object','required':['job_id','output','plan_status','plan_note'],'additionalProperties':False,
            'properties':{'job_id':{'type':'string','pattern':'^[a-f0-9]{16}$'},'output':{'type':'string','minLength':1},
                'plan_status':{'type':'string','enum':list(PLAN_STATUSES)},'plan_note':{'type':'string'}}}
        (p/'private_key.json').write_text(json.dumps(jobs));(p/'output.schema.json').write_text(json.dumps(schema))
        (p/'manifest.json').write_text(json.dumps({'files':{name:file_sha(p/name) for name in ('private_key.json','output.schema.json')}}))
        self.save(r,batches)
        return p,r,batches,jobs

    def save(self,path,batches):
        for i,rows in enumerate(batches,1):
            (path/('responses_%02d.jsonl'%i)).write_text('\n'.join(json.dumps(r) for r in rows)+'\n')

    def test_full_transport_and_explained_deviation_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            p,r,rows,jobs=self.fixture(Path(d))
            candidate=next(row for batch in rows for row in batch if row['plan_status']=='followed')
            candidate.update(plan_status='deviated_for_meaning',plan_note='Two claims could not be combined without changing scope.')
            self.save(r,rows)
            _,accepted,hashes=module.parse_responses(p,r)
            self.assertEqual(len(accepted),15);self.assertEqual(len(hashes),3)
            self.assertEqual(accepted[candidate['job_id']]['plan_note'],candidate['plan_note'])

    def test_missing_duplicate_wrong_batch_empty_and_extra_fields_fail_closed(self):
        for mode in ('missing','duplicate','batch','empty','extra'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                p,r,rows,jobs=self.fixture(Path(d))
                if mode=='missing':rows[0].pop()
                elif mode=='duplicate':rows[0].append(copy.deepcopy(rows[0][0]))
                elif mode=='batch':rows[0][0]['job_id']=rows[1][0]['job_id']
                elif mode=='empty':rows[0][0]['output']=' '
                else:rows[0][0]['rating']=1
                self.save(r,rows)
                with self.assertRaises(ValueError):module.parse_responses(p,r)

    def test_deviation_note_and_condition_must_agree(self):
        for mode in ('unexplained','misclassified','spurious_note'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                p,r,rows,jobs=self.fixture(Path(d))
                row=next(row for batch in rows for row in batch if row['plan_status']=='followed')
                if mode=='unexplained':row['plan_status']='deviated_for_meaning'
                elif mode=='misclassified':row['plan_status']='not_requested'
                else:row['plan_note']='A deviation.'
                self.save(r,rows)
                with self.assertRaises(ValueError):module.parse_responses(p,r)

    def test_grouping_counts_do_not_prove_assignment(self):
        self.assertEqual(module.group_status([['C1'],['C2','C3']],[['C1','C2'],['C3']],False),'not_matched')
        self.assertEqual(module.group_status([['C1'],['C2']],[['C1'],['C2']],True),'ambiguous_composite_claim')
        self.assertEqual(module.group_status([['C1'],['C2']],[['C1','C2']],True),'not_matched')

    def test_each_repeated_claim_assignment_needs_evidence(self):
        a={'semantic_ledger':{'claims':[{'claim_id':'C1'},{'claim_id':'C2'}]}}
        output='Firms invest. Banks lend.'
        review={'grouping_ambiguous':False,'grouping_note':'','output_claim_groups':[['C1'],['C1','C2']],
            'claims':[{'claim_id':'C1','status':'no_clear_change_identified','note':'Partial components across sentences.',
                'output_evidence':[{'quote':'Firms invest.','occurrence':1}]},
            {'claim_id':'C2','status':'minor_shift','note':'Qualifier changed.',
                'output_evidence':[{'quote':'Banks lend.','occurrence':1}]}]}
        with self.assertRaisesRegex(ValueError,'unanchored_group_claim'):module.validate_assessment(a,output,review)
        review['claims'][0]['output_evidence'].append({'quote':'Banks lend.','occurrence':1})
        module.validate_assessment(a,output,review)
        review['claims'][0]['output_evidence'][0]['quote']='Invented.'
        with self.assertRaises(ValueError):module.validate_assessment(a,output,review)

    def test_reserved_job_rejected_before_response_or_schema_read(self):
        with tempfile.TemporaryDirectory() as d:
            p,r,rows,jobs=self.fixture(Path(d));jobs[0]['split']='reserved_test'
            (p/'private_key.json').write_text(json.dumps(jobs))
            (p/'manifest.json').write_text(json.dumps({'files':{'private_key.json':file_sha(p/'private_key.json')}}))
            (p/'output.schema.json').unlink()
            with self.assertRaisesRegex(ValueError,'development_only'):module.parse_responses(p,r)
