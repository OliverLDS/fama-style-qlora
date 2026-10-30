"""Merge identifiability, source-only projection and whole-paper retrieval isolation."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.research import source_only_grouping as module


class SourceOnlyGroupingTests(unittest.TestCase):
    def row(self):
        return {'split':'training','paragraph_id':'q','paper_id':'query','schema_valid':True,
            'audit':{'errors':[],'program_errors':[]},'annotation':{
                'plain_english':'Firms borrow. Banks lend. Costs fall.',
                'original_paragraph':'REFERENCE_ONLY',
                'source_schema':{'sentences':[
                    {'exact_text':'Firms borrow.','claim_ids':['C1']},
                    {'exact_text':'Banks lend.','claim_ids':['C2']},
                    {'exact_text':'Costs fall.','claim_ids':['C3']}]},
                'target_schema':{'sentences':[{'claim_ids':['C1','C2']},{'claim_ids':['C3']}]},
                'semantic_ledger':{'claims':[{'claim_id':c,'alignment_status':'preserved'} for c in ('C1','C2','C3')]},
                'source_quality':{'needs_external_context':False},'needs_review':False}}

    def test_unique_partition_and_identity(self):
        r=self.row();a=module.audit_labels(r)
        self.assertEqual(a['status'],'eligible');self.assertEqual(a['merge_labels'],[1,0])
        r['annotation']['target_schema']=copy.deepcopy(r['annotation']['source_schema'])
        self.assertEqual(module.audit_labels(r)['merge_labels'],[0,0])

    def test_split_reorder_ambiguity_and_uncertain_alignment_excluded(self):
        r=self.row();r['annotation']['source_schema']['sentences']=[{'claim_ids':['C1','C2']},{'claim_ids':['C3']}]
        r['annotation']['target_schema']['sentences']=[{'claim_ids':['C1']},{'claim_ids':['C2']},{'claim_ids':['C3']}]
        self.assertEqual(module.audit_labels(r)['status'],'split_required')
        r=self.row();r['annotation']['target_schema']['sentences']=[{'claim_ids':['C3']},{'claim_ids':['C1','C2']}]
        self.assertEqual(module.audit_labels(r)['status'],'not_adjacent_merge_representable')
        self.assertEqual(len(module.partitions([['C1'],['C1'],['C1']],[['C1'],['C1']])),2)
        r=self.row();r['annotation']['semantic_ledger']['claims'][0]['alignment_status']='uncertain'
        self.assertEqual(module.audit_labels(r)['status'],'uncertain_claim_alignment')

    def test_claim_components_can_merge_but_nonidentifiable_partitions_abstain(self):
        self.assertEqual(module.partitions([['C1'],['C1'],['C2']],[['C1','C2']]),[(3,)])
        r=self.row();r['annotation']['semantic_ledger']['claims']=[{'claim_id':'C1','alignment_status':'preserved'}]
        for s in r['annotation']['source_schema']['sentences']:s['claim_ids']=['C1']
        r['annotation']['target_schema']['sentences']=[{'claim_ids':['C1']},{'claim_ids':['C1']}]
        self.assertEqual(module.audit_labels(r)['status'],'ambiguous_partition')

    def test_source_projection_and_prediction_ignore_poisoned_reference(self):
        r=self.row();v=module.source_view(r);f=module.boundary_features(v)
        r['annotation']['original_paragraph']=object();r['annotation']['target_schema']=None
        r['annotation']['semantic_ledger']=None
        self.assertEqual(module.source_view(r),v)
        self.assertEqual(module.boundary_features(module.source_view(r)),f)
        self.assertNotIn('REFERENCE_ONLY',str(v))
        self.assertEqual(len(f[0]),len(module.FEATURES))
        self.assertTrue(all(0<=x<=1 for row in f for x in row))

    def test_reserved_and_development_annotations_never_accessed(self):
        a,e=module.collect([{'split':'reserved_test'},{'split':'development'},self.row()])
        self.assertEqual(len(a),1);self.assertEqual(len(e),1)

    def pool(self):
        features=module.boundary_features(module.source_view(self.row()))[0]
        return [dict(paper_id=p,paragraph_id=p,boundary=0,features=features,label=label)
                for p,label in [('query',0),('a',1),('b',1),('c',0)]]

    def test_retrieval_excludes_query_paper_and_uses_one_neighbor_per_paper(self):
        pool=self.pool();pool.append({**pool[1],'paragraph_id':'a2','label':0})
        pred,neighbors=module.retrieve(pool[0]['features'],pool,'query')
        self.assertEqual(pred,1)
        self.assertEqual({n['paper_id'] for n in neighbors},{'a','b','c'})
        with self.assertRaises(ValueError):module.retrieve(pool[0]['features'],pool[:3],'query')

    def test_library_excludes_all_rows_from_heldout_paper(self):
        q=module.source_view(self.row())
        examples=[{'source':q,'labels':[1,0]}, {'source':{**q,'paragraph_id':'q2'},'labels':[0,1]},
                  {'source':{**q,'paragraph_id':'other','paper_id':'other'},'labels':[1,1]}]
        pool=module.library(examples,'query')
        self.assertEqual(len(pool),2);self.assertTrue(all(r['paper_id']=='other' for r in pool))

    def test_keep_fixed_pairs_and_full_source_coverage(self):
        view={'sentences':['A.','B.','C.','D.','E.'],'paper_id':'q'}
        fixed,_=module.predict(view,[],'fixed_pairs');keep,_=module.predict(view,[],'keep')
        self.assertEqual(module.plan_from_labels(view,fixed),[[1,2],[3,4],[5]])
        self.assertEqual(module.plan_from_labels(view,keep),[[1],[2],[3],[4],[5]])
        with self.assertRaises(ValueError):module.plan_from_labels(view,[1])

    def test_metrics_separate_identity_majority_and_merge_actions(self):
        rows=[{'paper_id':'a','gold_labels':[0,0],'predicted_labels':[0,0]},
              {'paper_id':'b','gold_labels':[1,0],'predicted_labels':[0,0]}]
        s=module.score(rows)
        self.assertEqual(s['identity_rows_exact'],1);self.assertEqual(s['merge_rows_exact'],0)
        self.assertEqual(s['merge_f1'],0);self.assertEqual(s['boundary_accuracy'],.75)
        self.assertIsNone(s['merge_precision'])
