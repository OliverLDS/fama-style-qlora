import copy
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from fama_style.research.human_prose_review import schema_errors, occurrence_span, apply_program_repairs
from fama_style.research.human_prose_inference import payload


class HumanProseReviewTests(unittest.TestCase):
    def test_closed_schema_rejects_bool_integer_and_extra(self):
        schema={'type':'object','required':['n'],'additionalProperties':False,
                'properties':{'n':{'type':'integer','minimum':1}}}
        self.assertEqual(schema_errors({'n':1},schema),[])
        self.assertTrue(schema_errors({'n':True},schema))
        self.assertTrue(schema_errors({'n':1,'extra':0},schema))
        with self.assertRaises(ValueError): schema_errors('x',{'unknown':True})

    def test_exact_evidence_occurrence(self):
        self.assertEqual(occurrence_span('a b a',{'quote':'a','occurrence':2}),(4,5))
        self.assertIsNone(occurrence_span('a b a',{'quote':'a','occurrence':3}))
        self.assertIsNone(occurrence_span('a',{'quote':'a','occurrence':True}))

    def row(self):
        return {'transformations':[{'operation':'SPLIT','claim_ids':['C1','C2'],
                    'source_sentence_ids':['S1','S2'],'target_sentence_ids':['S1']}],
                'source_schema':{'sentences':[{'sentence_id':'S1','claim_ids':['C1']},
                    {'sentence_id':'S2','claim_ids':['C2']} ]}}

    def test_direction_repair_is_derived_and_cardinality_bound(self):
        row=self.row();before=copy.deepcopy(row)
        patch={'operation_index':0,'field':'operation','before':'SPLIT','after':'MERGE'}
        self.assertEqual(apply_program_repairs(row,[patch])['transformations'][0]['operation'],'MERGE')
        self.assertEqual(row,before)
        patch['after']='KEEP'
        with self.assertRaises(ValueError): apply_program_repairs(row,[patch])

    def test_binding_repair_cannot_add_unrelated_sentence_or_prose(self):
        row=self.row();row['transformations'][0]['source_sentence_ids']=['S1']
        patch={'operation_index':0,'field':'source_sentence_ids','before':['S1'],'after':['S1','S2']}
        self.assertEqual(len(apply_program_repairs(row,[patch])['transformations'][0]['source_sentence_ids']),2)
        row['source_schema']['sentences'][1]['claim_ids']=[]
        with self.assertRaises(ValueError): apply_program_repairs(row,[patch])
        patch['field']='explanation'
        with self.assertRaises(ValueError): apply_program_repairs(row,[patch])

    def test_generator_projection_omits_reference_wording_and_bad_bindings(self):
        obs={'canonical_id':'P1','assignment_status':'assigned','value':'present',
             'claim_ids':['C1'],'sentence_ids':['S1'],'evidence':['REFERENCE_SECRET']}
        a={'plain_english':'SOURCE_ONLY','original_paragraph':'REFERENCE_SECRET',
           'semantic_ledger':{'claims':[{'claim_id':'C1','proposition':'REFERENCE_SECRET',
                'rewrite_evidence':[{'quote':'SOURCE_ONLY'}]}]},
           'target_schema':{'sentences':[{'sentence_id':'S1','claim_ids':['C1'],'exact_text':'REFERENCE_SECRET'}],
                'paragraph_structure':[obs],'sentence_structure':[],'language_features':[]}}
        catalog={'P1':{'control_status':'candidate_presentation_control','name':'Contrast','definition':'Generic contrast'}}
        ordinary=payload({'annotation':a},catalog,False)
        guided=payload({'annotation':a},catalog,True)
        self.assertEqual({k:v for k,v in guided.items() if k!='presentation_plan'},ordinary)
        self.assertNotIn('REFERENCE_SECRET',str(guided))
        self.assertEqual(len(guided['presentation_plan']['controls']),1)
        obs['claim_ids']=['C2']
        self.assertEqual(payload({'annotation':a},catalog,True)['presentation_plan']['controls'],[])
        obs['claim_ids']=['C1'];catalog['P1']['control_status']='semantic_sensitive_protected'
        self.assertEqual(payload({'annotation':a},catalog,True)['presentation_plan']['controls'],[])

    def test_reserved_transport_never_reaches_content_audit(self):
        from fama_style.research import human_prose_review as module
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);package=root/'package';responses=root/'responses'
            (package/'common').mkdir(parents=True);(responses/'reserved_test').mkdir(parents=True)
            def put(path,value): path.write_text(json.dumps(value))
            put(package/'common/annotation_output.schema.json',{'type':'object'})
            put(package/'common/taxonomy_catalog.compact.json',{})
            put(package/'manifest.json',{})
            put(package/'BATCH_INDEX.json',[{'save_response':'responses_01.jsonl','split':'reserved_test',
                'batch_id':'batch_01','input':'input.jsonl'}])
            put(package/'PRIVATE_SELECTION_AND_SPLITS.json',[])
            put(package/'input.jsonl',{'paragraph_id':'reserved','text':'Reserved text.'})
            put(responses/'reserved_test/responses_01.jsonl',{'paragraph_id':'reserved','original_paragraph':'Reserved text.'})
            put(root/'session.json',{})
            with patch.object(module,'verify_files'), patch.object(module.preparation,'expand_catalog',return_value=[]), \
                    patch.object(module,'audit_record',side_effect=AssertionError('reserved scoring forbidden')):
                result=module.run(package,responses,root/'out',root/'session.json')
            self.assertEqual(result['reserved']['content_rows_audited'],0)
            self.assertEqual(result['reserved']['rows'],1)
            self.assertEqual((root/'out/audits.private.jsonl').read_text(),'')
