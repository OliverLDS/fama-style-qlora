import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.research.author_pair_review import pair_holds, repair_schema_direction, review, structural_holds


def sample():
    source='Investors may trade less.'
    target='Trading frequency may decline for investors.'
    def schema(text):
        return {'sentences':[{'sentence_id':'S1','exact_text':text,'claim_ids':['C1']}],
                'paragraph_structure':[],'sentence_structure':[],'language_features':[]}
    return {'paragraph_id':'P1','batch_id':'batch_01','plain_english':source,'original_paragraph':target,
            'source_schema':schema(source),'target_schema':schema(target),'transformations':[],
            'source_quality':{'status':'usable','issues':[],'needs_external_context':False},
            'needs_review':False,'rewrite_status':'simplified','review_notes':[],
            'provenance':{'visible_model_label':None,'thinking_setting':None},
            'semantic_ledger':{'entities':[], 'relations':[],'unmatched_source_claims':[],
                'claims':[{'claim_id':'C1','actor_ids':[],'other_entity_ids':[],
                           'proposition':'Trading frequency may decline.',
                           'original_evidence':[{'quote':target,'occurrence':1}],
                           'rewrite_evidence':[{'quote':source,'occurrence':1}],
                           'alignment_status':'preserved'}]}}


class PairReviewTests(unittest.TestCase):
    def test_proven_swap_preserves_text_ledger_program_and_raw(self):
        row=sample();row['source_schema'],row['target_schema']=row['target_schema'],row['source_schema']
        before=copy.deepcopy(row);effective,repairs=repair_schema_direction(row)
        self.assertEqual(row,before)
        self.assertEqual(len(repairs),1)
        self.assertEqual(effective['source_schema']['sentences'][0]['exact_text'],row['plain_english'])
        for field in ('plain_english','original_paragraph','semantic_ledger','transformations','provenance'):
            self.assertEqual(effective[field],row[field])

    def test_partial_or_identical_text_cannot_prove_swap(self):
        row=sample();row['source_schema']['sentences'][0]['exact_text']='Only a fragment.'
        self.assertEqual(repair_schema_direction(row)[1],[])
        row=sample();row['plain_english']=row['original_paragraph']
        self.assertEqual(repair_schema_direction(row)[1],[])

    def test_unknown_content_and_numeric_changes_remain_held(self):
        row=sample();audit={'errors':[],'warnings':[]}
        self.assertEqual(pair_holds(row,audit),[])
        row['source_quality']['needs_external_context']=True
        self.assertIn('missing_external_context',pair_holds(row,audit))
        row=sample();row['semantic_ledger']['claims'][0]['alignment_status']='uncertain'
        self.assertIn('claim_alignment_not_preserved',pair_holds(row,audit))
        audit['warnings']=[{'code':'numeric_surface_change_needs_scope_check'}]
        self.assertIn('numeric_scope_check_pending',pair_holds(sample(),audit))

    def test_structure_failure_does_not_certify_or_repair_text(self):
        audit={'errors':[],'warnings':[], 'program_errors':[{'code':'operation_claim_outside_sentences'}]}
        self.assertEqual(pair_holds(sample(),audit),[])
        self.assertEqual(structural_holds(audit),['operation_claim_outside_sentences'])

    def fixture(self, root):
        release=root/'research_runs/author_pair_expansion_v1/release_v1'
        papers=[];raw_paths=[]
        def put(path,value):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')
        for number,name in enumerate(('existing_collection','new_collection'),1):
            package=release/name;row=sample();row['paragraph_id']='P'+str(number)
            paper_id='WORK'+str(number)
            papers.append({'paper_id':paper_id,'authors':['Synthetic Author'],'title':'Synthetic Work '+str(number),
                           'version_year':1990,'pdf_sha256':'synthetic'})
            put(package/'common/annotation_output.schema.json',{'type':'object'})
            put(package/'common/taxonomy_catalog.compact.json',{})
            put(package/'input.jsonl',{'paragraph_id':row['paragraph_id'],'text':row['original_paragraph']})
            put(package/'selection.private.json',[{'paragraph_id':row['paragraph_id'],'paper_id':paper_id,
                'role':'training','original_sha256':hashlib.sha256(row['original_paragraph'].encode()).hexdigest()}])
            put(package/'PAPER_ROLES.json',{'papers':[{'paper_id':paper_id,'role':'training'}]})
            put(package/'BATCH_INDEX.json',[{'paper_id':paper_id,'style_group':'Synthetic','role':'training',
                'batch_id':'batch_01','input':'input.jsonl',
                'save_response':'../responses/'+name+'/training/responses_01.jsonl'}])
            manifest={'output_sha256':{str(p.relative_to(package)):hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in package.rglob('*') if p.is_file()}}
            put(package/'manifest.json',manifest)
            raw=release/'responses'/name/'training/responses_01.jsonl';put(raw,row);raw_paths.append(raw)
        put(root/'docs/author_inventory_v3/inventory.json',{'papers':papers})
        (root/'reserved.jsonl').write_text('INVALID JSON: RESERVED CONTENT MUST STAY UNREAD')
        return release,raw_paths

    def test_import_retains_null_provenance_and_false_training_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);release,raw_paths=self.fixture(root)
            before={p:p.read_bytes() for p in raw_paths}
            with patch('fama_style.research.author_pair_review.expand_catalog',return_value=[]):
                result=review(release,root/'audit',root/'data')
            self.assertEqual(result['provisional_pairs'],2)
            self.assertEqual(result['training_ready_rows'],0)
            self.assertEqual(result['raw_provenance_counts'],[{'model':None,'thinking_effort':None,'rows':2}])
            self.assertEqual(result['session_provenance']['user_observed_thinking_effort'],'High')
            self.assertFalse(result['session_provenance']['actual_backend_effort_verified'])
            for path,raw in before.items():self.assertEqual(path.read_bytes(),raw)
            exported=json.loads((root/'data/training.provisional.private.jsonl').read_text().splitlines()[0])
            self.assertFalse(exported['training_ready'])
            self.assertEqual(exported['authors'],['Synthetic Author'])

    def test_changed_original_fails_before_pair_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);release,raw=self.fixture(root)
            row=json.loads(raw[0].read_text());row['original_paragraph']='Altered target.'
            raw[0].write_text(json.dumps(row)+'\n')
            with patch('fama_style.research.author_pair_review.expand_catalog',return_value=[]):
                with self.assertRaisesRegex(ValueError,'original_or_batch_changed'):
                    review(release,root/'audit',root/'data')
            self.assertFalse((root/'data').exists())


if __name__=='__main__': unittest.main()
