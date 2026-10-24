"""Synthetic checks for representative sampling, split isolation and schema lineage."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/data'))
spec=importlib.util.spec_from_file_location('taxonomy_followup',ROOT/'scripts/data/prepare_taxonomy_followup.py')
followup=importlib.util.module_from_spec(spec);spec.loader.exec_module(followup)


class TaxonomyFollowupTests(unittest.TestCase):
    def test_resolution_samples_distinct_papers_without_assigning_candidates(self):
        categories={'S001':{'control_status':'candidate_presentation_control'},'S002':{'control_status':'descriptive'}}
        rows=[];annotations={};papers={}
        for i in range(6):
            ident=f'a{i}';pair=f'pair{i//2}';papers[pair]=f'paper{i//2}'
            annotations[ident]={'node':{'label':{'name':'broad','definition':'Synthetic broad category.'},
                'evidence':[{'quote':f'Firms report item {i}.','occurrence':1}]}}
            rows.append({'inventory_id':'L0001','annotation_id':ident,'json_path':'/node/label','pair_id':pair,
                'mapping_status':'split_requires_context','canonical_id':None,'candidate_canonical_ids':['S001','S002']})
        source=copy.deepcopy(rows)
        chosen,counts=followup.select_resolution_cases(rows,categories,annotations,papers)
        self.assertEqual(counts,{'L0001':6});self.assertEqual(len(chosen),3)
        self.assertEqual(len({papers[r['pair_id']] for r in chosen}),3)
        self.assertTrue(all(row['canonical_id'] is None for row in chosen))
        self.assertEqual(source,rows)
        self.assertEqual(chosen,followup.select_resolution_cases(rows,categories,annotations,papers)[0])
        categories['S001']['control_status']='semantic_sensitive_protected'
        self.assertEqual(followup.select_resolution_cases(rows,categories,annotations,papers)[0],[])

    def synthetic_training(self):
        return [{'id':f'{paper}-{i}','paper_key':paper,'split':'train',
            'completion':f'Synthetic {paper} observation {i}. '+'Firms report revenue. '*i,
            'plain_english':f'Synthetic {paper} reference {i}. '+'Companies report sales. '*i}
            for paper in followup.PAPERS for i in range(6)]

    def test_fresh_selection_excludes_discovery_and_nontraining_rows(self):
        rows=self.synthetic_training();excluded={r['id'] for r in rows if int(r['id'].rsplit('-',1)[1])<4}
        texts=[r['completion'] for r in rows if r['id'] in excluded]
        selected=followup.select_fresh(rows,excluded,texts)
        self.assertEqual(len(selected),7)
        self.assertEqual({r['paper_key'] for r in selected},set(followup.PAPERS))
        self.assertFalse({r['id'] for r in selected}&excluded)
        self.assertEqual(selected,followup.select_fresh(rows,excluded,texts))
        changed=copy.deepcopy(rows);changed[0]['split']='test'
        with self.assertRaises(ValueError):followup.select_fresh(changed,excluded,texts)
        with self.assertRaises(ValueError):followup.select_fresh(rows,{r['id'] for r in rows},texts)

    def test_canonical_schema_is_derived_without_mutating_discovery_contract(self):
        base=json.loads((ROOT/'docs/taxonomy_discovery_v1/annotation_output.schema.json').read_text());before=copy.deepcopy(base)
        schema=followup.canonical_schema(base,['P001'],'synthetic_version','synthetic_hash')
        self.assertEqual(base,before)
        self.assertEqual(schema['properties']['taxonomy_sha256'],{'const':'synthetic_hash'})
        self.assertIn('coverage_gaps',schema['required'])
        self.assertIn('canonical_id',schema['$defs']['label']['properties'])
        self.assertNotIn('name',schema['$defs']['label']['properties'])
        self.assertEqual(schema['properties']['provenance']['properties']['annotation_context'],{'const':'single_paragraph'})

    def test_json_pointer_and_resolution_schema(self):
        value={'a/b':[{'~key':'evidence'}]}
        self.assertEqual(followup.pointer(value,'/a~1b/0/~0key'),'evidence')
        schema=followup.resolution_schema(['S001'],'version','hash')
        self.assertEqual(schema['properties']['taxonomy_sha256'],{'const':'hash'})
        self.assertTrue(schema['allOf'])


if __name__=='__main__':unittest.main()
