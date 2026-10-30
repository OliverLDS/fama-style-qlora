"""Cutoff enforcement, author isolation, taxonomy integrity and private batch wiring."""
import copy
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.research import human_prose_annotations as module


class HumanProseAnnotationTests(unittest.TestCase):
    def registry(self):
        papers=[]
        for i,split in enumerate(['training']*12+['development']*4+['reserved_test']*4):
            papers.append({'paper_id':str(i),'publication_year':1900+i,'publication_year_checked':True,
                'publication_evidence_url':'https://example.org/publication/'+str(i),'authors':['Author '+chr(65+i)],'split':split})
        return {'protocol':module.PROTOCOL,'papers':papers,'publication_cutoff_exclusive_year':2020,'paragraphs_per_paper':5}

    def test_recent_missing_or_unverified_dates_rejected(self):
        r=self.registry();module.validate_registry(r)
        for year in (2020,2022,None,True):
            changed=copy.deepcopy(r);changed['papers'][0]['publication_year']=year
            with self.assertRaisesRegex(ValueError,'publication_after_cutoff_or_missing'):module.validate_registry(changed)
        r['papers'][0]['publication_year_checked']=False
        with self.assertRaisesRegex(ValueError,'publication_evidence_required'):module.validate_registry(r)

    def test_same_author_cannot_cross_split_and_project_authors_excluded(self):
        r=self.registry();r['papers'][-1]['authors']=r['papers'][0]['authors']
        with self.assertRaisesRegex(ValueError,'author_crosses_split'):module.validate_registry(r)
        r=self.registry();r['papers'][0]['authors']=['Kenneth French']
        with self.assertRaisesRegex(ValueError,'project_author_excluded'):module.validate_registry(r)

    def test_catalogue_view_retains_rules_even_when_shared(self):
        rows=[{'canonical_id':'P001','level':'paragraph','definition':'A premise.',
            'inclusion_rules':['shared','only first'],'exclusion_rules':['no unsupported causation'],
            'evidence_requirements':[],'preconditions':[],'prohibited_effects':[]}]
        rows.append({**copy.deepcopy(rows[0]),'canonical_id':'P002','definition':'A conclusion.',
            'inclusion_rules':['shared','only second']})
        full={'taxonomy_version':'proposed','taxonomy_sha256':'hash','status':'proposed_not_frozen','canonical_items':rows}
        compact=module.compact_catalog(full)
        self.assertEqual(module.expand_catalog(compact),rows)
        self.assertEqual(full['canonical_items'][0]['inclusion_rules'],['shared','only first'])

    def test_missing_boundaries_and_formal_notation_not_selected(self):
        text='This complete economics paragraph has enough words to express a claim. '
        row={'id':'1','text':text*4,'word_count':44}
        self.assertIsNone(module.prose_reason(row))
        for bad in ('incomplete lowercase statement.','This statement is unfinished','This equation has x = y.'):
            self.assertIsNotNone(module.prose_reason({**row,'text':bad}))

    def test_schema_requires_immutable_reference_and_uncertainty_metadata(self):
        schema=module.output_schema('tax','hash')
        self.assertFalse(schema['additionalProperties'])
        for key in ('original_paragraph','plain_english','source_schema','target_schema','semantic_ledger','source_quality','provenance'):
            self.assertIn(key,schema['required'])
        self.assertEqual(schema['properties']['provenance']['properties']['independently_verified'],{'const':False})
        self.assertNotIn('detector_score',schema['properties'])

    @unittest.skipUnless(module.CATALOG.exists(),'Private taxonomy catalogue unavailable')
    def test_full_existing_taxonomy_is_losslessly_preserved(self):
        full=json.loads(module.CATALOG.read_text());compact=module.compact_catalog(full)
        self.assertEqual(module.expand_catalog(compact),full['canonical_items'])
        self.assertEqual(len(full['canonical_items']),132)


if __name__=='__main__':unittest.main()
