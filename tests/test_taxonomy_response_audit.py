"""Synthetic evidence/reference audit tests; no private data or jsonschema needed."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('response_audit',ROOT/'scripts/data/audit_taxonomy_discovery.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


class TaxonomyResponseAuditTests(unittest.TestCase):
    def setUp(self):
        example=json.loads((ROOT/'docs/taxonomy_discovery_v1/synthetic_example.json').read_text())
        self.row=example['annotation'];self.text=example['text']

    def test_valid_example_and_invalid_evidence(self):
        errors,counts=audit.check_annotation(self.row,self.text)
        self.assertEqual(errors,[]);self.assertGreater(counts['evidence_anchors'],0)
        changed=copy.deepcopy(self.row)
        changed['semantic_content']['claims'][0]['evidence'][0]['quote']='Unsupported invented evidence.'
        self.assertIn('evidence_not_found',{r['code'] for r in audit.check_annotation(changed,self.text)[0]})

    def test_dangling_reference_duplicate_and_coverage(self):
        changed=copy.deepcopy(self.row)
        changed['rhetorical_taxonomy']['paragraph_structure']['main_claim_refs']=['c99']
        self.assertIn('unresolved_reference',{r['code'] for r in audit.check_annotation(changed,self.text)[0]})
        changed=copy.deepcopy(self.row)
        changed['semantic_content']['claims'].append(copy.deepcopy(changed['semantic_content']['claims'][0]))
        self.assertIn('duplicate_ids',{r['code'] for r in audit.check_annotation(changed,self.text)[0]})
        changed=copy.deepcopy(self.row)
        changed['rhetorical_taxonomy']['sentence_structure'][0]['sentence_text']='Some firms may reduce investment.'
        self.assertIn('sentence_coverage_mismatch',{r['code'] for r in audit.check_annotation(changed,self.text)[0]})

    def test_inventory_deduplicates_types_and_keeps_occurrences(self):
        other=copy.deepcopy(self.row);other['paragraph_id']='other'
        inputs=[(self.row,'original_observed','pair1'),(other,'neutral_proposed_paired','pair1')]
        inventory,occurrences=audit.inventory(inputs)
        self.assertTrue(inventory)
        self.assertEqual(sum(r['occurrence_count'] for r in inventory),len(occurrences))
        self.assertTrue(all(r['distinct_pair_count']==1 for r in inventory))
        self.assertTrue(all(r['distinct_annotation_count']==2 for r in inventory))
        self.assertEqual((inventory,occurrences),audit.inventory(inputs))
        self.assertTrue(all('annotation_id' in r and 'json_path' in r for r in occurrences))


if __name__=='__main__':unittest.main()
