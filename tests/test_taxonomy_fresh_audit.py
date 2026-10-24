"""Separate category assignment disagreement from differing evidence selection."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/data'))
from audit_taxonomy_discovery import labels
from audit_taxonomy_fresh_validation import check_canonical,compare_repeat,quote_span


def fixture():
    example=json.loads((ROOT/'docs/taxonomy_discovery_v1/synthetic_example.json').read_text())
    row=example['annotation'];row['coverage_gaps']=[];categories={};ids={}
    for level,dimension,label,path,owner in labels(row):
        if (level,dimension) not in ids:
            ident=f'C{len(ids)+1}';ids[(level,dimension)]=ident;categories[ident]={'level':level,'dimension':dimension}
        label.clear();label.update(canonical_id=ids[(level,dimension)],status='assigned',candidate_canonical_ids=[],reason='Synthetic format fixture.')
    return row,example['text'],categories


class FreshAuditTests(unittest.TestCase):
    def test_repeated_same_annotation_agrees_even_with_different_paragraph_id(self):
        row,text,categories=fixture();other=copy.deepcopy(row);other['paragraph_id']='repeat'
        self.assertEqual(check_canonical(row,text,categories)[0],[])
        report=compare_repeat(row,other,text)
        self.assertEqual(report['category_set_jaccard'],1)
        self.assertEqual(report['shared_anchor_groups'],report['shared_anchor_groups_with_identical_categories'])

    def test_different_evidence_is_not_reported_as_same_span_disagreement(self):
        row,text,_=fixture();other=copy.deepcopy(row)
        other['rhetorical_taxonomy']['paragraph_structure']['moves'][0]['evidence']=[{'quote':'firms','occurrence':1}]
        report=compare_repeat(row,other,text)
        self.assertEqual(report['category_set_jaccard'],1)
        self.assertEqual(report['same_anchor_category_differences'],[])
        self.assertEqual(report['base_only_anchor_groups'],1)
        self.assertEqual(report['repeat_only_anchor_groups'],1)

    def test_changed_label_on_same_evidence_and_wrong_dimension_detected(self):
        row,text,categories=fixture();other=copy.deepcopy(row)
        other['rhetorical_taxonomy']['paragraph_structure']['moves'][0]['role']['canonical_id']='NEW'
        self.assertTrue(compare_repeat(row,other,text)['same_anchor_category_differences'])
        categories['NEW']={'level':'language','dimension':'connectives'}
        self.assertIn('wrong_canonical_level_or_dimension',{r['code'] for r in check_canonical(other,text,categories)[0]})

    def test_quote_occurrence_uses_exact_position(self):
        self.assertEqual(quote_span('may may','may',2),(4,7))
        with self.assertRaises(ValueError):quote_span('may may','may',3)


if __name__=='__main__':unittest.main()
