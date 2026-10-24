"""Synthetic checks for mandatory negative/uncertain states and fixed evidence scope."""
import copy
import sys
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/data'))
from prepare_fixed_unit_check import freeze_units,expected_cells,check_fixed_response,compare_fixed_responses


def fixture():
    text='Firms report revenue. Investors read reports if firms publish them.'
    annotation={'rhetorical_taxonomy':{'sentence_structure':[
        {'sentence_text':'Firms report revenue.','clauses':[{'clause_text':'Firms report revenue'}]},
        {'sentence_text':'Investors read reports if firms publish them.','clauses':[
            {'clause_text':'Investors read reports'},{'clause_text':'if firms publish them'}]}]}}
    units=freeze_units(annotation,text)
    paragraph={'paragraph_id':'synthetic','text':text,'unit_geometry_sha256':'geometry','units':units,
        'semantic_context':{'claims':[{'claim_id':'c1'}],'relations':[]},'expected_cells':expected_cells('synthetic',units)}
    row={'paragraph_id':'synthetic','unit_geometry_sha256':'geometry','protocol_sha256':'protocol','taxonomy_sha256':'taxonomy',
        'unit_assessments':[{'unit_id':u['unit_id'],'suitability':'usable','reason':'Synthetic test unit.'} for u in units],
        'observations':[{**cell,'state':'unresolved','evidence':[],'claim_refs':[],'semantic_relation_refs':[],
                         'binding_status':'uncertain','rationale':'Synthetic test.','unit_or_binding_issues':[]} for cell in paragraph['expected_cells']],
        'coverage_notes':[]}
    return paragraph,row


class FixedUnitTests(unittest.TestCase):
    def test_exact_units_and_complete_cell_matrix(self):
        paragraph,row=fixture()
        self.assertEqual(len(paragraph['units']),7)
        self.assertEqual(len(paragraph['expected_cells']),22)
        self.assertTrue(all(paragraph['text'][u['start']:u['end']]==u['text'] for u in paragraph['units']))
        self.assertEqual(check_fixed_response(row,paragraph),[])

    def test_missing_duplicate_cells_and_cross_unit_evidence_rejected(self):
        paragraph,row=fixture();row['observations'].pop()
        self.assertIn('missing_extra_or_duplicate_cells',{e['code'] for e in check_fixed_response(row,paragraph)})
        paragraph,row=fixture();row['observations'].append(copy.deepcopy(row['observations'][0]))
        self.assertIn('missing_extra_or_duplicate_cells',{e['code'] for e in check_fixed_response(row,paragraph)})
        paragraph,row=fixture();cell=next(r for r in row['observations'] if r['unit_id']=='sentence_02')
        cell['evidence']=[{'quote':'Firms report revenue.','occurrence':1}]
        self.assertIn('evidence_outside_fixed_unit',{e['code'] for e in check_fixed_response(row,paragraph)})

    def test_uncertain_agreement_is_not_informative(self):
        _,row=fixture();report=compare_fixed_responses(row,copy.deepcopy(row))
        self.assertEqual(report['state_matches'],22)
        self.assertEqual(report['both_informative_cells'],0)
        self.assertIsNone(report['informative_agreement'])
        other=copy.deepcopy(row);row['observations'][0]['state']='present';other['observations'][0]['state']='absent'
        report=compare_fixed_responses(row,other)
        self.assertEqual(report['both_informative_cells'],1);self.assertEqual(report['both_informative_matches'],0)

    def test_ambiguous_subsentence_anchor_fails_closed(self):
        annotation={'rhetorical_taxonomy':{'sentence_structure':[{'sentence_text':'firms or firms.',
            'clauses':[{'clause_text':'firms'}]}]}}
        with self.assertRaises(ValueError):freeze_units(annotation,'firms or firms.')


if __name__=='__main__':unittest.main()
