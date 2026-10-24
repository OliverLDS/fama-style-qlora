"""Input provenance and uncertain-cell aggregation remain explicit."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts/data'))
from audit_fixed_unit_check import match_input_round,aggregate


class FixedUnitAuditTests(unittest.TestCase):
    def test_route_by_actual_ids_not_filename(self):
        prepared={1:{'first':{}},2:{'second':{}}}
        self.assertEqual(match_input_round(['second'],prepared),2)
        with self.assertRaises(ValueError):match_input_round(['unknown'],prepared)
        with self.assertRaises(ValueError):match_input_round(['second','second'],prepared)

    def test_aggregation_uses_cell_counts_not_average_percentages(self):
        def comparison(n,m,i,im):
            return {'cells_compared':n,'state_matches':m,'both_informative_cells':i,'both_informative_matches':im,
                'both_usable_and_informative_cells':i,'both_usable_and_informative_matches':im,'state_transition_counts':{}}
        result=aggregate([comparison(33,30,31,30),comparison(22,21,22,21)])
        self.assertEqual(result['cells'],55);self.assertEqual(result['matching_states'],51)
        self.assertAlmostEqual(result['informative_agreement'],51/53)
        self.assertIsNone(result['semantic_accuracy'])
        self.assertFalse(result['general_reliability_established'])

    def test_all_uncertain_has_no_informative_score(self):
        row={'cells_compared':5,'state_matches':5,'both_informative_cells':0,'both_informative_matches':0,
             'both_usable_and_informative_cells':0,'both_usable_and_informative_matches':0,
             'state_transition_counts':{'unresolved -> unresolved':5}}
        result=aggregate([row]);self.assertEqual(result['state_agreement'],1)
        self.assertIsNone(result['informative_agreement'])


if __name__=='__main__':unittest.main()
