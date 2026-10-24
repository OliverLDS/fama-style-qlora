"""Safe scope normalization uses synthetic clauses and never changes text or claims."""
import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/data'))
from prepare_taxonomy_consolidation_v2 import normalize_clause_ids


def fixture():
    return {'paragraph_id':'synthetic','semantic_content':{'claims':[{'claim_id':'c1'}]},
        'rhetorical_taxonomy':{'sentence_structure':[
            {'sentence_id':'s1','sentence_text':'Firms report revenue.','clauses':[
                {'clause_id':'u1','clause_text':'Firms report revenue','depends_on_clause_ids':[]}]},
            {'sentence_id':'s2','sentence_text':'Investors read reports if firms publish them.','clauses':[
                {'clause_id':'u1','clause_text':'Investors read reports','depends_on_clause_ids':[]},
                {'clause_id':'u2','clause_text':'if firms publish them','depends_on_clause_ids':['u1']}]}]}}


class ClauseNormalizationTests(unittest.TestCase):
    def test_local_ids_and_dependencies_are_renamed_together(self):
        original=fixture();before=copy.deepcopy(original)
        result,log=normalize_clause_ids(original)
        self.assertEqual(original,before)
        clauses=result['rhetorical_taxonomy']['sentence_structure'][1]['clauses']
        self.assertEqual([r['clause_id'] for r in clauses],['u2','u3'])
        self.assertEqual(clauses[1]['depends_on_clause_ids'],['u2'])
        self.assertEqual(result['semantic_content'],original['semantic_content'])
        self.assertTrue(all(r['path'].endswith(('clause_id','depends_on_clause_ids')) for r in log))
        again,changes=normalize_clause_ids(result)
        self.assertEqual(again,result);self.assertEqual(changes,[])

    def test_ambiguous_duplicates_and_nonlocal_dependencies_rejected(self):
        row=fixture();row['rhetorical_taxonomy']['sentence_structure'][1]['clauses'][1]['clause_id']='u1'
        with self.assertRaises(ValueError):normalize_clause_ids(row)
        row=fixture();row['rhetorical_taxonomy']['sentence_structure'][1]['clauses'][1]['depends_on_clause_ids']=['u99']
        with self.assertRaises(ValueError):normalize_clause_ids(row)


if __name__=='__main__':unittest.main()
