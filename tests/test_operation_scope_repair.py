"""Explicit excerpt boundaries preserve raw evidence and cannot waive pilot gates."""
import copy
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.rhetoric import operation_scope_repair as e

class ScopeRecoveryTests(unittest.TestCase):
    def test_rebase_resolves_occurrence_without_changing_quote(self):
        self.assertEqual(e.rebase([{'quote':'same','occurrence':2}],'same. same.',6,11),[{'quote':'same','occurrence':1}])
        with self.assertRaises(ValueError):e.rebase([{'quote':'same','occurrence':1}],'same. same.',6,11)

    def test_noncontiguous_units_rejected(self):
        record={'source':'One. Two. Three.','source_units':[{'unit_id':'S1','text':'One.'},{'unit_id':'S2','text':'Two.'},{'unit_id':'S3','text':'Three.'}]}
        with self.assertRaises(ValueError):e.excerpt(record,'source',['S1','S3'])
        text,bounds,units=e.excerpt(record,'source',['S2','S3'])
        self.assertEqual(text,'Two. Three.');self.assertEqual(record['source'][bounds['start']:bounds['end']],text)

    def test_scoped_real_records_revalidate_and_leave_parents_untouched(self):
        root=e.ROOT/'research_runs/operation_expansion_v1';path=root/'package_v1/private_key.private.json'
        if not path.exists():self.skipTest('Private annotation inputs absent')
        key=json.loads(path.read_text());infos={r['case_id']:r for r in key['records']};annotations={}
        for f in (root/'responses').glob('annotations_*.jsonl'):
            annotations.update({a['case_id']:a for a in e.read_lines(f)})
        if not all(s['case_id'] in annotations for s in e.SPECS):self.skipTest('Responses not present')
        for spec in e.SPECS:
            r=infos[spec['case_id']]['record'];a=annotations[spec['case_id']];before=copy.deepcopy((r,a))
            with self.assertRaisesRegex(ValueError,'program_omits_claims'):e.base.validate_annotation(r,a)
            result=e.recover(r,a,spec);self.assertEqual((r,a),before)
            derived=result['record'];e.base.validate_annotation(derived,result['annotation'])
            for side,key_name in [('source','source'),('target','reference')]:
                bounds=result['provenance'][side+'_scope']
                self.assertEqual(r[key_name][bounds['start']:bounds['end']],derived[key_name])
            self.assertFalse(result['provenance']['parent_paragraphs_and_raw_annotations_changed'])
            self.assertTrue(result['compiled']['packing_or_order_contrast'])
            if spec['drop_negation_claim']:
                c=next(c for c in result['annotation']['claims'] if c['claim_id']==spec['drop_negation_claim'])
                self.assertNotIn('negation',{v['kind'] for v in c['protected']})
                self.assertEqual({v['kind'] for v in c['protected']},{'quantity','uncertainty','comparison'})

if __name__=='__main__':unittest.main()
