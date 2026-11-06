import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.research.data_scaling import nested_cohorts


class ScalingTests(unittest.TestCase):
    def rows(self):
        return [{'paper_id':g+str(i%2),'paragraph_id':str(i),'style_group':g,
                 'split':'training','training_ready':True,'source':g+' input '+str(i),
                 'target':g+' original '+str(i)} for g in ['A','B'] for i in range(4)]

    def test_nested_balanced_and_input_order_independent(self):
        rows=self.rows();a=nested_cohorts(rows,(2,4,8),('A','B'))
        b=nested_cohorts(list(reversed(rows)),(2,4,8),('A','B'))
        self.assertEqual(a,b)
        self.assertEqual(a[2],a[4][:2]);self.assertEqual(a[4],a[8][:4])
        for n,items in a.items():self.assertEqual(sum(r['style_group']=='A' for r in items),n//2)
        self.assertEqual(len({r['paper_id'] for r in a[4]}),4)

    def test_unapproved_development_and_reserved_cannot_enter(self):
        for patch in [{'training_ready':False},{'split':'development'},{'split':'reserved_test'}]:
            rows=self.rows();rows[0].update(patch)
            with self.assertRaisesRegex(ValueError,'unapproved_or_nontraining'):
                nested_cohorts(rows,(2,),('A','B'))

    def test_paraphrase_variants_do_not_inflate_target_count(self):
        rows=self.rows()[:1];variant=copy.deepcopy(rows[0]);variant['source']='Another neutral input'
        with self.assertRaisesRegex(ValueError,'insufficient_approved_unique_targets'):
            nested_cohorts(rows+[variant],(2,),('A',))

    def test_duplicate_cross_parent_targets_and_conflicting_repairs_fail(self):
        rows=self.rows();rows[1]['target']=rows[0]['target']
        with self.assertRaisesRegex(ValueError,'duplicate_target_across_parents'):
            nested_cohorts(rows,(2,),('A','B'))
        rows=self.rows();variant=copy.deepcopy(rows[0]);variant['target']='Changed target version'
        with self.assertRaisesRegex(ValueError,'conflicting_parent_versions'):
            nested_cohorts(rows+[variant],(2,),('A','B'))

    def test_new_collection_retains_previously_frozen_stage(self):
        rows=self.rows();initial=nested_cohorts(rows[:2]+rows[4:6],(4,),('A','B'))[4]
        expanded=nested_cohorts(rows,(4,8),('A','B'),frozen_prefix=initial)
        self.assertEqual(expanded[4],initial);self.assertEqual(expanded[8][:4],initial)
        changed=copy.deepcopy(initial);changed[0]['target']='A changed repaired target'
        with self.assertRaisesRegex(ValueError,'frozen_prefix'):
            nested_cohorts(rows,(4,),('A','B'),frozen_prefix=changed)


if __name__=='__main__':unittest.main()
