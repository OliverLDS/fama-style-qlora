"""Optimizer correctness, train-only weighting and nested paper isolation."""
import copy
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.research import boundary_classifier as module


class BoundaryClassifierTests(unittest.TestCase):
    def test_gradient_and_hessian_agree_with_finite_differences(self):
        data=[([(0,1),(1,.2),(2,.8)],1,2), ([(0,1),(1,.7)],0,1)]
        weights=[-.3,.4,-.2];penalty=.1;epsilon=1e-5
        _,g,h=module.objective(weights,data,penalty)
        for i in range(3):
            plus=weights.copy();minus=weights.copy();plus[i]+=epsilon;minus[i]-=epsilon
            lp,gp,_=module.objective(plus,data,penalty);lm,gm,_=module.objective(minus,data,penalty)
            self.assertAlmostEqual(g[i],(lp-lm)/(2*epsilon),places=7)
            for j in range(3):self.assertAlmostEqual(h[j][i],(gp[j]-gm[j])/(2*epsilon),places=7)

    def test_intercept_is_unpenalized_and_single_class_is_finite(self):
        rows=[{'features':[0,0],'label':y} for y in [0,0,0,1]]
        model=module.fit(rows,1,'none')
        self.assertAlmostEqual(module.probability(model,[0,0]),.25,places=6)
        self.assertAlmostEqual(model['weights'][0],math.log(1/3),places=6)
        model=module.fit(rows[:3],.1,'balanced')
        self.assertTrue(model['single_class_smoothed_constant'])
        self.assertTrue(0<module.probability(model,[0,0])<1)

    def test_regularization_gives_finite_separable_fit(self):
        rows=[{'features':[x],'label':y} for x,y in [(0,0),(.1,0),(.9,1),(1,1)]]
        model=module.fit(rows,.01,'balanced')
        self.assertTrue(model['converged'])
        self.assertLess(module.probability(model,[0]),.5)
        self.assertGreater(module.probability(model,[1]),.5)
        strong=module.fit(rows,1,'balanced')
        self.assertLess(abs(strong['weights'][1]),abs(model['weights'][1]))

    def test_weighting_uses_only_supplied_training_labels(self):
        self.assertEqual(module.sample_weights([0,0,0,1],'balanced'),[2/3,2/3,2/3,2])
        self.assertEqual(module.sample_weights([0,1],'none'),[1,1])

    def examples(self):
        return [{'source':{'paragraph_id':str(p)+'_'+str(y),'paper_id':str(p),
            'sentences':['Firms buy goods.','This supports output.' if y else 'Investors hold capital.']},'labels':[y]}
            for p in range(4) for y in (0,1)]

    def small_config(self):
        return {**module.CONFIG,'penalties':[.1],'class_weights':['none'],'thresholds':[.5]}

    def test_outer_paper_cannot_enter_inner_tuning(self):
        examples=self.examples()
        with self.assertRaisesRegex(ValueError,'outer_paper_in_tuning'):module.tune(examples,'0')
        with patch.object(module,'CONFIG',self.small_config()):
            _,_,traces=module.tune([e for e in examples if e['source']['paper_id']!='0'],'0')
        for t in traces:
            self.assertNotIn('0',t['training_papers'])
            self.assertNotIn(t['inner_paper'],t['training_papers'])

    def test_outer_reference_labels_do_not_affect_selection_or_predictions(self):
        examples=self.examples();changed=copy.deepcopy(examples)
        for e in changed:
            if e['source']['paper_id']=='0':e['labels']=[1-e['labels'][0]]
        with patch.object(module,'CONFIG',self.small_config()):
            original,folds=module.evaluate(examples);mutated,other=module.evaluate(changed)
        f=next(f for f in folds if f['outer_paper']=='0');g=next(f for f in other if f['outer_paper']=='0')
        self.assertEqual(f['selected'],g['selected']);self.assertEqual(f['outer_model'],g['outer_model'])
        self.assertEqual([p['probabilities'] for p in original if p['paper_id']=='0'],
                         [p['probabilities'] for p in mutated if p['paper_id']=='0'])

    def test_invalid_fit_and_prediction_shapes_rejected(self):
        with self.assertRaises(ValueError):module.fit([],1,'none')
        with self.assertRaises(ValueError):module.fit([{'features':[math.nan],'label':0}],1,'none')
        with self.assertRaises(ValueError):module.probability({'weights':[0,1]},[])

    def test_selection_prioritizes_merge_f1_before_majority_layout_accuracy(self):
        a={'penalty':.1,'threshold':.5,'class_weight':'none',
           'inner_metrics':{'merge_f1':0,'merge_precision':None,'macro_paper_exact_rate':.9}}
        b=copy.deepcopy(a);b['inner_metrics']={'merge_f1':.4,'merge_precision':.3,'macro_paper_exact_rate':.2}
        self.assertGreater(module.selection_key(b),module.selection_key(a))
