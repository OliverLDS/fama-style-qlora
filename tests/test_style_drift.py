"""Offline tradeoff policy: warnings are evidence, not automatic vetoes."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.research import style_drift as d
from fama_style.research.signal import fit_profile


class StyleDriftTests(unittest.TestCase):
    def setUp(self):
        self.policy=json.loads((d.ROOT/'docs/style_drift_tradeoff_v1/policy.json').read_text())

    def test_minor_drift_with_substantial_style_can_outscore_unchanged_text(self):
        a=d.assess_tradeoff('moderate','minor',self.policy)
        self.assertTrue(a['substantial_style_candidate'])
        self.assertGreater(a['assessed_utilities']['bounded_drift_default'],0)
        # Even high-cost changes stay visible under the style-only sensitivity.
        b=d.assess_tradeoff('large','high',self.policy)
        self.assertEqual(b['assessed_utilities']['style_only_sensitivity'],1)

    def test_unknown_severity_is_not_treated_as_minor_or_zero(self):
        a=d.assess_tradeoff('moderate','unknown',self.policy)
        self.assertIsNone(a['assessed_drift_cost'])
        self.assertTrue(all(v is None for v in a['assessed_utilities'].values()))
        with self.assertRaises(ValueError):d.assess_tradeoff('invented','minor',self.policy)

    def test_best_alignment_avoids_contradiction_from_unrelated_candidates(self):
        good={'entailment':.98,'contradiction':.01}
        bad={'entailment':.01,'contradiction':.98}
        verdict={'global_nli':{'checks':[
            {'direction':'whole_forward','probabilities':good},
            {'direction':'whole_reverse','probabilities':good}]},
            'alignments':[{'direction':'source_coverage','candidates':[
                {'probabilities':bad},{'probabilities':good}]}]}
        self.assertAlmostEqual(d.nli_pressure(verdict)['contradiction'],.01)
        verdict['alignments'][0]['candidates'].append({'probabilities':None})
        self.assertFalse(d.nli_pressure(verdict)['complete'])

    def test_role_broadening_is_scoped_and_identity_has_no_cost_or_gain(self):
        source='Align incentives with other shareholders. Total security-holder wealth rises.'
        output='Align incentives with other security holders. Total security-holder wealth rises.'
        self.assertEqual(len(d.high_cost_evidence(source,output,None)),1)
        self.assertFalse(d.high_cost_evidence(source,source,None))
        profile=fit_profile([{'split':'train','plain_english':'Firms invest. Banks lend.',
            'completion':'Firms invest and banks lend.'}])
        score=d.score_candidate(source,source,profile,None,self.policy)
        self.assertEqual(score['semantic_cost_proxy'],0)
        self.assertEqual(score['style_alignment_benefit_proxy'],0)
        unknown=d.score_candidate(source,'Firms invest.',profile,None,self.policy)
        self.assertIsNone(unknown['semantic_cost_proxy'])

    def test_frontier_retains_risky_but_better_style_and_separates_unknown(self):
        rows=[{'candidate_id':i,'style_alignment_benefit_proxy':b,'semantic_cost_proxy':c}
              for i,b,c in [('copy',0,0),('strong',.8,.4),('worse',.1,.5),('unknown',1,None)]]
        self.assertEqual(d.frontier(rows),['copy','strong'])

    def test_assessments_bind_all_three_texts_and_require_evidence(self):
        c={'candidate_id':'id','source':'Source.','reference':'Reference.','output':'Output.'}
        a={'candidate_id':'id',**{f+'_sha256':d.sha(c[f]) for f in ('source','reference','output')},
           'style_grade':'moderate','drift_grade':'minor','style_evidence':'Reference-specific feature.',
           'drift_evidence':'Changed emphasis.'}
        p={'protocol':d.PROTOCOL,'provenance':{'assessor':'test'},'assessments':[a]}
        self.assertEqual(len(d.apply_assessments([c],p,self.policy)),1)
        p['assessments']=[a,a]
        with self.assertRaises(ValueError):d.apply_assessments([c],p,self.policy)
        p['assessments']=[a]
        changed=copy.deepcopy(c);changed['reference']='Different reference.'
        with self.assertRaises(ValueError):d.apply_assessments([changed],p,self.policy)
        a['drift_evidence']=''
        with self.assertRaises(ValueError):d.apply_assessments([c],p,self.policy)


if __name__=='__main__':unittest.main()
