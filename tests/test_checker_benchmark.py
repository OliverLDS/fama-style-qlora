"""Scoring, split isolation, freeze checks and gated prompt preparation."""
import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.research import checker_benchmark as bench
from fama_style.rhetoric import soft_guidance as soft
from fama_style.rhetoric.concise_plan import synthetic_pack
from fama_style.research.signal import fit_profile

ROOT=Path(__file__).resolve().parents[1]


class CheckerBenchmarkTests(unittest.TestCase):
    def test_degenerate_reject_and_abstain_cannot_pass(self):
        protocol=json.loads((ROOT/'docs/claim_checker_benchmark_v1/frozen_protocol.json').read_text())
        rows=json.loads((ROOT/'docs/claim_checker_benchmark_v1/held_out.json').read_text())
        rejected=bench.summarize([{**r,'prediction':'risk_signal'} for r in rows])
        abstained=bench.summarize([{**r,'prediction':'abstain'} for r in rows])
        self.assertEqual(rejected['error_detection_rate'],1);self.assertEqual(rejected['faithful_false_alarm_rate'],1)
        self.assertFalse(bench.passes_gate(rejected,protocol['readiness_gate'])['passed'])
        self.assertEqual(abstained['error_detection_rate'],0);self.assertEqual(abstained['abstention_rate'],1)
        self.assertFalse(bench.passes_gate(abstained,protocol['readiness_gate'])['passed'])

    def test_abstention_stays_in_error_denominator(self):
        rows=[{'expected':'changed','variant':'mutation','prediction':'risk_signal'},
              {'expected':'changed','variant':'mutation','prediction':'abstain'},
              {'expected':'faithful','variant':'identity','prediction':'surface_accept'},
              {'expected':'faithful','variant':'paraphrase','prediction':'abstain'}]
        m=bench.summarize(rows)
        self.assertEqual(m['error_detection_rate'],.5);self.assertEqual(m['changed_abstention_rate'],.5)
        self.assertEqual(m['nontrivial_faithful_surface_accept_rate'],0)
        self.assertEqual(m['faithful_surface_accept_rate'],.5)

    def test_frozen_evaluation_is_reproducible_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);protocol=root/'protocol';shutil.copytree(ROOT/'docs/claim_checker_benchmark_v1',protocol)
            with contextlib.redirect_stdout(io.StringIO()):
                a=bench.run(protocol,root/'a');b=bench.run(protocol,root/'b')
            self.assertEqual(a,b);self.assertEqual((root/'a/results.json').read_bytes(),(root/'b/results.json').read_bytes())
            self.assertEqual(a['model_calls'],0);self.assertEqual(a['reserved_paper_test_rows_used'],0)
            self.assertFalse(a['readiness_gate']['passed'])
            with self.assertRaises(ValueError):bench.run(protocol,root/'a')
            p=protocol/'held_out.json';p.write_text(p.read_text()+' ')
            with self.assertRaises(ValueError):bench.run(protocol,root/'c')

    def test_scenarios_are_disjoint_with_paired_positive_controls(self):
        dev=json.loads((ROOT/'docs/claim_checker_benchmark_v1/development.json').read_text())
        held=json.loads((ROOT/'docs/claim_checker_benchmark_v1/held_out.json').read_text())
        self.assertFalse({r['source'] for r in dev}&{r['source'] for r in held})
        self.assertFalse({r['scenario_group'] for r in dev}&{r['scenario_group'] for r in held})
        for rows in (dev,held):
            self.assertEqual(len(rows),36)
            for group in {r['scenario_group'] for r in rows}:
                cases=[r for r in rows if r['scenario_group']==group]
                self.assertEqual({r['variant'] for r in cases},{'identity','paraphrase','mutation'})
                self.assertEqual(sum(r['expected']=='faithful' for r in cases),2)

    def pack(self):
        profile=fit_profile([{'split':'train','plain_english':r['source'],'completion':r['reference']} for r in synthetic_pack()['rows']])
        return {'rows':[{'id':'synthetic-'+str(i),'split':'validation','source':'Firms publish reports. Investors read reports.','reference':'REFERENCE_SENTINEL'} for i in range(21)],
                'style_profile':profile,'execution_gate':{'ready':False}}

    def test_same_claims_with_source_bound_soft_plan_no_reference(self):
        pack=self.pack();jobs=soft.jobs_for(pack);self.assertEqual(len(jobs),42)
        self.assertNotIn('REFERENCE_SENTINEL',json.dumps(jobs));self.assertTrue(all(j['oracle'] is False for j in jobs))
        changed=copy.deepcopy(pack)
        for row in changed['rows']:row['reference']='DIFFERENT_REFERENCE'
        self.assertEqual(jobs,soft.jobs_for(changed))
        for row in pack['rows']:
            prompts={j['condition']:j['prompt'] for j in jobs if j['id']==row['id']}
            self.assertTrue(prompts[soft.CONDITIONS[1]].startswith(prompts[soft.CONDITIONS[0]]))
            self.assertIn('Choose the sentence count freely',prompts[soft.CONDITIONS[0]])
            self.assertIsNone(soft.soft_plan(row['source'],pack['style_profile'])['sentence_structure']['hard_sentence_count'])
        with self.assertRaises(ValueError):soft.execution_allowed(pack)
        changed['rows'][0]['split']='test'
        with self.assertRaises(ValueError):soft.jobs_for(changed)
        changed=copy.deepcopy(pack);changed['style_profile']['fit_split']='validation'
        with self.assertRaises(ValueError):soft.jobs_for(changed)


if __name__=='__main__':unittest.main()
