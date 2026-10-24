"""CPU checks for source-only constraints, matched factors and calibrated scoring."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.rhetoric import claim_constrained as exp


class ClaimConstrainedTests(unittest.TestCase):
    def test_ledger_keeps_evidence_and_direction_without_inventing_claims(self):
        source='Trading may decline when costs do not exceed 5 percent. Some firms do not trade.'
        result=exp.constraint_ledger(source)
        self.assertEqual(' '.join(c['source_evidence'] for c in result['claims']),source)
        cues=[x for c in result['claims'] for x in c['protected_evidence']]
        self.assertIn({'kind':'upper_bound','evidence':'do not exceed'},cues)
        self.assertTrue(all(x['evidence'] in source for x in cues))
        self.assertNotIn('lower_bound',[x['kind'] for x in cues])
        changed=exp.constraint_ledger('Dependence does not fall below the threshold.')
        self.assertIn('lower_bound',[x['kind'] for c in changed['claims'] for x in c['protected_evidence']])

    def test_factorial_prompts_have_matched_constraints_and_no_reference_leak(self):
        pack=exp.synthetic_pack()
        for r in pack['rows']:r['reference']='REFERENCE_SENTINEL';r['plan']['sentences']=[{'word_band':'FORBIDDEN','shape_hint':'FORBIDDEN'}]
        jobs=exp.jobs_for(pack);self.assertEqual(len(jobs),16)
        self.assertNotIn('REFERENCE_SENTINEL',json.dumps(jobs));self.assertNotIn('FORBIDDEN',json.dumps(jobs))
        for r in pack['rows']:
            group={j['condition']:j['prompt'] for j in jobs if j['id']==r['id']}
            a,b,c,d=(group[k] for k in exp.CONDITIONS)
            self.assertTrue(b.startswith(a));self.assertTrue(c.startswith(a));self.assertTrue(d.startswith(b))
            self.assertEqual(c[len(a):],d[len(b):])
            ledger=json.dumps(exp.constraint_ledger(r['source']),ensure_ascii=False,separators=(',',':'))
            self.assertIn(ledger,b);self.assertIn(ledger,d);self.assertNotIn(ledger,a);self.assertNotIn(ledger,c)

    def test_only_declared_fence_normalization_is_allowed(self):
        raw='```json\n'+json.dumps({'sentences':['Firms invest.','Investors save.']})+'\n```'
        r=exp.realize(raw,'D_claims_slots',2)
        self.assertTrue(r['format_valid']);self.assertFalse(r['raw_format_valid']);self.assertTrue(r['outer_fence_removed'])
        self.assertEqual(r['text'],'Firms invest. Investors save.')
        for bad in (raw+' explanation','explanation\n'+raw,raw.replace('```json','```'),
                    '```json\n{"sentences":["One.","Two."],"extra":1}\n```'):
            self.assertFalse(exp.realize(bad,'D_claims_slots',2)['format_valid'])
        self.assertFalse(exp.realize(raw,'D_claims_slots',3)['format_valid'])
        self.assertFalse(exp.realize(raw,'B_claims_paragraph',2)['format_valid'])

    def fixtures(self):
        pack=exp.synthetic_pack();jobs=exp.jobs_for(pack);rows={r['id']:r for r in pack['rows']};outputs=[]
        for j in jobs:
            text=rows[j['id']]['reference']
            if j['condition'].endswith('_slots'):text='```json\n'+json.dumps({'sentences':exp.sentences(text)})+'\n```'
            outputs.append({'id':j['id'],'condition':j['condition'],'prompt_sha256':j['prompt_sha256'],'output':text,'synthetic':True,'hit_output_limit':False})
        return pack,jobs,outputs

    def test_score_calibration_and_identity_guards(self):
        pack,jobs,outputs=self.fixtures();semantic={(o['id'],o['condition']):{'status':'proxy_consistent'} for o in outputs}
        records,summary=exp.evaluate(pack,jobs,outputs,semantic,calibration={'passed':True},synthetic=True)
        c=summary['conditions']['D_claims_slots'];self.assertEqual(c['format_valid'],4);self.assertEqual(c['raw_format_valid'],0)
        self.assertEqual(c['format_count_semantic_proxy_passes'],4)
        _,summary=exp.evaluate(pack,jobs,outputs,semantic,calibration={'passed':False},synthetic=True)
        self.assertIsNone(summary['conditions']['D_claims_slots']['format_count_semantic_proxy_passes'])
        bad=copy.deepcopy(outputs);bad[0]['hit_output_limit']=True
        with self.assertRaises(ValueError):exp.evaluate(pack,jobs,bad,synthetic=True)
        bad=copy.deepcopy(outputs);bad[0]['prompt_sha256']='swapped'
        with self.assertRaises(ValueError):exp.evaluate(pack,jobs,bad,synthetic=True)
        bad=copy.deepcopy(jobs);bad[0]['prompt']='changed'
        with self.assertRaises(ValueError):exp.evaluate(pack,bad,outputs,synthetic=True)

    def test_targeted_controls_reject_insensitive_checker(self):
        def always_entails(a,b):return {'entailment':1.,'neutral':0.,'contradiction':0.}
        checks=exp.calibration_controls(always_entails)
        self.assertFalse(checks['passed']);self.assertEqual(checks['targeted_mutations_rejected'],0)
        self.assertIn('upper_bound_reversal',checks['targeted_mutation_checks'])
        def abstain(a,b):return None
        self.assertFalse(exp.calibration_controls(abstain)['passed'])

    def test_standalone_export_and_cpu_dry_run(self):
        root=Path(__file__).resolve().parents[1];book=json.loads((root/'kaggle/claim_constrained_realization.ipynb').read_text())
        self.assertTrue(all(not c.get('outputs') for c in book['cells']))
        setup=next(''.join(c['source']) for c in book['cells'] if c['cell_type']=='code' and 'files = json.loads' in ''.join(c['source']))
        with tempfile.TemporaryDirectory() as tmp:
            runtime=Path(tmp)/'src';namespace={}
            exec(setup.replace('/kaggle/working/_claim_constrained_runtime/src',str(runtime)),namespace)
            for name,source in namespace['files'].items():self.assertEqual(source,(root/'src'/name).read_text())
            result=subprocess.run([sys.executable,'-m','fama_style.rhetoric.claim_constrained','--dry-run','--output-dir',str(Path(tmp)/'dry')],cwd=tmp,
                env={**os.environ,'PYTHONPATH':str(runtime),'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr);self.assertIn('16 jobs; no model calls',result.stdout)


if __name__=='__main__':unittest.main()
