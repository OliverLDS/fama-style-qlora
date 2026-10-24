"""CPU checks for first-attempt formatting, matched prompts and honest scoring."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.rhetoric import sentence_slots as slots


class SentenceSlotTests(unittest.TestCase):
    def test_closed_count_plan_no_reference_leakage(self):
        pack=slots.synthetic_pack()
        for row in pack['rows']:row['reference']='REFERENCE_SENTINEL';row['plan']['sentences']=[{'word_band':'FORBIDDEN','shape_hint':'FORBIDDEN'}]
        jobs=slots.jobs_for(pack)
        self.assertEqual(len(jobs),12)
        self.assertNotIn('REFERENCE_SENTINEL',json.dumps(jobs));self.assertNotIn('FORBIDDEN',json.dumps(jobs))
        for row in pack['rows']:
            group={j['condition']:j for j in jobs if j['id']==row['id']}
            a,b,c=(group[k]['prompt'] for k in slots.CONDITIONS)
            self.assertTrue(b.startswith(a));self.assertTrue(c.startswith(b))
            self.assertNotIn('semicolon',b);self.assertNotIn('words.',b)
            self.assertIn('exactly '+str(row['plan']['sentence_count']),b)
        changed=copy.deepcopy(pack);changed['rows'][0]['plan']['sentence_count']=6
        self.assertEqual(slots.jobs_for(pack)[0],slots.jobs_for(changed)[0])

    def test_format_failures_are_not_repaired(self):
        c='C_sentence_slots'
        good=json.dumps({'sentences':['Firms report sales.','Investors read reports.']})
        self.assertTrue(slots.realize(good,c,2)['format_valid'])
        for bad in ('```json\n'+good+'\n```',good+' commentary','{"sentences":["One."],"extra":1}',
                    '{"sentences":["One."],"sentences":["Two."]}',
                    '{"sentences":["One. Two.","Three."]}',
                    '{"sentences":["Unfinished","Complete."]}',
                    '{"sentences":["One."]}','{"sentences":[2,"Two."]}'):
            self.assertFalse(slots.realize(bad,c,2)['format_valid'],bad)
        self.assertFalse(slots.realize('{"sentences":["One.","Two. Three."]}',c,2)['format_valid'])

    def make_outputs(self):
        pack=slots.synthetic_pack();jobs=slots.jobs_for(pack);rows={r['id']:r for r in pack['rows']}
        outputs=[]
        for j in jobs:
            text=rows[j['id']]['reference']
            if j['condition']=='C_sentence_slots':text=json.dumps({'sentences':slots.sentences(text)})
            outputs.append({'id':j['id'],'condition':j['condition'],'prompt_sha256':j['prompt_sha256'],
                'output':text,'hit_output_limit':False,'synthetic':True})
        return pack,jobs,outputs

    def test_valid_semantics_does_not_hide_format_failure(self):
        pack,jobs,outputs=self.make_outputs();out=next(o for o in outputs if o['condition']=='C_sentence_slots');out['output']='```json\n'+out['output']+'\n```'
        semantic={(o['id'],o['condition']):{'status':'proxy_consistent'} for o in outputs}
        records,summary=slots.evaluate(pack,jobs,outputs,semantic,calibration={'passed':True},synthetic=True)
        group=summary['conditions']['C_sentence_slots']
        self.assertEqual(group['rows'],4);self.assertEqual(group['format_valid'],3)
        self.assertEqual(group['format_count_semantic_proxy_passes'],3)
        self.assertIsNone(next(r for r in records if not r['format_valid'])['style_gain_proxy'])
        _,summary=slots.evaluate(pack,jobs,outputs,semantic,calibration={'passed':False},synthetic=True)
        self.assertEqual(summary['conditions']['A_ordinary']['format_count_semantic_proxy_passes'],0)
        self.assertIsNone(summary['conditions']['A_ordinary']['nli_proxy_passes_valid_only'])

    def test_output_identity_and_cutoff_are_enforced(self):
        pack,jobs,outputs=self.make_outputs()
        for field,value in [('hit_output_limit',True),('synthetic',False),('prompt_sha256','wrong')]:
            altered=copy.deepcopy(outputs);altered[0][field]=value
            with self.assertRaises(ValueError):slots.evaluate(pack,jobs,altered,synthetic=True)
        altered=copy.deepcopy(outputs);altered[1]=altered[0]
        with self.assertRaises(ValueError):slots.evaluate(pack,jobs,altered,synthetic=True)

    def test_standalone_notebook_exports_and_dry_runs(self):
        root=Path(__file__).resolve().parents[1]
        book=json.loads((root/'kaggle/sentence_slot_realization.ipynb').read_text())
        self.assertTrue(all(not c.get('outputs') for c in book['cells']))
        setup=next(''.join(c['source']) for c in book['cells'] if c['cell_type']=='code' and 'files = json.loads' in ''.join(c['source']))
        with tempfile.TemporaryDirectory() as tmp:
            runtime=Path(tmp)/'src';namespace={}
            exec(setup.replace('/kaggle/working/_sentence_slot_runtime/src',str(runtime)),namespace)
            for name,source in namespace['files'].items():self.assertEqual(source,(root/'src'/name).read_text())
            result=subprocess.run([sys.executable,'-m','fama_style.rhetoric.sentence_slots','--dry-run','--output-dir',str(Path(tmp)/'dry')],
                cwd=tmp,env={**os.environ,'PYTHONPATH':str(runtime),'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('12 jobs; no model calls',result.stdout)


if __name__=='__main__':unittest.main()
