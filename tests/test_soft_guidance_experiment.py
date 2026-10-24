"""CPU contracts for matched prompts, gated loading and honest all-row scoring."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.rhetoric import soft_guidance_experiment as exp


class SoftGuidanceExperimentTests(unittest.TestCase):
    def fixtures(self):
        pack=exp.synthetic_pack();jobs=exp.jobs_for(pack);rows={r['id']:r for r in pack['rows']}
        outputs=[{'id':j['id'],'condition':j['condition'],'prompt_sha256':j['prompt_sha256'],
            'output':rows[j['id']]['source'],'hit_output_limit':False,'synthetic':False} for j in jobs]
        semantic={(o['id'],o['condition']):{'status':'model_consistent','semantic_proof':False} for o in outputs}
        return pack,jobs,outputs,semantic

    def test_matched_guidance_no_target_words_or_count(self):
        pack,jobs,_,_=self.fixtures();changed=copy.deepcopy(pack)
        for r in changed['rows']:r['reference']='REFERENCE_SENTINEL'
        self.assertEqual(jobs,exp.jobs_for(changed));self.assertNotIn('REFERENCE_SENTINEL',json.dumps(exp.jobs_for(changed)))
        self.assertEqual(len(jobs),42)
        for row in pack['rows']:
            prompts={j['condition']:j['prompt'] for j in jobs if j['id']==row['id']}
            self.assertTrue(prompts[exp.CONDITIONS[1]].startswith(prompts[exp.CONDITIONS[0]]))
            self.assertIn('Choose the sentence count freely',prompts[exp.CONDITIONS[0]])
            self.assertIsNone(exp.soft_plan(row['source'],pack['style_profile'])['sentence_structure']['hard_sentence_count'])
        changed['rows'][0]['split']='test'
        with self.assertRaises(ValueError):exp.jobs_for(changed)

    def test_format_failure_and_abstention_remain_in_denominator(self):
        pack,jobs,outputs,semantic=self.fixtures()
        a=next(o for o in outputs if o['condition']==exp.CONDITIONS[0]);a['output']='```text\n'+a['output']+'\n```'
        b=next(o for o in outputs if o['condition']==exp.CONDITIONS[1]);semantic[(b['id'],b['condition'])]={'status':'abstain','semantic_proof':False}
        records,summary=exp.evaluate(pack,jobs,outputs,semantic,canary_passed=True)
        ag=summary['conditions'][exp.CONDITIONS[0]];bg=summary['conditions'][exp.CONDITIONS[1]]
        self.assertEqual(ag['rows'],21);self.assertEqual(bg['rows'],21)
        self.assertEqual(ag['valid_formats'],20);self.assertEqual(ag['valid_semantic_proxy_passes'],20)
        self.assertEqual(bg['verifier_status_counts']['abstain'],1);self.assertEqual(bg['valid_semantic_proxy_passes'],20)
        self.assertEqual(bg['near_copies'],21);self.assertEqual(bg['joint_proxy_successes'],0)
        self.assertEqual(len(summary['paired']),21);self.assertFalse(summary['training_ready'])

    def test_missing_swapped_cutoff_or_synthetic_output_is_not_success(self):
        pack,jobs,outputs,semantic=self.fixtures()
        missing=dict(semantic);missing.pop(next(iter(missing)))
        with self.assertRaises(ValueError):exp.evaluate(pack,jobs,outputs,missing,canary_passed=True)
        for key,value in (('hit_output_limit',True),('synthetic',True),('prompt_sha256','wrong')):
            altered=copy.deepcopy(outputs);altered[0][key]=value
            with self.assertRaises(ValueError):exp.evaluate(pack,jobs,altered,semantic,canary_passed=True)
        altered=copy.deepcopy(outputs);altered[1]=altered[0]
        with self.assertRaises(ValueError):exp.evaluate(pack,jobs,altered,semantic,canary_passed=True)
        _,s=exp.evaluate(pack,jobs,outputs,semantic,canary_passed=False)
        self.assertIsNone(s['conditions'][exp.CONDITIONS[0]]['valid_semantic_proxy_passes'])

    def synthetic_contract_pack(self):
        # Metadata exercises loading only; it is not claimed to be real training data.
        p=exp.synthetic_pack();p.update(parent_training_sha256=exp.TRAIN_SHA,validation_sha256=exp.VALIDATION_SHA,
            execution_gate={'ready':True},generation={'model_id':exp.MODEL_ID,'revision':exp.MODEL_REVISION,'max_new_tokens':768,'context_limit':4096,'do_sample':False,'seed':42,'truncate':False})
        p['style_profile']['training_rows']=247
        protocol,_=exp.verifier.validate_protocol(exp.VERIFIER_FOLDER)
        p['verifier_evidence']={'protocol_sha256':exp.file_sha(exp.VERIFIER_FOLDER/'protocol.json'),'code_files':protocol['code_files'],'fixture_files':protocol['fixture_files'],
            'held_out_metrics':{'error_detection_rate':1.,'faithful_false_alarm_rate':0.,'nontrivial_faithful_acceptance_rate':1.,'changed_false_acceptance_rate':0.}}
        p['runtime_code_files']={str(f.relative_to(exp.ROOT)):exp.file_sha(f) for f in exp.runtime_paths()}
        p['jobs_sha256']=exp.sha(json.dumps(exp.jobs_for(p),sort_keys=True,ensure_ascii=False));return p

    def test_pack_gate_hash_source_and_implementation_pins(self):
        pack=self.synthetic_contract_pack()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'pack.json'
            def load(p):
                path.write_text(json.dumps(p));return exp.load_pack(path,exp.file_sha(path))
            self.assertEqual(len(load(pack)['rows']),21)
            for change in ('gate','source','code','controls','profile'):
                altered=copy.deepcopy(pack)
                if change=='gate':altered['execution_gate']['ready']=False
                if change=='source':altered['rows'][0]['source']='Different source.'
                if change=='code':altered['runtime_code_files'][next(iter(altered['runtime_code_files']))]='0'*64
                if change=='controls':altered['generation']['max_new_tokens']=512
                if change=='profile':altered['style_profile']['fit_split']='validation'
                with self.assertRaises(ValueError):load(altered)
            path.write_text(json.dumps(pack))
            with self.assertRaises(ValueError):exp.load_pack(path,'0'*64)

    def test_context_guard_and_format_checks(self):
        job=exp.jobs_for(exp.synthetic_pack())[0]
        self.assertEqual(exp.audit_tokens(job,[1,2])['input_tokens'],2)
        for ids in ([],[1]*(exp.CONTEXT_LIMIT-exp.MAX_NEW_TOKENS+1)):
            with self.assertRaises(ValueError):exp.audit_tokens(job,ids)
        for text in ('{"paragraph":"Firms invest."}','```text\nFirms invest.\n```','Firms invest.\n\nInvestors save.','- Firms invest.'):
            self.assertFalse(exp.paragraph_format(text))
        self.assertTrue(exp.paragraph_format('Firms invest; investors save.'))

    def test_self_contained_export_dry_run(self):
        root=Path(__file__).resolve().parents[1];book=json.loads((root/'kaggle/soft_guidance_validation.ipynb').read_text())
        self.assertTrue(all(not c.get('outputs') for c in book['cells']))
        setup=next(''.join(c['source']) for c in book['cells'] if c['cell_type']=='code' and 'files = json.loads' in ''.join(c['source']))
        with tempfile.TemporaryDirectory() as tmp:
            runtime=Path(tmp)/'runtime';ns={}
            exec(setup.replace('/kaggle/working/_soft_guidance_runtime',str(runtime)),ns)
            for name,source in ns['files'].items():self.assertEqual(source,(root/name).read_text())
            result=subprocess.run([sys.executable,'-m','fama_style.rhetoric.soft_guidance_experiment','--dry-run','--output-dir',str(Path(tmp)/'dry')],cwd=tmp,
                env={**os.environ,'PYTHONPATH':str(runtime/'src'),'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr);self.assertIn('42 jobs; no model calls',result.stdout)
            summary=json.loads((Path(tmp)/'dry/synthetic_evaluation.json').read_text())
            self.assertEqual(summary['execution_kind'],'synthetic_wiring_check');self.assertFalse(summary['runtime_canary_passed'])


if __name__=='__main__':unittest.main()
