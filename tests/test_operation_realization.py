"""Matched prompt/evidence boundaries, failure honesty and inference-only CPU checks."""
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.rhetoric import operation_realization as e

class OperationRealizationTests(unittest.TestCase):
    def pack(self):
        source='Firms may invest. Prices may rise.';reference='Firms may invest. Prices might increase.'
        rows=[]
        for i in range(5):
            claims=[{'claim_id':'c1','source_spans':[{'quote':'Firms may invest.','start':0,'end':17}],'epistemic_status':'hypothesis','protected_evidence':[]},
                {'claim_id':'c2','source_spans':[{'quote':'Prices may rise.','start':18,'end':34}],'epistemic_status':'hypothesis','protected_evidence':[]}]
            payload={'source':source,'source_claim_index':claims,'source_facts':[],'source_relations':[],
                'program':{'operation':'separate_premise_inference_qualification','stages':[{'role':'premise','source_claim_refs':['c1']},{'role':'supported_inference','source_claim_refs':['c2']}]}}
            rows.append({'id':str(i),'use':'contrast_hypothesis' if i<4 else 'preservation_control','source':source,'reference':reference,
                'source_sha256':e.sha(source),'operation':payload['program']['operation'],'payload':payload,'shallow_plan':e.previous.presentation_plan(reference)})
        return {'rows':rows,'style_profile':e.fit_profile([{'split':'train','plain_english':source,'completion':reference}]*2)}

    def outputs(self,pack,synthetic=True):
        return [{**{k:j[k] for k in ('id','model','condition','prompt_sha256')},'output':'Firms may invest. Prices may rise.','hit_output_limit':False,'synthetic':synthetic} for j in e.jobs_for(pack)]

    def test_identical_source_scaffold_and_reference_exclusion(self):
        p=self.pack();jobs=e.jobs_for(p);self.assertEqual(len(jobs),30)
        for model in e.MODELS:
            for row in p['rows']:
                group=[j for j in jobs if j['model']==model and j['id']==row['id']]
                prefix=e.source_prefix(row['payload'])
                self.assertTrue(all(j['prompt'].startswith(prefix) for j in group))
                self.assertEqual(len({j['condition'] for j in group}),3)
        modified=copy.deepcopy(p)
        for row in modified['rows']:row['reference']='PRIVATE_REFERENCE_SENTINEL';row['target_sentence_claim_groups_evaluation_only']=['PRIVATE_TARGET_NOTE']
        self.assertEqual(jobs,e.jobs_for(modified))
        self.assertTrue(all('PRIVATE_' not in j['prompt'] for j in jobs))

    def test_all_attempts_include_cutoffs_control_separate(self):
        p=self.pack();outs=self.outputs(p);outs[0]['hit_output_limit']=True;outs[0]['output']='Partial result'
        records,summary=e.evaluate(p,outs,synthetic=True)
        self.assertEqual(len(records),30);self.assertFalse(records[0]['available']);self.assertIsNone(records[0]['verifier'])
        self.assertEqual(records[0]['output'],'Partial result')
        self.assertEqual(sum(r['attempts'] for r in summary['groups']['contrast_hypothesis'].values()),24)
        self.assertEqual(sum(r['attempts'] for r in summary['groups']['preservation_control'].values()),6)
        with self.assertRaises(ValueError):e.evaluate(p,outs[:-1],synthetic=True)
        outs[1]=outs[0]
        with self.assertRaises(ValueError):e.evaluate(p,outs,synthetic=True)

    def test_full_evidence_bound_to_exact_source_and_output(self):
        p=self.pack();outs=self.outputs(p,False);evidence={e.job_key(o):{'source_sha256':e.sha(p['rows'][0]['source']),'output_sha256':e.sha(o['output']),
            'verifier':{'status':'risk_signal','nested':['evidence']},'operation_diagnostics':{'status':'unresolved'}} for o in outs}
        records,summary=e.evaluate(p,outs,evidence)
        self.assertEqual(records[0]['verifier']['nested'],['evidence']);self.assertIsNone(summary['semantic_accuracy'])
        evidence[e.job_key(outs[0])]['output_sha256']='wrong'
        with self.assertRaises(ValueError):e.evaluate(p,outs,evidence)

    def test_localization_abstains_on_unknown_and_keeps_composite_limit(self):
        row=self.pack()['rows'][0]
        row['payload']['program']['stages'].append({'role':'criterion_definition','source_claim_refs':['c2']})
        result=e.operation_diagnostics(row,'Firms may invest. Prices may rise.',lambda a,b:None)
        self.assertFalse(result['rhetorical_adherence_verified']);self.assertEqual(result['composite_claims_used_by_multiple_stages'],['c2'])
        self.assertTrue(all(c['localized_position'] is None for c in result['claim_sentence_scores']))
        self.assertTrue(all(s['status']=='unresolved' for s in result['stage_positions']))
        with self.assertRaises(ValueError):e.audit_tokens(e.jobs_for(self.pack())[0],[1]*4096)

    def test_adapter_bytes_and_revision_checked_without_loading_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            directory=Path(folder)/'one/adapter_planned';directory.mkdir(parents=True)
            config={'base_model_name_or_path':e.MODEL_ID,'revision':e.MODEL_REVISION}
            (directory/'adapter_config.json').write_text(json.dumps(config));(directory/'adapter_model.safetensors').write_bytes(b'synthetic weights placeholder')
            pack={'adapter_file_hashes':{p.name:e.file_sha(p) for p in directory.iterdir()}}
            self.assertEqual(e.adapter_directory(Path(folder),pack),directory)
            (directory/'adapter_model.safetensors').write_bytes(b'changed')
            with self.assertRaises(ValueError):e.adapter_directory(Path(folder),pack)

    def test_inference_runtime_has_no_training_actions(self):
        tree=ast.parse(Path(e.__file__).read_text())
        calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)]
        self.assertFalse(any(isinstance(c.func,ast.Attribute) and c.func.attr in ('backward','step','train') for c in calls))
        self.assertFalse(any(isinstance(n,ast.Name) and n.id in ('AdamW','Trainer','LoraConfig') for n in ast.walk(tree)))

    def test_notebook_size_and_lossless_runtime_transport(self):
        path=e.ROOT/'kaggle/operation_realization.ipynb'
        book=json.loads(path.read_text())
        self.assertLess(path.stat().st_size,900_000)
        self.assertLess(sum(len(''.join(c['source']).encode('utf-8')) for c in book['cells']),900_000)
        with tempfile.TemporaryDirectory() as folder:
            setup=''.join(book['cells'][3]['source']).replace('/kaggle/working/_operation_runtime',folder+'/runtime')
            namespace={};exec(compile(setup,'notebook-setup','exec'),namespace)
            files=namespace['files']
            # The completed experiment freezes its embedded runtime. Unrelated
            # modules added later must not force rewriting its notebook or pack.
            for name,source in files.items():
                file=e.ROOT/name
                self.assertEqual(source,file.read_text())
                self.assertEqual(e.file_sha(Path(folder)/'runtime'/name),e.file_sha(file))
            packpath=e.ROOT/'research_runs/grounded_operation_realization_v1/input_pack/operation_pack.json'
            if packpath.exists():
                required=json.loads(packpath.read_text())['runtime_code_files']
                self.assertTrue(set(required)<=set(files))
                for name,digest in required.items():self.assertEqual(e.file_sha(Path(folder)/'runtime'/name),digest)

    def test_isolated_notebook_preflight_and_private_dry_run(self):
        packpath=e.ROOT/'research_runs/grounded_operation_realization_v1/input_pack/operation_pack.json'
        if not packpath.exists():self.skipTest('Private inputs absent in public checkout')
        pack=e.load_pack(packpath,e.file_sha(packpath));self.assertEqual(len(e.jobs_for(pack)),30)
        with tempfile.TemporaryDirectory() as folder,patch.object(e,'run_training',side_effect=AssertionError('GPU')),patch.object(e,'secret_token',side_effect=AssertionError('Secret')):
            e.main(['--pack',str(packpath),'--expected-pack-sha',e.file_sha(packpath),'--dry-run','--output-dir',folder+'/dry'])
            self.assertEqual(json.loads(Path(folder+'/dry/synthetic_evaluation.json').read_text())['status'],'synthetic_wiring_check')
        book=json.loads((e.ROOT/'kaggle/operation_realization.ipynb').read_text())
        for cell in book['cells']:
            self.assertFalse(cell.get('outputs'));self.assertIsNone(cell.get('execution_count'))
            if cell['cell_type']=='code' and not ''.join(cell['source']).startswith('%'):compile(''.join(cell['source']),'notebook','exec')
        with tempfile.TemporaryDirectory() as folder:
            setup=''.join(book['cells'][3]['source']).replace('/kaggle/working/_operation_runtime',folder+'/runtime')
            check=''.join(book['cells'][4]['source']).replace('list(Path("/kaggle/input").rglob("operation_pack.json"))',repr([str(packpath)])).replace('/kaggle/working/operation_preflight',folder+'/preflight')
            artifact_root=e.ROOT/'research_runs/plan_realizer_pilot_v1/kaggle_version_1'
            check=check.replace('"/kaggle/input"',repr(str(artifact_root)))
            result=subprocess.run([sys.executable,'-I','-c',setup+'\n'+check],cwd=folder,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

if __name__=='__main__':unittest.main()
