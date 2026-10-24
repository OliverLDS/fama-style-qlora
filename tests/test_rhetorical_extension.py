"""All fixtures are synthetic; no local weights, GPU, credentials or private data."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import subprocess
import os
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.rhetoric.schema import validate, validate_links
from fama_style.rhetoric.extract import extract, paired, execute_operation, inspect_legacy
from fama_style.rhetoric.mapping import probe
from fama_style.rhetoric.oracle import jobs_for, prompt_for, structural_template, generate_jobs, validate_outputs, score, CONDITIONS
from fama_style.rhetoric.semantics import assess, calibration_controls
from fama_style.rhetoric.experiment import prepare, audit_input_ids, MAX_NEW_TOKENS, CONTEXT_LIMIT, main
from fama_style.research.run_extension import synthetic
from fama_style.research.claims import sha
from fama_style.research.signal import fit_profile


class RhetoricalSchemaTests(unittest.TestCase):
    def test_three_levels_and_semantic_separation(self):
        text='If demand falls, firms may reduce output by 5%. Investors do not trade.'
        annotation=extract(text); structure=annotation['structure']
        self.assertEqual(set(structure),{'version','paragraph','sentences','language'})
        self.assertEqual(structure['sentences'][0]['condition_position'],'front')
        self.assertTrue(validate_links(structure,annotation['semantic_ledger']))
        self.assertFalse(annotation['checks']['semantic_parse_verified'])
        self.assertNotIn('5%',json.dumps(structure))
        self.assertEqual(annotation['semantic_ledger']['claims'][0]['modality'][0]['text'],'may')

    def test_closed_schema_and_valid_claim_bindings(self):
        annotation=extract('Firms report revenue.')
        malformed=copy.deepcopy(annotation['structure']);malformed['language']['free_text']='Forbidden wording'
        with self.assertRaises(ValueError):validate(malformed)
        malformed=copy.deepcopy(annotation['structure']);malformed['paragraph']['moves'][0]['claims']=['c99']
        with self.assertRaises(ValueError):validate_links(malformed,annotation['semantic_ledger'])

    def test_extraction_independent_of_other_side(self):
        source='Firms report revenue. Investors read reports.'
        one=paired(source,'Revenue is reported by firms.')
        two=paired(source,'Another unrelated example appears.')
        self.assertEqual(one['source'],two['source'])
        self.assertEqual(one['source'],extract(source))
        self.assertFalse(one['consistency']['claim_reordering_allowed'])
        for operation in one['template_changes']:
            self.assertTrue(operation['source_claim_refs'])
            self.assertFalse(operation['introduces_semantic_relation'])

    def test_executable_operations_cannot_create_causal_relation(self):
        source='Firms report revenue. Investors read reports.'
        text=execute_operation(source,'join_independent_sentences',['c1','c2'],sha(source))
        self.assertIn(';',text)
        for op,refs,digest in [('add_causal_link',['c1','c2'],sha(source)),
                               ('join_independent_sentences',['c99'],sha(source)),
                               ('join_independent_sentences',['c1','c2'],'changed')]:
            with self.assertRaises(ValueError):execute_operation(source,op,refs,digest)

    def test_legacy_metadata_retained(self):
        rows=[{'rhetorical_taxonomy':None},{'rhetorical_taxonomy':{'R1':{'discourse_objective':'EXPLAIN'},'R2':{},'R3':{}}}]
        before=copy.deepcopy(rows);report=inspect_legacy(rows)
        self.assertEqual(rows,before);self.assertEqual(report['formats']['missing'],1)
        self.assertFalse(report['reused_as_source_labels'])

    def test_hedging_and_citation_changes_are_blocked(self):
        annotation=paired('Firms may invest.', 'Firms invest, according to a study in 2000.')
        for operation in annotation['template_changes']:
            if operation['feature'] in ('hedging','citation'):
                self.assertEqual(operation['operation'],'preserve_source_feature')
                self.assertEqual(operation['to'],operation['from'])


class OracleProtocolTests(unittest.TestCase):
    def setUp(self):
        self.train,self.validation=synthetic()
        for index,row in enumerate(self.train):row['paper_key']=str(index)
        self.profile=fit_profile(self.train)
        self.validation[0]['completion']='REFERENCE_SENTINEL_MUST_NOT_REACH_GENERATION has an unusual form.'

    def test_no_reference_wording_in_oracle_prompt_or_job(self):
        jobs=jobs_for(self.validation)
        self.assertEqual(len(jobs),3)
        for job in jobs:
            self.assertNotIn('REFERENCE_SENTINEL',json.dumps(job))
            self.assertNotIn('reference',job)
            self.assertEqual(sha(job['prompt']),job['prompt_sha256'])
        structure=extract(self.validation[0]['completion'])['structure']
        template=structural_template(structure)
        self.assertNotIn('claims',json.dumps(template));self.assertNotIn('hedging',template['language'])
        source=extract(self.validation[0]['plain_english'])
        for condition in CONDITIONS[1:]:
            prompt=prompt_for(self.validation[0]['plain_english'],condition,source,structure)
            self.assertIn('Source evidence units:',prompt)

    def test_rotation_same_examples_and_test_exclusion(self):
        validation=[]
        for index in range(3):
            row=copy.deepcopy(self.validation[0]);row['id']=f'v{index}';validation.append(row)
        jobs=jobs_for(validation)
        self.assertEqual([job['condition'] for job in jobs[0:3]],list(CONDITIONS))
        self.assertEqual(jobs[3]['condition'],CONDITIONS[1])
        validation[0]['split']='test'
        with self.assertRaises(ValueError):jobs_for(validation)

    def outputs(self):
        jobs=jobs_for(self.validation)
        def generate(job):
            self.assertEqual(set(job),{'id','split','condition','prompt','prompt_sha256','source_sha256','oracle'})
            return {'output':self.validation[0]['plain_english'],'hit_output_limit':False,'synthetic':True}
        return jobs,generate_jobs(jobs,generate)

    def test_fake_inference_is_not_a_research_result(self):
        jobs,outputs=self.outputs()
        metrics,summary=score(self.validation,jobs,outputs,self.profile,synthetic=True)
        self.assertEqual(summary['execution_kind'],'synthetic_wiring_check')
        self.assertFalse(summary['oracle_supported_by_automated_proxies'])
        self.assertEqual(summary['conditions'][CONDITIONS[0]]['exact_copy'],1)
        self.assertTrue(all(row['semantic_consistency']=='unresolved' for row in metrics))

    def test_incomplete_swapped_or_cutoff_results_rejected(self):
        jobs,outputs=self.outputs()
        bad_sets=[outputs[:-1]]
        swapped=copy.deepcopy(outputs);swapped[0]['prompt_sha256']='changed';bad_sets.append(swapped)
        cutoff=copy.deepcopy(outputs);cutoff[0]['hit_output_limit']=True;bad_sets.append(cutoff)
        duplicate=copy.deepcopy(outputs);duplicate[1]=duplicate[0];bad_sets.append(duplicate)
        for bad in bad_sets:
            with self.assertRaises(ValueError):validate_outputs(jobs,bad,True)
        calls=[]
        def generate(job):
            calls.append(job)
            return {'output':'','hit_output_limit':True,'synthetic':True}
        with self.assertRaises(ValueError):generate_jobs(jobs,generate)
        self.assertEqual(len(calls),1)

    def test_context_budget_never_truncates(self):
        job=jobs_for(self.validation)[0]
        n=CONTEXT_LIMIT-MAX_NEW_TOKENS
        self.assertEqual(audit_input_ids(job,[1]*n)['input_tokens'],n)
        with self.assertRaises(ValueError):audit_input_ids(job,[1]*(n+1))

    def test_preparation_reproducibility_and_no_secret_access(self):
        with tempfile.TemporaryDirectory() as directory:
            a,b=Path(directory)/'a',Path(directory)/'b'
            prepare(self.train,self.validation,a,True);prepare(self.train,self.validation,b,True)
            self.assertEqual((a/'manifest.json').read_bytes(),(b/'manifest.json').read_bytes())
            manifest=json.loads((a/'manifest.json').read_text())
            for name,digest in manifest['files'].items():
                self.assertEqual(hashlib.sha256((a/name).read_bytes()).hexdigest(),digest)
            with self.assertRaises(ValueError):prepare(self.train,self.validation,a,True)
            with patch('fama_style.rhetoric.experiment.secret_token',side_effect=AssertionError('Secret access forbidden')):
                main(['--dry-run','--output-dir',str(Path(directory)/'cli')])

    def test_paper_grouped_probes_are_train_only(self):
        pairs=[{'paper_key':row['paper_key'],'split':'train','annotation':paired(row['plain_english'],row['completion'])} for row in self.train]
        report=probe(pairs)
        self.assertEqual(report['papers'],3)
        self.assertEqual(set(report['leave_one_training_paper_out']),{'fixed_prior','source_identity','retrieval','smoothed_conditional'})
        pairs[0]['split']='test'
        with self.assertRaises(ValueError):probe(pairs)


class IndependentSemanticTests(unittest.TestCase):
    def test_overconfident_judge_fails_automatic_controls(self):
        always_entail=lambda a,b:{'entailment':1.,'neutral':0.,'contradiction':0.}
        result=calibration_controls(always_entail)
        self.assertFalse(result['passed']);self.assertEqual(result['rejected_mutations'],0)
        self.assertFalse(assess('Firms invest.','Firms invest.',always_entail)['semantic_proof'])

    def test_context_abstention_and_invalid_probabilities(self):
        self.assertEqual(assess('Firms invest.','Firms invest.',lambda a,b:None)['status'],'unknown_over_context')
        with self.assertRaises(ValueError):assess('Firms invest.','Firms invest.',lambda a,b:{'entailment':float('nan')})


class RhetoricalNotebookTests(unittest.TestCase):
    def test_notebook_embeds_current_source_and_runs_cpu_without_checkout(self):
        root=Path(__file__).resolve().parents[1]
        book=json.loads((root/'kaggle/rhetorical_oracle.ipynb').read_text())
        self.assertTrue(all(not cell.get('outputs') and cell.get('execution_count') is None for cell in book['cells']))
        for cell in book['cells']:
            source=''.join(cell['source'])
            if cell['cell_type']=='code' and not source.startswith('%'):compile(source,'notebook_cell','exec')
        setup=''.join(book['cells'][3]['source'])
        with tempfile.TemporaryDirectory() as directory:
            runtime=Path(directory)/'src'
            # Execute only source export, not the final inference cell.
            setup=setup.replace('/kaggle/working/_rhetorical_runtime/src',str(runtime))
            namespace={};exec(setup,namespace)
            for name,source in namespace['files'].items():
                self.assertEqual(source,(root/'src'/name).read_text())
            env={**os.environ,'PYTHONPATH':str(runtime)}
            result=subprocess.run([sys.executable,'-m','fama_style.rhetoric.experiment','--dry-run',
                                   '--output-dir',str(Path(directory)/'dry')],cwd=directory,env=env,
                                  capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('synthetic_check_complete',result.stdout)


if __name__=='__main__':unittest.main()
