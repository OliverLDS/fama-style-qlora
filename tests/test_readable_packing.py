"""Source-only readable rendering, historical control identity and inference wiring."""
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fama_style.rhetoric import readable_packing as e


class ReadablePackingTests(unittest.TestCase):
    def pack(self):
        source = 'Firms may invest. Prices may rise.'
        evidence = [('Firms may invest.', 0), ('Prices may rise.', 18)]
        claims = [{'claim_id': 'c' + str(i + 1), 'composite': False, 'epistemic_status': 'hypothesis',
                   'source_spans': [{'quote': q, 'start': s, 'end': s + len(q)}],
                   'protected': [{'kind': 'uncertainty', 'source_spans': [{'quote': 'may', 'start': s + q.index('may'), 'end': s + q.index('may') + 3}]}]}
                  for i, (q, s) in enumerate(evidence)]
        payload = {'source': source, 'source_units': [{'unit_id': 'S' + str(i + 1), 'text': q} for i, (q, s) in enumerate(evidence)],
                   'claim_index': claims, 'source_relations': [],
                   'edits': [{'op': 'MERGE', 'source_unit_ids': ['S1', 'S2'], 'output_slots': [{'claim_refs': ['c1', 'c2']}]}]}
        rows = [{'id': c, 'source_id': str(i), 'paper_key': 'paper' + str(i), 'split': 'train', 'source': source,
                 'reference': 'PRIVATE_REFERENCE_SENTINEL', 'source_sha256': e.sha(source), 'reference_sha256': e.sha('PRIVATE_REFERENCE_SENTINEL'),
                 'use': 'new_oracle_candidate', 'payload': copy.deepcopy(payload)} for i, c in enumerate(e.CASE_IDS)]
        return {'rows': rows, 'style_profile': e.parent.fit_profile([{'split': 'train', 'plain_english': source, 'completion': 'Firms may invest and prices may rise.'}])}

    def outputs(self, pack, synthetic=True):
        return [{**{k: j[k] for k in ('id', 'model', 'condition', 'prompt_sha256')}, 'output': 'Firms may invest and prices may rise.',
                 'synthetic': synthetic, 'hit_output_limit': False} for j in e.jobs_for(pack)]

    def test_historical_control_identity_and_reference_exclusion(self):
        pack = self.pack(); before = copy.deepcopy(pack); jobs = e.jobs_for(pack)
        self.assertEqual(len(jobs), 4); self.assertEqual(pack, before)
        for row in pack['rows']:
            a = next(j for j in jobs if j['id'] == row['id'] and j['condition'] == e.CONDITIONS[0])
            old = next(j for j in e.parent.jobs_for({'rows': [row]}) if j['condition'] == e.parent.CONDITIONS[1])
            self.assertEqual(a['prompt'], old['prompt']); self.assertEqual(a['prompt_sha256'], old['prompt_sha256'])
        changed = copy.deepcopy(pack)
        for r in changed['rows']: r['reference'] = 'CHANGED_REFERENCE'; r['evaluation_only'] = 'PRIVATE_TARGET_NOTES'
        self.assertEqual(jobs, e.jobs_for(changed))
        self.assertTrue(all('PRIVATE_' not in j['prompt'] and j['prompt'].startswith(e.parent.COMMON) for j in jobs))
        self.assertEqual([j['condition'] for j in jobs], [e.CONDITIONS[0], e.CONDITIONS[1], e.CONDITIONS[1], e.CONDITIONS[0]])

    def test_readable_evidence_is_exact_and_retains_bindings(self):
        payload = self.pack()['rows'][0]['payload']; before = copy.deepcopy(payload)
        rendered = e.readable_source(payload)
        self.assertEqual(rendered['source_units'], payload['source_units'])
        self.assertEqual(rendered['source_relations'], payload['source_relations'])
        for c, original in zip(rendered['claim_index'], payload['claim_index']):
            self.assertEqual(c['source_evidence'][0]['quote'], original['source_spans'][0]['quote'])
            self.assertEqual(c['protected'][0]['source_evidence'][0]['quote'], 'may')
            self.assertEqual(c['epistemic_status'], 'hypothesis')
        self.assertEqual([c['source_unit_ids'] for c in rendered['claim_index']], [['S1'], ['S2']])
        self.assertEqual(payload, before)
        self.assertEqual(e.quoted('same same', [{'quote': 'same', 'start': 5, 'end': 9}]), [{'quote': 'same', 'occurrence': 2}])
        with self.assertRaises(ValueError): e.quoted('same', [{'quote': 'changed', 'start': 0, 'end': 4}])
        with self.assertRaises(ValueError): e.quoted('same', [{'quote': 'same', 'start': -1, 'end': 4}])

    def test_direct_plan_preserves_slots_and_rejects_incomplete_claims(self):
        payload = self.pack()['rows'][0]['payload']; text = e.readable_plan(payload)
        self.assertIn('exactly 1 complete sentences', text)
        self.assertIn('MERGE on source sentence(s) S1, S2', text)
        self.assertIn('Output sentence 1: express whole claim(s) c1, c2.', text)
        bad = copy.deepcopy(payload); bad['edits'][0]['output_slots'][0]['claim_refs'] = ['c1']
        with self.assertRaises(ValueError): e.readable_plan(bad)
        bad = copy.deepcopy(payload); bad['edits'][0]['op'] = 'INVENT_CAUSE'
        with self.assertRaises(ValueError): e.readable_plan(bad)
        bad = copy.deepcopy(payload); bad['edits'][0]['source_unit_ids'] = ['S9']
        with self.assertRaises(ValueError): e.readable_plan(bad)

    def test_failed_attempts_remain_in_paired_results(self):
        pack = self.pack(); outputs = self.outputs(pack); outputs[0].update(output='Partial', hit_output_limit=True)
        rows, summary = e.evaluate(pack, outputs, synthetic=True)
        self.assertFalse(rows[0]['available']); self.assertIsNone(rows[0]['style_distance_gain_proxy'])
        self.assertEqual(rows[0]['output'], 'Partial'); self.assertEqual(summary['generation_calls'], 4)
        self.assertFalse(summary['paired'][0]['both_available']); self.assertIsNone(summary['paired'][0]['style_proxy_B_minus_A'])
        self.assertEqual(sum(r['attempts'] for r in summary['groups'].values()), 4)
        with self.assertRaises(ValueError): e.evaluate(pack, outputs[:-1], synthetic=True)
        outputs[1] = outputs[0]
        with self.assertRaises(ValueError): e.evaluate(pack, outputs, synthetic=True)

    def test_semantic_evidence_is_bound_and_not_an_acceptance_gate(self):
        pack = self.pack(); outputs = self.outputs(pack, False)
        with self.assertRaises(ValueError): e.evaluate(pack, outputs)
        evidence = {e.job_key(o): {'source_sha256': pack['rows'][0]['source_sha256'], 'output_sha256': e.sha(o['output']),
                                  'verifier': {'status': 'risk_signal', 'nested': ['retained']}, 'operation_diagnostics': {'status': 'unresolved'}} for o in outputs}
        rows, summary = e.evaluate(pack, outputs, evidence)
        self.assertEqual(rows[0]['verifier']['nested'], ['retained']); self.assertIsNone(rows[0]['acceptable_claim_drift'])
        self.assertIsNone(summary['author_style_accuracy'])
        evidence[e.job_key(outputs[0])]['output_sha256'] = 'wrong'
        with self.assertRaises(ValueError): e.evaluate(pack, outputs, evidence)

    def test_runtime_and_decoding_match_frozen_parent_without_training(self):
        def function(module, name):
            return next(n for n in ast.parse(Path(module.__file__).read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
        self.assertEqual(ast.dump(function(e, 'run_training'), include_attributes=False), ast.dump(function(e.parent, 'run_training'), include_attributes=False))
        self.assertEqual(e.CONFIG, e.parent.CONFIG); self.assertIs(e.secret_token, e.parent.secret_token)
        self.assertEqual(e.MODEL_REVISION, e.parent.MODEL_REVISION)
        with self.assertRaises(ValueError): e.audit_tokens(e.jobs_for(self.pack())[0], [1] * e.CONFIG['context_limit'])
        self.assertNotIn('import torch', Path(e.__file__).read_text().split('def run_training(')[0])
        calls = [n for n in ast.walk(ast.parse(Path(e.__file__).read_text())) if isinstance(n, ast.Call)]
        self.assertFalse(any(isinstance(n.func, ast.Attribute) and n.func.attr in ('backward', 'step', 'train', 'push_to_hub') for n in calls))

    def test_real_pack_retains_original_rows_programs_and_prompts(self):
        path = e.ROOT / 'research_runs/readable_packing_ablation_v1/input_pack_v3/readable_pack.json'
        if not path.exists(): self.skipTest('Private inputs absent')
        pack = e.load_pack(path, e.file_sha(path))
        oldpath = e.ROOT / 'research_runs/elementary_packing_realization_v1/input_pack/elementary_pack.json'
        old = e.parent.load_pack(oldpath, e.file_sha(oldpath))
        self.assertEqual(pack['rows'], [next(r for r in old['rows'] if r['id'] == c) for c in e.CASE_IDS])
        for row in pack['rows']:
            source = e.readable_source(row['payload']); self.assertEqual(source['source'], row['source'])
            self.assertNotIn(row['reference'], next(j['prompt'] for j in e.jobs_for(pack) if j['id'] == row['id'] and j['condition'] == e.CONDITIONS[1]))
        split = next(r for r in pack['rows'] if any(op['op'] == 'SPLIT' for op in r['payload']['edits']))
        self.assertIn('exactly 8 complete sentences', e.readable_plan(split['payload']))
        self.assertIn('Output sentence 2: express whole claim(s) c2.', e.readable_plan(split['payload']))
        self.assertIn('Output sentence 3: express whole claim(s) c3.', e.readable_plan(split['payload']))

    def test_notebook_transport_size_and_isolated_cpu_dry_run(self):
        path = e.ROOT / 'kaggle/readable_packing.ipynb'
        if not path.exists(): self.skipTest('Notebook not yet built')
        book = json.loads(path.read_text()); self.assertLess(path.stat().st_size, 900000)
        self.assertLess(sum(len(''.join(c['source']).encode()) for c in book['cells']), 900000)
        self.assertTrue(all(not c.get('outputs') and c.get('execution_count') is None for c in book['cells']))
        packpath = e.ROOT / 'research_runs/readable_packing_ablation_v1/input_pack_v3/readable_pack.json'
        if not packpath.exists(): self.skipTest('Private input pack absent')
        with tempfile.TemporaryDirectory() as folder:
            setup = ''.join(book['cells'][3]['source']).replace('/kaggle/working/_readable_packing_runtime', folder + '/runtime')
            namespace = {}; exec(compile(setup, 'notebook-setup', 'exec'), namespace)
            for name, text in namespace['files'].items(): self.assertEqual((e.ROOT / name).read_text(), text)
            command = [sys.executable, '-c', 'import sys; sys.path.insert(0,sys.argv[1]); from fama_style.rhetoric.readable_packing import main; main(sys.argv[2:]); assert not any(n in sys.modules for n in ("torch", "transformers", "peft"))',
                       str(Path(folder) / 'runtime/src'), '--pack', str(packpath), '--expected-pack-sha', e.file_sha(packpath), '--dry-run', '--output-dir', str(Path(folder) / 'dry')]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((Path(folder) / 'dry/synthetic_evaluation.json').read_text())
            self.assertEqual(summary['generation_calls'], 4); self.assertEqual(summary['status'], 'synthetic_wiring_check')


if __name__ == '__main__': unittest.main()
