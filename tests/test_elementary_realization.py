"""Inference-only matrix, oracle boundaries, diagnostics and notebook transport."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fama_style.rhetoric import elementary_realization as e


class ElementaryRealizationTests(unittest.TestCase):
    def pack(self):
        source = 'Firms may invest. Prices may rise.'
        claims = [{'claim_id': 'c' + str(i + 1), 'source_spans': [{'quote': text, 'start': start, 'end': end}],
                   'composite': False, 'epistemic_status': 'hypothesis', 'protected': []}
                  for i, (text, start, end) in enumerate([(source[:17], 0, 17), (source[18:], 18, len(source))])]
        payload = {'source': source, 'source_units': [{'unit_id': 'S1', 'text': source[:17]}, {'unit_id': 'S2', 'text': source[18:]}],
                   'claim_index': claims, 'source_relations': [],
                   'edits': [{'op': 'MERGE', 'source_unit_ids': ['S1', 'S2'], 'output_slots': [{'claim_refs': ['c1', 'c2']}]}]}
        rows = [{'id': str(i), 'paper_key': 'paper' + str(i), 'source': source, 'reference': 'PRIVATE_REFERENCE_SENTINEL',
                 'source_sha256': e.sha(source), 'payload': copy.deepcopy(payload),
                 'use': 'reserved_diagnostic_probe' if i == 0 else 'new_oracle_candidate'} for i in range(6)]
        return {'rows': rows, 'style_profile': e.fit_profile([{'split': 'train', 'plain_english': source, 'completion': 'Firms may invest and prices may rise.'}])}

    def outputs(self, pack, synthetic=True):
        return [{**{k: j[k] for k in ('id', 'model', 'condition', 'prompt_sha256')}, 'output': 'Firms may invest and prices may rise.',
                 'synthetic': synthetic, 'hit_output_limit': False} for j in e.jobs_for(pack)]

    def test_matched_scaffold_reference_exclusion_and_order_balance(self):
        pack = self.pack(); jobs = e.jobs_for(pack)
        self.assertEqual(len(jobs), 12)
        firsts = [jobs[i]['condition'] for i in range(0, 12, 2)]
        self.assertEqual(firsts.count(e.CONDITIONS[0]), 3)
        for row in pack['rows']:
            group = [j for j in jobs if j['id'] == row['id']]
            self.assertTrue(all(j['prompt'].startswith(e.source_prefix(row['payload'])) for j in group))
            self.assertTrue(all('PRIVATE_REFERENCE_SENTINEL' not in j['prompt'] for j in group))
        changed = copy.deepcopy(pack)
        for row in changed['rows']: row['reference'] = 'DIFFERENT_TARGET'; row['evaluation_only'] = 'SECRET_NOTE'
        self.assertEqual(jobs, e.jobs_for(changed))
        self.assertNotIn('MERGE', next(j['prompt'] for j in jobs if j['condition'] == e.CONDITIONS[0]))

    def test_failure_denominators_probe_separation_and_no_acceptability_claim(self):
        pack = self.pack(); outputs = self.outputs(pack); outputs[0].update(hit_output_limit=True, output='Partial')
        records, summary = e.evaluate(pack, outputs, synthetic=True)
        self.assertFalse(records[0]['available']); self.assertEqual(records[0]['output'], 'Partial')
        self.assertIsNone(records[0]['source_similarity'])
        self.assertIsNone(records[1]['acceptable_claim_drift'])
        self.assertEqual(sum(g['attempts'] for g in summary['groups']['new_oracle_candidate'].values()), 10)
        self.assertEqual(sum(g['attempts'] for g in summary['groups']['reserved_diagnostic_probe'].values()), 2)
        self.assertFalse(summary['paired'][0]['both_available'])
        with self.assertRaises(ValueError): e.evaluate(pack, outputs[:-1], synthetic=True)
        outputs[1] = outputs[0]
        with self.assertRaises(ValueError): e.evaluate(pack, outputs, synthetic=True)

    def test_exact_output_bound_semantic_evidence_required(self):
        pack = self.pack(); outputs = self.outputs(pack, False)
        with self.assertRaises(ValueError): e.evaluate(pack, outputs)
        evidence = {e.job_key(o): {'source_sha256': pack['rows'][0]['source_sha256'], 'output_sha256': e.sha(o['output']),
                                  'verifier': {'status': 'risk_signal', 'evidence': ['retained']}, 'operation_diagnostics': {'status': 'unresolved'}} for o in outputs}
        records, summary = e.evaluate(pack, outputs, evidence)
        self.assertEqual(records[0]['verifier']['evidence'], ['retained'])
        self.assertIsNone(records[0]['acceptable_claim_drift'])
        evidence[e.job_key(outputs[0])]['output_sha256'] = 'different'
        with self.assertRaises(ValueError): e.evaluate(pack, outputs, evidence)

    def test_localization_is_not_sentence_count_or_fidelity_proof(self):
        row = self.pack()['rows'][0]
        unknown = e.operation_diagnostics(row, 'Firms may invest. Prices may rise.', lambda a, b: None)
        self.assertIsNone(unknown['slot_order_and_grouping_proxy'])
        self.assertFalse(unknown['rhetorical_adherence_verified'])
        def predict(sentence, claim):
            yes = ('Firms' in sentence) == ('Firms' in claim)
            return {'entailment': .95 if yes else .02, 'neutral': .03 if yes else .96, 'contradiction': .02}
        wrong = e.operation_diagnostics(row, 'Firms may invest. Prices may rise.', predict)
        self.assertFalse(wrong['slot_order_and_grouping_proxy'])
        self.assertTrue(wrong['localization_only_not_semantic_or_style_proof'])

    def test_token_budget_no_truncation_and_frozen_inference_runtime(self):
        job = e.jobs_for(self.pack())[0]
        with self.assertRaises(ValueError): e.audit_tokens(job, [1] * e.CONFIG['context_limit'])
        self.assertEqual(e.audit_tokens(job, [1, 2])['input_tokens'], 2)
        tree = ast.parse(Path(e.__file__).read_text())
        forbidden = {'backward', 'step', 'train', 'fit', 'push_to_hub'}
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in forbidden for n in ast.walk(tree)))
        self.assertFalse(any(isinstance(n, ast.Name) and n.id in {'Trainer', 'AdamW', 'LoraConfig', 'PeftModel'} for n in ast.walk(tree)))
        self.assertNotIn('import torch', Path(e.__file__).read_text().split('def run_training(')[0])

    def test_private_pack_selection_and_frozen_notebook_runtime(self):
        packpath = e.ROOT / 'research_runs/elementary_packing_realization_v1/input_pack/elementary_pack.json'
        if not packpath.exists(): self.skipTest('Private input pack unavailable')
        pack = e.load_pack(packpath, e.file_sha(packpath))
        self.assertEqual(len(pack['rows']), 6)
        self.assertEqual(sum(any(o['op'] == 'SPLIT' for o in r['payload']['edits']) for r in pack['rows']), 1)
        compiled = e.read_lines(e.ROOT / 'research_runs/elementary_packing_edits_v1/compiled_v1/oracle_candidates.private.jsonl')
        self.assertEqual([r['id'] for r in pack['rows']], [r['case_id'] for r in e.select_six(compiled)])
        bookpath = e.ROOT / 'kaggle/elementary_realization.ipynb'; book = json.loads(bookpath.read_text())
        self.assertLess(bookpath.stat().st_size, 900000)
        self.assertLess(sum(len(''.join(c['source']).encode()) for c in book['cells']), 900000)
        self.assertTrue(all(not c.get('outputs') and c.get('execution_count') is None for c in book['cells']))
        with tempfile.TemporaryDirectory() as temp:
            namespace = {}; setup = ''.join(book['cells'][3]['source']).replace('/kaggle/working/_elementary_runtime', temp + '/runtime')
            exec(compile(setup, 'notebook-setup', 'exec'), namespace)
            for name, text in namespace['files'].items(): self.assertEqual((e.ROOT / name).read_text(), text)
            command = [sys.executable, '-c', 'import sys; sys.path.insert(0,sys.argv[1]); from fama_style.rhetoric.elementary_realization import main; main(sys.argv[2:])',
                       str(Path(temp) / 'runtime/src'), '--pack', str(packpath), '--expected-pack-sha', e.file_sha(packpath), '--dry-run', '--output-dir', str(Path(temp) / 'dry')]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((Path(temp) / 'dry/synthetic_evaluation.json').read_text())['generation_calls'], 12)


if __name__ == '__main__': unittest.main()
