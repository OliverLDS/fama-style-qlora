"""No-plan boundary, six-paper selection and strict 12-output CPU import."""
import ast
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fama_style.rhetoric import paper_plan_comparison as e


class PaperPlanComparisonTests(unittest.TestCase):
    def row(self):
        text = 'Firms may invest. Prices may rise.'
        return {'id': 'synthetic', 'source': text, 'reference': 'PRIVATE_REFERENCE_SENTINEL',
                'payload': {'source': text, 'source_units': [{'unit_id': 'S1', 'text': text[:17]}, {'unit_id': 'S2', 'text': text[18:]}],
                'claim_index': [{'claim_id': 'c' + str(i + 1), 'composite': False, 'epistemic_status': 'hypothesis', 'protected': [],
                                'source_spans': [{'quote': q, 'start': s, 'end': s + len(q)}]} for i, (q, s) in enumerate([(text[:17], 0), (text[18:], 18)])],
                'source_relations': [], 'edits': [{'op': 'MERGE', 'source_unit_ids': ['S1', 'S2'], 'output_slots': [{'claim_refs': ['c1', 'c2']}]}]}}

    def package(self):
        p = e.ROOT / 'research_runs/paper_plan_comparison_v1/package_v1'
        if not p.exists(): self.skipTest('Private package unavailable')
        return p, e.check_package(p)

    def responses(self, folder, key):
        for name in ['responses_01.jsonl', 'responses_02.jsonl']:
            e.write_lines(folder / name, [{'job_id': j['job_id'], 'prompt_sha256': j['prompt_sha256'], 'output': '  Synthetic first response.\nSpacing preserved.  '}
                                        for j in key['jobs'] if j['response_file'] == name])

    def test_ordinary_arm_has_same_evidence_but_no_program_information(self):
        row = self.row(); a, b = [e.prompt_for(row, c) for c in e.CONDITIONS]
        self.assertTrue(a.startswith(e.shared_prefix(row))); self.assertTrue(b.startswith(e.shared_prefix(row)))
        self.assertEqual(a, e.shared_prefix(row) + '\nReturn only the rewritten paragraph.')
        self.assertIn('MERGE', b); self.assertIn('Output sentence 1', b)
        self.assertNotIn('MERGE', a); self.assertNotIn('Output sentence', a); self.assertNotIn('exactly 1 complete', a)
        changed = copy.deepcopy(row); changed['reference'] = 'CHANGED_REFERENCE'
        self.assertEqual(a, e.prompt_for(changed, e.CONDITIONS[0])); self.assertEqual(b, e.prompt_for(changed, e.CONDITIONS[1]))
        changed['payload']['edits'] = [{'op': 'KEEP', 'source_unit_ids': ['S1'], 'output_slots': [{'claim_refs': ['c1']}]},
                                       {'op': 'KEEP', 'source_unit_ids': ['S2'], 'output_slots': [{'claim_refs': ['c2']}]}]
        self.assertEqual(a, e.prompt_for(changed, e.CONDITIONS[0]))
        self.assertNotEqual(b, e.prompt_for(changed, e.CONDITIONS[1]))
        self.assertNotIn('PRIVATE_REFERENCE', b)
        with self.assertRaises(ValueError): e.prompt_for(row, 'unknown')

    def test_selection_forces_anchors_excludes_probes_and_uses_length_ties(self):
        candidates = [{'case_id': c, 'paper_key': 'P' + str(i), 'diagnostic_probe': False, 'prompt': 'long'} for i, c in enumerate(e.ANCHORS)]
        candidates += [{'case_id': 'x' + str(i), 'paper_key': 'P' + str(i), 'diagnostic_probe': False, 'prompt': 'short'} for i in range(2, 6)]
        candidates += [{'case_id': 'longer', 'paper_key': 'P2', 'diagnostic_probe': False, 'prompt': 'much longer prompt'},
                       {'case_id': 'probe', 'paper_key': 'RESERVED', 'diagnostic_probe': True, 'prompt': ''}]
        picked = e.select_six(candidates)
        self.assertEqual(len(picked), 6); self.assertEqual(len({r['paper_key'] for r in picked}), 6)
        self.assertEqual({r['case_id'] for r in picked}, set(e.ANCHORS) | {'x2', 'x3', 'x4', 'x5'})
        self.assertEqual(picked, e.select_six(list(reversed(candidates))))

    def test_real_package_has_matched_rows_six_papers_and_no_reference_leak(self):
        package, key = self.package()
        self.assertEqual(len(key['rows']), 6); self.assertEqual(len({r['paper_key'] for r in key['rows']}), 6)
        self.assertEqual(sum(r['use'] == 'prior_success_anchor' for r in key['rows']), 2)
        self.assertTrue(all(r['split'] == 'train' for r in key['rows']))
        orders = [[j['source_id'] for j in key['jobs'] if j['condition'] == c] for c in e.CONDITIONS]
        self.assertEqual(orders[0], orders[1])
        for name in ['jobs_01.jsonl', 'jobs_02.jsonl']:
            records = e.read_lines(package / 'batches' / name); self.assertEqual(len(records), 6)
            self.assertTrue(all(set(r) == {'job_id', 'prompt_sha256', 'prompt'} for r in records))
            self.assertTrue(all(row['reference'] not in r['prompt'] for row in key['rows'] for r in records))
        self.assertIn('DIFFERENT NEW chat', (package / 'START_HERE.md').read_text())

    def test_import_preserves_raw_outputs_and_anchor_extension_denominators(self):
        package, key = self.package()
        with tempfile.TemporaryDirectory() as temp:
            responses = Path(temp) / 'responses'; responses.mkdir(); self.responses(responses, key)
            output = Path(temp) / 'audit'; e.audit_responses(package, responses, output)
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(summary['accepted_outputs'], 12); self.assertEqual(summary['expected_outputs'], 12)
            self.assertEqual(summary['roles']['prior_success_anchor'], {'expected_outputs': 4, 'accepted_outputs': 4})
            self.assertEqual(summary['roles']['cross_paper_extension'], {'expected_outputs': 8, 'accepted_outputs': 8})
            self.assertTrue(all(s['expected'] == s['accepted'] == 6 for s in summary['conditions'].values()))
            self.assertIsNone(summary['semantic_accuracy']); self.assertIsNone(summary['author_style_accuracy'])
            self.assertEqual(summary['model_calls'], 0); self.assertFalse(summary['matched_decoding'])
            for r in e.read_lines(output / 'metrics.private.jsonl'):
                self.assertEqual(r['output'], '  Synthetic first response.\nSpacing preserved.  ')
                self.assertEqual(r['termination_reason'], 'not_reported'); self.assertNotIn('hit_output_limit', r)
                self.assertEqual(r['operation_realization'], 'not_assessed'); self.assertIsNone(r['acceptable_claim_drift'])
            for n in ['responses_01.jsonl', 'responses_02.jsonl']: self.assertEqual(e.file_sha(responses / n), e.file_sha(output / 'raw' / n))

    def test_partial_and_duplicate_imports_never_select_or_repair(self):
        package, key = self.package()
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp); self.responses(folder, key)
            first = folder / 'responses_01.jsonl'; records = e.read_lines(first)
            e.write_lines(first, records + [records[0]])
            (folder / 'responses_02.jsonl').unlink()
            valid, coverage = e.transport.parse_responses(key, folder)
            self.assertEqual(len(valid), 5); self.assertFalse(coverage['complete'])
            self.assertEqual(len(coverage['missing_or_rejected_job_ids']), 7)
            self.assertIn('duplicate_job_all_copies_excluded', {r['code'] for r in coverage['errors']})
            self.assertTrue(coverage['no_repairs_or_best_of_selection'])

    def test_prepared_file_mutation_is_rejected(self):
        package, key = self.package()
        with tempfile.TemporaryDirectory() as temp:
            copied = Path(temp) / 'package'; shutil.copytree(package, copied)
            p = copied / 'batches/jobs_01.jsonl'; p.write_text(p.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'Prepared file changed'): e.check_package(copied)

    def test_cpu_module_contains_no_model_or_network_actions(self):
        tree = ast.parse(Path(e.__file__).read_text())
        imports = {a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        imports.update((n.module or '').split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom))
        self.assertFalse(imports & {'torch', 'transformers', 'requests', 'subprocess', 'peft'})
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ('backward', 'generate', 'train', 'push_to_hub') for n in ast.walk(tree)))


if __name__ == '__main__': unittest.main()
