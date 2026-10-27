"""Frozen prompt transport and strict CPU-only external-response import."""
import ast
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fama_style.rhetoric import readable_strong_check as e


class ReadableStrongCheckTests(unittest.TestCase):
    def package(self):
        p = e.ROOT / 'research_runs/readable_strong_realizer_v1/package_v2'
        if not p.exists(): self.skipTest('Private package absent')
        return p, e.check_package(p)

    def responses(self, key, folder):
        for job in key['jobs']:
            e.write_lines(folder / job['response_file'], [{'job_id': job['job_id'], 'prompt_sha256': job['prompt_sha256'],
                                                         'output': '  Synthetic first paragraph.\nOriginal spacing remains.  '}])

    def test_requests_copy_exact_prompts_without_evaluation_fields(self):
        jobs = [{'id': str(i), 'condition': str(i % 2), 'prompt': 'Synthetic ' + str(i), 'prompt_sha256': e.sha('Synthetic ' + str(i)),
                 'reference': 'PRIVATE_REFERENCE', 'use': 'PRIVATE_ASSESSMENT'} for i in range(4)]
        before = copy.deepcopy(jobs); requests = e.requests_for(jobs)
        self.assertEqual(jobs, before); self.assertEqual(len({r['job_id'] for r in requests}), 4)
        self.assertEqual(requests, e.requests_for(jobs))
        self.assertTrue(all(set(r) == {'job_id', 'prompt_sha256', 'prompt'} for r in requests))
        self.assertEqual([r['prompt'] for r in requests], [j['prompt'] for j in jobs])
        self.assertNotIn('PRIVATE_', json.dumps(requests))

    def test_delivery_contract_preserves_uncertainty_and_single_job_setup(self):
        self.assertEqual(set(e.RESPONSE_SCHEMA['properties']), {'job_id', 'prompt_sha256', 'output'})
        self.assertFalse(e.RESPONSE_SCHEMA['additionalProperties'])
        self.assertIn('separate new chat', e.START_HERE)
        self.assertIn('do not prove', e.START_HERE)
        self.assertIn('first returned', e.START_HERE)
        self.assertNotIn('grade your answer', e.START_HERE)
        self.assertIn('No API key or HF_TOKEN is needed', e.START_HERE)

    def test_real_package_contains_four_exact_single_job_prompts(self):
        p, key = self.package(); pack, jobs, provenance = e.parent_inputs()
        files = sorted((p / 'batches').glob('*.jsonl')); self.assertEqual(len(files), 4)
        for f, original in zip(files, jobs):
            records = e.read_lines(f); self.assertEqual(len(records), 1)
            self.assertEqual(records[0]['prompt'], original['prompt'])
            self.assertEqual(records[0]['prompt_sha256'], original['prompt_sha256'])
            self.assertEqual(set(records[0]), {'job_id', 'prompt_sha256', 'prompt'})
            self.assertTrue(all(row['reference'] not in records[0]['prompt'] for row in key['rows']))
        self.assertEqual(key['rows'], pack['rows'])

    def test_roundtrip_preserves_raw_files_and_does_not_claim_quality(self):
        package, key = self.package()
        with tempfile.TemporaryDirectory() as temp:
            responses = Path(temp) / 'responses'; responses.mkdir(); self.responses(key, responses)
            (responses / 'notes.txt').write_text('Visible model label not recorded.\n')
            output = Path(temp) / 'audit'; e.audit_responses(package, responses, output)
            summary = json.loads((output / 'summary.json').read_text()); self.assertEqual(summary['accepted_outputs'], 4)
            self.assertEqual(summary['status'], 'import_complete_quality_assessment_pending')
            self.assertFalse(summary['matched_decoding']); self.assertIsNone(summary['semantic_accuracy'])
            self.assertEqual(summary['model_calls'], 0); self.assertFalse(summary['training_called'])
            for j in key['jobs']: self.assertEqual(e.file_sha(responses / j['response_file']), e.file_sha(output / 'raw' / j['response_file']))
            self.assertEqual(e.file_sha(responses / 'notes.txt'), e.file_sha(output / 'raw/notes.txt'))
            for r in e.read_lines(output / 'metrics.private.jsonl'):
                self.assertEqual(r['output'], '  Synthetic first paragraph.\nOriginal spacing remains.  ')
                self.assertEqual(r['termination_reason'], 'not_reported'); self.assertNotIn('hit_output_limit', r)
                self.assertEqual(r['operation_realization'], 'not_assessed'); self.assertIsNone(r['acceptable_claim_drift'])

    def test_partial_wrong_hash_and_duplicate_responses_are_not_repaired(self):
        package, key = self.package()
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp); self.responses(key, folder)
            (folder / key['jobs'][0]['response_file']).unlink()
            duplicated = folder / key['jobs'][1]['response_file']; duplicated.write_text(duplicated.read_text() * 2)
            wrong = folder / key['jobs'][2]['response_file']; row = json.loads(wrong.read_text()); row['prompt_sha256'] = 'f' * 64
            wrong.write_text(json.dumps(row) + '\n')
            valid, coverage = e.transport.parse_responses(key, folder)
            self.assertFalse(coverage['complete']); self.assertEqual(coverage['accepted_jobs'], 1)
            self.assertEqual(len(coverage['missing_or_rejected_job_ids']), 3)
            self.assertTrue(coverage['no_repairs_or_best_of_selection'])
            self.assertEqual(valid[0]['job_id'], key['jobs'][3]['job_id'])

    def test_package_tampering_fails_before_import(self):
        package, key = self.package()
        with tempfile.TemporaryDirectory() as temp:
            copied = Path(temp) / 'package'; shutil.copytree(package, copied)
            f = copied / 'batches/jobs_01.jsonl'; f.write_text(f.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'Prepared file changed'): e.check_package(copied)

    def test_module_has_no_model_network_or_training_actions(self):
        tree = ast.parse(Path(e.__file__).read_text())
        imports = {a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        imports.update((n.module or '').split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom))
        self.assertFalse(imports & {'torch', 'transformers', 'peft', 'requests', 'subprocess'})
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        self.assertFalse(any(isinstance(n.func, ast.Attribute) and n.func.attr in ('backward', 'train', 'generate', 'push_to_hub') for n in calls))


if __name__ == '__main__': unittest.main()
