"""CPU tests for the saved-adapter prompt comparison; no private data required."""
import ast
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fama_style.evaluation import corpus_200_simple_prompt as runtime


class SimplePromptTest(unittest.TestCase):
    def fixture(self, root):
        adapter = root / 'adapter'
        adapter.mkdir()
        for name in ['adapter_config.json', 'adapter_model.safetensors']:
            (adapter / name).write_bytes(b'synthetic fixture')
        rows = [dict(example_id=str(i), source=f'Source {i}.', reference=f'Reference {i}.',
                     paper_id=str(i // 2), style_group=str(i // 4)) for i in range(16)]
        report = dict(protocol='corpus_200_pilot_v1', complete=True,
                      training_examples_seen=200, optimizer_steps=100, adapter_reload_weights_match=True,
                      config=dict(model_id=runtime.MODEL_ID, model_revision=runtime.REVISION,
                                  decoding=runtime.DECODING, packages=runtime.PACKAGES,
                                  development_ids=[r['example_id'] for r in rows],
                                  detector_case_ids=[str(i) for i in range(8)]),
                      snapshots=[dict(name=name, complete=True, generations=rows)
                                 for name in ['base', 'epoch_1']],
                      adapter_files_sha256={p.name: runtime.file_sha(p) for p in adapter.iterdir()})
        path = root / 'evaluation.json'
        runtime.write_json(path, report)
        return path, adapter, runtime.file_sha(path)

    def test_cpu_check_and_exact_prompt(self):
        self.assertEqual(runtime.prompt('A source.'),
                         'Rewrite the following passage in natural, human-written academic economics prose.\n\n'
                         'Return only the rewritten paragraph.\n\nPassage:\nA source.')
        with tempfile.TemporaryDirectory() as folder:
            path, adapter, digest = self.fixture(Path(folder))
            rows, _ = runtime.load_inputs(path, adapter, digest)
            self.assertEqual(len(rows), 16)
            with patch.object(runtime, 'run_training', side_effect=AssertionError('GPU forbidden')):
                with contextlib.redirect_stdout(io.StringIO()):
                    runtime.main(['--report', str(path), '--adapter', str(adapter),
                                  '--report-sha256', digest, '--check'])

    def test_report_and_adapter_mutations_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path, adapter, digest = self.fixture(Path(folder))
            with self.assertRaises(ValueError):
                runtime.load_inputs(path, adapter, 'wrong digest')
            (adapter / 'adapter_model.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'adapter_changed'):
                runtime.load_inputs(path, adapter, digest)

    def test_changed_matched_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path, adapter, _ = self.fixture(Path(folder))
            report = json.loads(path.read_text())
            report['snapshots'][1]['generations'][0]['source'] = 'Different input'
            runtime.write_json(path, report)
            with self.assertRaisesRegex(ValueError, 'matched_inputs_differ'):
                runtime.load_inputs(path, adapter, runtime.file_sha(path))

    def test_notebook_is_self_contained_clean_and_inference_only(self):
        path = ROOT / 'kaggle/corpus_200_simple_prompt_4b.ipynb'
        book = json.loads(path.read_text())
        self.assertLess(path.stat().st_size, 1_000_000)
        embedded = None
        for cell in book['cells']:
            if cell['cell_type'] != 'code':
                continue
            self.assertIsNone(cell['execution_count'])
            self.assertEqual(cell['outputs'], [])
            code = ''.join(cell['source'])
            if code.startswith('%%writefile'):
                code = code.split('\n', 1)[1]
                embedded = code.rstrip()
            ast.parse(code)
        self.assertEqual(embedded, Path(runtime.__file__).read_text().rstrip())
        calls = [n for n in ast.walk(ast.parse(embedded)) if isinstance(n, ast.Call)]
        self.assertFalse(any(isinstance(n.func, ast.Attribute) and
                             n.func.attr in {'backward', 'step', 'save_pretrained', 'get_peft_model'}
                             for n in calls))


if __name__ == '__main__':
    unittest.main()
