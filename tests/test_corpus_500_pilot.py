"""Check paired held-out variants, fixed training scope and the upload notebook."""
import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fama_style.training import corpus_500_pilot as runtime
from fama_style.training import corpus_200_pilot as old_runtime
from fama_style.evaluation import corpus_200_simple_prompt as simple
from fama_style.research.corpus_500_packet import build
from fama_style.research.corpus_200_packet import file_sha, read_rows, write_rows

SPEC = importlib.util.spec_from_file_location('parent200tests', ROOT / 'tests/test_corpus_200_pilot.py')
parent_tests = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(parent_tests)


class Corpus500Pilot(unittest.TestCase):
    def package(self, root):
        parents, parent, old_data, old_config = parent_tests.Corpus200Pilot().package(root)
        rows = read_rows(parents[0])
        for i in range(75):
            for g in runtime.GROUPS:
                source = f'Additional neutral training {g} case {i}.'
                target = f'Additional author training {g} case {i}.'
                rows.append(dict(id=f'new|{g}|{i}', parent_id=f'newparent|{g}|{i}', paper_id=f'newpaper|{g}',
                    style_group=g, authors=[g + ' author'], split='training', training_ready=True,
                    source=source, target=target, source_sha256=runtime.sha(source), target_sha256=runtime.sha(target),
                    quality=dict(text_pair_status='automated_pilot_approved')))
        for r in rows[:200]:
            r.update(source=r['plain_english'], target=r['original_paragraph'], training_ready=True)
        folder = root / 'training500'
        folder.mkdir()
        data = folder / 'training.private.jsonl'
        write_rows(data, rows)
        runtime.write_json(folder / 'manifest.json', dict(files={data.name:file_sha(data)}))
        dev = {r['example_id']:r for r in read_rows(old_data) if r['split'] == 'development'}
        ids = json.loads(old_config.read_text())['detector_case_ids']
        teacher = root / 'gemini'
        teacher.mkdir()
        path = teacher / 'accepted_inputs.private.jsonl'
        write_rows(path, [dict(content_id=rid, source='Gemini ' + dev[rid]['source'],
            source_sha256=runtime.sha('Gemini ' + dev[rid]['source']), reference_sha256=dev[rid]['target_sha256'],
            model='gemini-3-flash-preview', acceptance='automated_meaning_review') for rid in ids])
        runtime.write_json(teacher / 'accepted_manifest.json', dict(files={path.name:file_sha(path)}))
        out = root / 'packet500'
        build(data, parent, path, out)
        return data, parent, path, out, out / 'upload/corpus_500_pairs.jsonl', out / 'upload/corpus_500_config.json'

    def test_real_scope_and_CPU_check_never_train(self):
        with tempfile.TemporaryDirectory() as t:
            _, _, _, _, data, config = self.package(Path(t))
            with patch.object(runtime, 'run_training', side_effect=AssertionError('No GPU')):
                with contextlib.redirect_stdout(io.StringIO()):
                    runtime.main(['--data',str(data),'--config',str(config),'--check'])
                with self.assertRaises(ValueError):
                    runtime.main(['--data',str(data),'--config',str(config)])
            splits, c = runtime.load_inputs(data, config)
            self.assertEqual([len(splits[k]) for k in splits], [500,24])
            self.assertEqual(len({r['content_id'] for r in splits['development']}), 16)
            self.assertEqual(len(c['detector_priority_ids']), 2)
            self.assertEqual(runtime.TRAIN_INSTRUCTION, old_runtime.INSTRUCTION)
            self.assertEqual(runtime.INSTRUCTION, simple.INSTRUCTION)
            self.assertEqual(runtime.prompt('Example.'), simple.prompt('Example.'))

    def test_variants_cannot_change_reference_or_become_training(self):
        for issue in ['reference','parent','role','prompt','budget','paired_id']:
            with self.subTest(issue=issue), tempfile.TemporaryDirectory() as t:
                _, _, _, _, data, path = self.package(Path(t))
                rows = read_rows(data)
                c = json.loads(path.read_text())
                r = rows[-1]
                if issue == 'reference':
                    r['target'] += ' Added claim.'
                    r['target_sha256'] = runtime.sha(r['target'])
                elif issue == 'parent': r['parent_id'] = 'different parent'
                elif issue == 'role': r['split'] = 'training'
                elif issue == 'prompt': r['prompt'] += '\nReference: ' + r['target']
                elif issue == 'budget': c['hyperparameters']['optimizer_steps'] = 500
                elif issue == 'paired_id': r['content_id'] = 'unmatched'
                write_rows(data, rows)
                c['dataset_sha256'] = file_sha(data)
                runtime.write_json(path, c)
                with self.assertRaises(ValueError): runtime.load_inputs(data, path)

    def test_shared_paper_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            _, _, _, _, data, path = self.package(Path(t))
            rows = read_rows(data)
            rows[0]['paper_id'] = rows[-1]['paper_id']
            write_rows(data, rows)
            c = json.loads(path.read_text()); c['dataset_sha256'] = file_sha(data)
            runtime.write_json(path, c)
            with self.assertRaisesRegex(ValueError, 'paper_leakage'):
                runtime.load_inputs(data, path)

    def test_packet_rebuild_and_teacher_hash_binding(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            train, parent, teacher, out, _, _ = self.package(root)
            replica = root / 'replica'
            build(train, parent, teacher, replica)
            for p in out.rglob('*'):
                if p.is_file(): self.assertEqual(p.read_bytes(), (replica / p.relative_to(out)).read_bytes())
            with self.assertRaises(ValueError): build(train, parent, teacher, out)
            teacher.write_text(teacher.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'teacher_inputs_changed'):
                build(train, parent, teacher, root / 'changed')

    def test_notebook_is_small_self_contained_and_has_no_private_rows(self):
        spec = importlib.util.spec_from_file_location('builder500', ROOT / 'scripts/build_corpus_500_pilot.py')
        builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as t:
            _, _, _, out, _, _ = self.package(Path(t))
            path = Path(t) / 'book.ipynb'
            builder.build_notebook(out, path)
            book = json.loads(path.read_text())
            text = '\n'.join(''.join(c['source']) for c in book['cells'])
            self.assertIn(Path(runtime.__file__).read_text(), text)
            self.assertNotIn('Additional author training Fama_coauthors case 0.', text)
            self.assertIn("os.environ['CUDA_VISIBLE_DEVICES']='0'", text)
            self.assertLess(path.stat().st_size, 1_000_000)
            for c in book['cells']:
                if c['cell_type'] == 'code':
                    self.assertIsNone(c['execution_count']); self.assertEqual(c['outputs'], [])
                    if not ''.join(c['source']).startswith('%%writefile'): ast.parse(''.join(c['source']))
        for node in ast.parse(Path(runtime.__file__).read_text()).body:
            if isinstance(node, ast.Import):
                self.assertFalse(any(x.name.split('.')[0] in {'torch','transformers','peft','huggingface_hub'} for x in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn((node.module or '').split('.')[0], {'torch','transformers','peft','huggingface_hub'})

    def test_optimizer_schedule_is_bounded_to_one_500_pair_epoch(self):
        self.assertEqual(runtime.learning_rate(1), 2e-5)
        self.assertGreater(runtime.learning_rate(250), 0)
        for step in [0,251,1.5]:
            with self.assertRaises(ValueError): runtime.learning_rate(step)


if __name__ == '__main__': unittest.main()
