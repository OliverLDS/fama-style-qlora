import tests  # Make src/ importable without installing ML dependencies.
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from fama_style.training.track_reviewed_4b import validate_smoke, validate_snapshots, write_claim_review


class ReviewedPilotTests(unittest.TestCase):
    def test_smoke_gate_rejects_wrong_dataset_failed_or_nonfinite_updates(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset = Path(folder) / 'reviewed.jsonl'
            status = Path(folder) / 'smoke_status.json'
            dataset.write_bytes(b'reviewed dataset')
            good = {'model_id': 'google/gemma-3-4b-it', 'complete': True, 'phase': 'complete',
                    'optimizer_steps_completed': 5, 'adapter_weights_changed': True,
                    'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
                    'gradient_checks': [{'finite': True, 'norm': 1.0} for _ in range(5)]}
            status.write_text(json.dumps(good))
            validate_smoke(status, dataset)
            for change in ({'complete': False}, {'dataset_sha256': 'other'},
                           {'optimizer_steps_completed': 4}, {'adapter_weights_changed': False},
                           {'gradient_checks': [{'finite': True, 'norm': float('nan')}] * 5}):
                with self.subTest(change=change):
                    status.write_text(json.dumps(dict(good, **change)))
                    with self.assertRaises(ValueError):
                        validate_smoke(status, dataset)

    def make_report(self):
        examples = [{'source_ids': [f'v{i}'], 'source': f'Source claim {i}.',
                     'reference': f'Reference {i}.'} for i in range(12)]
        return {'validation_examples': examples, 'snapshots': [
            {'epoch': epoch, 'optimizer_step': epoch*24, 'complete': True, 'validation_loss': 3.0,
             'losses': [{'source_ids': r['source_ids'], 'loss': 3.0} for r in examples],
             'generations': [{'source_ids': r['source_ids'], 'output': f'Rewrite at epoch {epoch}.'}
                             for r in examples]} for epoch in range(5)]}

    def test_completion_requires_all_sixty_outputs_correct_ids_and_finite_losses(self):
        good = self.make_report()
        validate_snapshots(good)
        for failure in ('missing_epoch', 'missing_generation', 'wrong_id', 'blank_output', 'nan_loss', 'wrong_step'):
            report = copy.deepcopy(good)
            last = report['snapshots'][-1]
            if failure == 'missing_epoch': report['snapshots'].pop()
            elif failure == 'missing_generation': last['generations'].pop()
            elif failure == 'wrong_id': last['generations'][0]['source_ids'] = ['training-row']
            elif failure == 'blank_output': last['generations'][0]['output'] = ' '
            elif failure == 'nan_loss': last['losses'][0]['loss'] = float('nan')
            elif failure == 'wrong_step': last['optimizer_step'] = 5
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                validate_snapshots(report)

    def test_worksheet_has_each_source_and_epoch_and_leaves_claim_judgments_blank(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'review.md'
            write_claim_review(self.make_report(), path)
            text = path.read_text()
            self.assertEqual(text.count('### Epoch '), 60)
            self.assertEqual(text.count('- [ ] No unsupported claims added'), 60)
            self.assertNotIn('[x]', text)
            self.assertIn('Source claim 11.', text)
            self.assertIn('Causal direction and mechanisms retained', text)


if __name__ == '__main__':
    unittest.main()
