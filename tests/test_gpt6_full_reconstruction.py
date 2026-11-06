import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('full_reconstruction', Path(__file__).resolve().parents[1] / 'scripts/data/prepare_gpt6_full_reconstruction.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class FullReconstructionTests(unittest.TestCase):
    def test_reuse_and_hold_never_enter_new_queue(self):
        targets = [dict(id=str(i)) for i in range(5)]
        queue, held = m.partition(targets, {'0', '2'}, {'4'})
        self.assertEqual([r['id'] for r in queue], ['1', '3'])
        self.assertEqual([r['id'] for r in held], ['4'])

    def test_invalid_identity_and_ambiguous_hold_fail(self):
        for targets, existing, holds in [([dict(id='a'), dict(id='a')], set(), set()),
                                         ([dict(id='a')], {'missing'}, set()),
                                         ([dict(id='a')], {'a'}, {'a'})]:
            with self.assertRaises(ValueError):
                m.partition(targets, existing, holds)


if __name__ == '__main__':
    unittest.main()
