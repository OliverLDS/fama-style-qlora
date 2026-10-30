"""Synthetic safeguards for the targeted prompt-only repair."""
import copy
import unittest
from fama_style.research import neutral_reconstruction_instruction as n


class ReconstructionInstructionCheck(unittest.TestCase):
    def test_prompt_preserves_entire_base_with_one_rule_and_coordination_header(self):
        base = 'Old instructions.\nIf a missing fact, unresolved context gap or inconsistency prevents faithful\ngeneration, block.\n'
        result = n.revised_prompt(base)
        self.assertEqual(result.removeprefix(n.READY).replace(n.RULE+'\n', ''), base)
        with self.assertRaises(ValueError):
            n.revised_prompt('Unrecognized template')

    def test_selection_preserves_payload_and_keeps_reference_out(self):
        payloads = [{'record_id': 'synthetic', 'content_brief': {'claims': [{'modality': 'conditional'}]}}]
        observed = [{'record_id': 'synthetic', 'method': 'neutral_reconstruction', 'text': 'Prior output.',
                     'original_paragraph': 'Private reference.', 'assessment': {'local_review_flags': [
                         {'type': 'added_epistemic_metacommentary'}]}}]
        before = copy.deepcopy(payloads)
        selected, baseline = n.select(payloads, observed, {'synthetic'})
        self.assertEqual(selected, before)
        self.assertEqual(payloads, before)
        self.assertNotIn('text', selected[0])
        self.assertEqual(baseline, observed)
        with self.assertRaises(ValueError):
            n.select(payloads, observed, {'other'})
        with self.assertRaises(ValueError):
            n.select(payloads+payloads, observed, {'synthetic'})

    def test_direct_method_commentary_does_not_select_a_case(self):
        observed = [{'record_id': 's', 'method': 'direct_rewrite', 'assessment': {
            'local_review_flags': [{'type': 'added_epistemic_metacommentary'}]}}]
        with self.assertRaises(ValueError):
            n.select([], observed, {'s'})


if __name__ == '__main__':
    unittest.main()
