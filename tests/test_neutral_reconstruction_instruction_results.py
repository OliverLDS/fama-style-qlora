"""Response integrity checks for the targeted three-case realization repair."""
import unittest
from fama_style.research.neutral_reconstruction_instruction_results import validate_received


class InstructionResults(unittest.TestCase):
    def row(self):
        return {'record_id':'synthetic','method':'neutral_reconstruction','status':'generated',
                'text':'Synthetic economics paragraph.','reason':''}

    def test_missing_duplicate_and_unexpected_records_rejected(self):
        row=self.row()
        validate_received([row], {'synthetic'})
        for received in ([], [row,row], [row|{'record_id':'other'}]):
            with self.assertRaises(ValueError):validate_received(received, {'synthetic'})

    def test_blocked_case_is_retained_in_coverage(self):
        row=self.row()|{'status':'blocked','text':None,'reason':'Unresolved condition.'}
        validate_received([row], {'synthetic'})
        with self.assertRaises(ValueError):
            validate_received([row|{'reason':''}], {'synthetic'})

    def test_wrong_method_and_extra_reference_payload_rejected(self):
        for change in ({'method':'direct_rewrite'}, {'original_paragraph':'Private source.'}):
            with self.assertRaises(ValueError):validate_received([self.row()|change], {'synthetic'})


if __name__=='__main__':unittest.main()
