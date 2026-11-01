"""Downloaded prose review requires complete frozen coverage and intact bindings."""
import copy
import unittest
from fama_style.research.reconstruction_prose_4b_review import validate_outputs


class ReconstructionProse4BReview(unittest.TestCase):
    def fixture(self):
        jobs=[];outputs=[]
        for i in range(3):
            for method in ('direct_rewrite','detailed_brief','compact_claims'):
                job={'job_id':str(i)+'_'+method,'record_id':str(i),'method':method,'prompt_sha256':'hash'}
                jobs.append(job)
                outputs.append(job|{'raw_output':' Synthetic prose. ',
                                    'parsed':{'status':'generated','format_flags':[]}})
        return jobs,outputs

    def test_full_coverage_preserves_exact_raw_text(self):
        jobs,outputs=self.fixture();before=copy.deepcopy(outputs)
        result=validate_outputs(outputs,jobs)
        self.assertEqual(len(result),9)
        self.assertEqual(result[0]['text'],' Synthetic prose. ')
        self.assertEqual(outputs,before)
        with self.assertRaises(ValueError):validate_outputs(outputs[:-1],jobs)
        with self.assertRaises(ValueError):validate_outputs(outputs[:-1]+[outputs[0]],jobs)

    def test_identity_and_recorded_flags_cannot_change(self):
        for change in ({'prompt_sha256':'changed'}, {'method':'other'},
                       {'parsed':{'status':'empty','format_flags':[]}}):
            jobs,outputs=self.fixture();outputs[0].update(change)
            with self.assertRaises(ValueError):validate_outputs(outputs,jobs)


if __name__=='__main__':unittest.main()
