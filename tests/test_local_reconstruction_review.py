"""Saved-text review must have exact coverage and hash-bound judgments."""
import hashlib
import unittest
from fama_style.research.local_reconstruction_review import join_assessments


class LocalReconstructionReview(unittest.TestCase):
    def test_exact_text_hash_and_coverage_required(self):
        rows=[{'job_id':'synthetic','text':'Synthetic output.'}]
        assessment={'job_id':'synthetic','text_sha256':hashlib.sha256(rows[0]['text'].encode()).hexdigest()}
        self.assertEqual(join_assessments(rows,[assessment])[0]['text'],rows[0]['text'])
        with self.assertRaises(ValueError):join_assessments(rows,[])
        with self.assertRaises(ValueError):join_assessments(rows,[assessment,assessment])
        with self.assertRaises(ValueError):join_assessments(rows,[assessment|{'text_sha256':'wrong'}])
        with self.assertRaises(ValueError):join_assessments(rows+rows,[assessment])


if __name__=='__main__':unittest.main()
