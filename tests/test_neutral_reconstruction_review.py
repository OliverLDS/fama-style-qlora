"""Synthetic tests for reversible review patches and marker-only source variants."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from fama_style.research.neutral_reconstruction_review import patch_brief,clean_markers


class NeutralReconstructionReview(unittest.TestCase):
    def test_patch_checks_before_value_and_preserves_raw(self):
        raw={'claims':[{'modality':['may']}],'needs_review':True};saved=copy.deepcopy(raw)
        patch={'path':['claims',0,'modality'],'before':['may'],'after':['may under condition X']}
        result=patch_brief(raw,[patch])
        self.assertEqual(raw,saved)
        self.assertEqual(result['claims'][0]['modality'],['may under condition X'])
        patch['before']=['certain']
        with self.assertRaises(ValueError):patch_brief(raw,[patch])

    def test_marker_removal_preserves_economic_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);evidence=p/'synthetic.png';evidence.write_bytes(b'synthetic evidence')
            repair={'before':'conditions.4 The','after':'conditions. The','marker':'4',
                    'evidence_image':evidence.name,'evidence_image_sha256':hashlib.sha256(evidence.read_bytes()).hexdigest(),
                    'paper_pdf_sha256':'p'*64}
            text='The rate is 4 percent under specified conditions.4 The scope remains conditional.'
            clean=clean_markers(text,[repair],p,'p'*64)
            self.assertIn('4 percent',clean)
            self.assertNotIn('conditions.4',clean)

    def test_other_source_changes_and_missing_evidence_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);evidence=p/'synthetic.png';evidence.write_bytes(b'synthetic evidence')
            repair={'before':'12 percent','after':'15 percent','marker':'12','evidence_image':evidence.name,
                    'evidence_image_sha256':hashlib.sha256(evidence.read_bytes()).hexdigest(),'paper_pdf_sha256':'p'*64}
            with self.assertRaises(ValueError):clean_markers('12 percent',[repair],p,'p'*64)
            repair.update(before='facts.2 Next',after='facts. Next',marker='2',evidence_image_sha256='wrong')
            with self.assertRaises(ValueError):clean_markers('facts.2 Next',[repair],p,'p'*64)

    def test_nonunique_marker_anchor_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);evidence=p/'synthetic.png';evidence.write_bytes(b'synthetic evidence')
            repair={'before':'facts.2 Next','after':'facts. Next','marker':'2','evidence_image':evidence.name,
                    'evidence_image_sha256':hashlib.sha256(evidence.read_bytes()).hexdigest(),'paper_pdf_sha256':'p'*64}
            with self.assertRaises(ValueError):clean_markers('facts.2 Next facts.2 Next',[repair],p,'p'*64)


if __name__=='__main__':unittest.main()
