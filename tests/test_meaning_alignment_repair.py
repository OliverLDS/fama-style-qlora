"""Versioned source repairs retain targets, prompts, patch provenance and gates."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.research import meaning_alignment_repair as e


class MeaningAlignmentRepairTests(unittest.TestCase):
    def fixture(self):
        row={'id':'x','paper_key':'ME1997','split':'train','plain_english':'A guess. A guess.',
             'completion':'A qualified assumption. A second qualification.','original_paragraph':'A qualified assumption. A second qualification.',
             'prompt':'Old prompt.','quality':{'repair_count':0}}
        spec={'source_id':'x','paper_key':'ME1997','expected_source_sha256':e.sha(row['plain_english']),
              'expected_target_sha256':e.sha(row['completion']),'decision':'clarified','remaining_context':[],
              'edits':[{'old':'A guess.','occurrence':2,'new':'Another qualified assumption.','reason':'Synthetic second qualifier.',
                        'target_evidence':[{'quote':'A second qualification.','occurrence':1}]},
                       {'old':'A guess.','occurrence':1,'new':'A qualified assumption.','reason':'Synthetic first qualifier.',
                        'target_evidence':[{'quote':'A qualified assumption.','occurrence':1}]}]}
        return row,spec

    def test_repeated_occurrences_unsorted_edits_and_evidence_stay_bound(self):
        row,spec=self.fixture();before=copy.deepcopy(row)
        revised,log=e.repair_row(row,spec)
        self.assertEqual(revised['plain_english'],'A qualified assumption. Another qualified assumption.')
        self.assertEqual(log[0]['target_evidence'][0]['quote'],'A qualified assumption.')
        self.assertEqual(log[1]['target_evidence'][0]['quote'],'A second qualification.')
        self.assertEqual(row,before);self.assertEqual(revised['completion'],row['completion'])
        self.assertEqual(revised['prompt'],e.make_prompt(revised['plain_english']))
        self.assertFalse(revised['alignment_revision']['training_ready'])
        self.assertFalse(revised['alignment_revision']['automatic_generation_ready'])
        self.assertFalse(revised['alignment_revision']['old_source_plan_reuse_allowed'])

    def test_changed_parent_or_holdout_cannot_be_repaired(self):
        row,spec=self.fixture()
        for field,value in [('split','test'),('plain_english','Changed source'),('completion','Changed target')]:
            changed=copy.deepcopy(row);changed[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):e.repair_row(changed,spec)

    def test_overlaps_and_invented_target_support_are_rejected(self):
        row,spec=self.fixture();bad=copy.deepcopy(spec)
        bad['edits'][1]['occurrence']=2
        with self.assertRaisesRegex(ValueError,'Overlapping'):e.repair_row(row,bad)
        bad=copy.deepcopy(spec);bad['edits'][0]['target_evidence'][0]['quote']='Unsupported invention'
        with self.assertRaises(ValueError):e.repair_row(row,bad)
        bad=copy.deepcopy(spec);bad['decision']='clarified_context_bound'
        with self.assertRaises(ValueError):e.repair_row(row,bad)

    def test_pdf_normalization_does_not_invent_ocr_letters(self):
        anchor=e.normalized_pdf_anchor('rati-\nfication: choice','ratification choice')
        self.assertEqual(anchor['quote'],'rati-\nfication: choice')
        with self.assertRaises(ValueError):e.normalized_pdf_anchor('~vealth','wealth')

    def test_real_overlay_is_reversible_target_preserving_and_inactive(self):
        root=e.ROOT;specpath=root/'research_runs/meaning_alignment_repair_v1/specification_v1/repairs.private.json'
        if not specpath.exists():self.skipTest('Private repair inputs absent')
        spec=json.loads(specpath.read_text());train={r['id']:r for r in e.read_lines(root/'data/fama_corpus_v1/train.jsonl')}
        self.assertEqual(e.file_sha(root/'data/fama_corpus_v1/train.jsonl'),e.TRAIN_SHA)
        overlay=e.read_lines(root/'data/meaning_alignment_repair_v1/repaired_pairs.private.jsonl')
        self.assertEqual(len(overlay),4);self.assertEqual(len({r['id'] for r in overlay}),4)
        for s in spec['records']:
            parent=train[s['source_id']];before=copy.deepcopy(parent);new,log=e.repair_row(parent,s)
            self.assertEqual(parent,before);self.assertEqual(new,next(r for r in overlay if r['id']==s['source_id']))
            self.assertEqual(new['original_paragraph'],parent['original_paragraph']);self.assertNotEqual(new['plain_english'],parent['plain_english'])
            self.assertEqual(new['rhetorical_taxonomy'],parent['rhetorical_taxonomy'])
            for item in log:
                a=item['before_span'];b=item['after_span']
                self.assertEqual(parent['plain_english'][a['start']:a['end']],a['quote'])
                self.assertEqual(new['plain_english'][b['start']:b['end']],b['quote'])
                for evidence in item['target_evidence']:self.assertEqual(parent['completion'][evidence['start']:evidence['end']],evidence['quote'])
        nl=next(r for r in overlay if r['paper_key']=='NL2003')
        self.assertEqual(nl['alignment_revision']['decision'],'clarified_context_bound');self.assertTrue(nl['alignment_revision']['remaining_context'])
        self.assertEqual(sum(len(s['edits']) for s in spec['records']),9)

    def test_private_release_rebuilds_without_changing_frozen_train(self):
        base=e.ROOT/'research_runs/meaning_alignment_repair_v1'
        spec=base/'specification_v1/repairs.private.json'
        if not spec.exists():self.skipTest('Private repair inputs absent')
        train=e.ROOT/'data/fama_corpus_v1/train.jsonl';before=e.file_sha(train)
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'release';e.build(spec,base/'evidence_v1',output)
            active=e.ROOT/'data/meaning_alignment_repair_v1'
            for name in ['repaired_pairs.private.jsonl','repair_log.private.json','pdf_evidence.private.json','summary.json']:
                self.assertEqual((output/name).read_bytes(),(active/name).read_bytes())
        self.assertEqual(e.file_sha(train),before)


if __name__=='__main__':unittest.main()
