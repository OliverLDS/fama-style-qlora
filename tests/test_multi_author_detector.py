"""Provider-report interpretation and quota-aware subset preservation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.research import multi_author_detector as m
from fama_style.research.author_control_replication import verify_files


class MultiAuthorDetectorTests(unittest.TestCase):
    def page(self,text='Firms may invest. Costs fall.',label='Human Generated',score='3'):
        return 'Version 2026-10-07-baseTitle - 10/30/2026\nWe predict this text is\n'+label+'\nAI Probability\n'+score+'%\n'+text

    def development(self):
        rows=[]
        for paper in range(4):
            for i in range(3):
                source=('Synthetic economic source statement. '*18)+str(paper)+str(i)
                target=('Synthetic economic reference paragraph. '*18)+str(paper)+str(i)
                rows.append({'split':'development','paper_id':str(paper),'example_id':str(paper)+'_'+str(i),
                    'source':source,'target':target,'source_sha256':m.sha(source),'target_sha256':m.sha(target)})
        return rows

    def test_probability_label_date_and_no_invented_threshold(self):
        r=m.parse_first_page(self.page('Firms may\ninvest. Costs fall.'),'Firms may invest. Costs fall.')
        self.assertEqual(r['ai_probability_percent'],3);self.assertFalse(r['flagged'])
        self.assertEqual(r['provider_version'],'2026-10-07-base');self.assertEqual(r['report_date'],'2026-10-30')
        self.assertIsNone(r['report_timezone']);self.assertIsNone(r['numeric_decision_threshold'])
        self.assertIn('not percent of words',r['score_interpretation'])
        self.assertTrue(m.parse_first_page(self.page(label='AI Generated'),'Firms may invest. Costs fall.')['flagged'])

    def test_wrong_text_or_invalid_score_cannot_be_recorded(self):
        with self.assertRaisesRegex(ValueError,'does_not_match'):m.parse_first_page(self.page(),'Different text.')
        with self.assertRaisesRegex(ValueError,'probability_range'):m.parse_first_page(self.page(score='101'),'Firms may invest. Costs fall.')
        with self.assertRaises(ValueError):m.parse_first_page(self.page(label='Unknown'),'Firms may invest. Costs fall.')

    def test_subset_is_paper_balanced_length_screened_and_outcome_independent(self):
        rows=self.development();first=m.sparse_subset(rows)
        self.assertEqual(len(first),8);self.assertEqual(len({r['paper_id'] for r in first[:4]}),4)
        self.assertEqual(len({r['paper_id'] for r in first[4:]}),4)
        for r in rows:r.update(detector_score=0,output='POISON',fidelity='POISON')
        self.assertEqual(first,m.sparse_subset(list(reversed(rows))))
        rows[0]['split']='training'
        with self.assertRaisesRegex(ValueError,'development_only'):m.sparse_subset(rows)

    def test_receipt_preserves_original_template_and_uses_estimates(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);package=root/'package';(package/'detector_controls').mkdir(parents=True);(package/'datasets').mkdir()
            text='Firms may invest. Costs fall.'
            items=[{'item_id':k,'text':v,'text_sha256':m.sha(v)} for k,v in [('ai',text),('human','Costs fall; firms may invest.')]]
            (package/'detector_controls/items.private.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in items))
            key=[{'item_id':r['item_id'],'example_id':'pair','origin':'GPT6_rewrite' if r['item_id']=='ai' else 'historical_human'} for r in items]
            (package/'detector_controls/PRIVATE_KEY.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in key))
            (package/'detector_controls/results_template.csv').write_text('item_id,provider,provider_version,threshold_rule,status,provider_label,ai_score,score_scale,notes\nai,GPTZero,,,not_tested,,,,\nhuman,GPTZero,,,not_tested,,,,\n')
            (package/'datasets/development.private.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.development()))
            m.file_manifest(package)
            page=root/'page.json';page.write_text(json.dumps({'page_extracted':1,'text':self.page()}))
            pdf=root/'report.pdf';pdf.write_bytes(b'%PDF-synthetic-report')
            before=(package/'detector_controls/results_template.csv').read_bytes()
            summary=m.run(package,page,pdf,'ai',root/'out')
            self.assertEqual(summary['next_control_id'],'human');self.assertFalse(summary['model_output_word_counts_observed'])
            self.assertFalse(summary['detector_accuracy_or_feasibility_established'])
            self.assertEqual(before,(package/'detector_controls/results_template.csv').read_bytes())
            verify_files(root/'out')
            with self.assertRaises(ValueError):m.run(package,page,pdf,'ai',root/'out')
