"""Native detector semantics, complete text binding and matched denominators."""
import unittest

from fama_style.evaluation.milestone_checkpoint_4b import sha
from fama_style.research.multi_author_detector import parse_gptzero_header
from fama_style.research.detector_provider_comparison import parse_pangram,bind,compare,account_batches


class ProviderComparison(unittest.TestCase):
    def test_mixed_with_zero_ai_probability_is_not_human(self):
        text='Version 2026-10-07-base\n11/1/2026\nWe are highly confident this text is\nMix of AI and Human\nAI Probability\n0%'
        result=parse_gptzero_header(text)
        self.assertEqual(result['provider_label'],'Mix of AI and Human');self.assertTrue(result['flagged'])
        self.assertEqual(result['ai_probability_percent'],0)

    def test_pangram_shares_and_displayed_rounding_are_not_ai_probability(self):
        text='Nov 1, 2026 130 Words Pangram 4.1\nSummary\nWe believe that this entire text is AI-assisted.\n0.1 % of the document\n100 % of the document\n0.1 % of the document'
        result=parse_pangram(text)
        self.assertEqual(result['ai_assisted_document_percent'],100);self.assertEqual(result['ai_generated_document_percent'],0.1)
        self.assertAlmostEqual(result['displayed_percentage_sum'],100.2);self.assertFalse(result['percentages_rescaled'])
        self.assertNotIn('ai_probability_percent',result);self.assertEqual(result['provider_word_count'],130)
        with self.assertRaises(ValueError):parse_pangram(text.replace('100 %','101 %'))

    def test_full_text_across_pages_and_metadata_removal_do_not_change_numbers(self):
        text='Firms may invest—especially in sector 2. Costs can fall.'
        item={'scan_id':'SCAN_1234567890abcdef','text':text,'text_sha256':sha(text)}
        report={'text':'Header only','layout_pages':['AI Assisted 8 Words Medium Confidence\nFirms may invest— especially in sector 2.\nUUID Nov 1, 2026 Page 1/2',
            'AI Assisted 4 Words Low Confidence\nCosts can fall.']}
        got,method=bind(report,[item],'pdf','automatic.pdf',{})
        self.assertEqual(got,item);self.assertIn('reading_order',method)
        report['layout_pages'][0]=report['layout_pages'][0].replace('sector 2','sector 3')
        with self.assertRaises(ValueError):bind(report,[item],'pdf','automatic.pdf',{})

    def test_word_boundary_artifacts_require_hash_bound_visual_audit(self):
        text='Trading costs fall.';sid='SCAN_1234567890abcdef';item={'scan_id':sid,'text':text,'text_sha256':sha(text)}
        report={'text':'Incorrect extraction','layout_pages':['T rading costs fall.']}
        with self.assertRaisesRegex(ValueError,'visual_confirmation'):bind(report,[item],'pdf',sid+'.pdf',{})
        audit={'pdf':{'scan_id':sid,'text_sha256':sha(text)}}
        self.assertEqual(bind(report,[item],'pdf',sid+'.pdf',audit)[0],item)
        with self.assertRaises(ValueError):bind(report,[item],'different-pdf',sid+'.pdf',audit)
        with self.assertRaisesRegex(ValueError,'filename_conflicts'):bind(report,[item],'pdf','SCAN_abcdef1234567890.pdf',audit)

    def test_accuracy_comparison_uses_intersection_and_preserves_rewrite_misses(self):
        observations=[]
        for provider in ('GPTZero','Pangram'):
            for sid,condition,flag,label in [('h','historical_human_reference',False,'Human'),('m','adapter_ordinary',True,'AI'),('r','GPT6_input',False,'Human')]:
                observations.append({'provider':provider,'scan_id':sid,'condition':condition,'flagged':flag,'provider_label':label})
        observations.append({'provider':'GPTZero','scan_id':'extra','condition':'base_ordinary','flagged':True,'provider_label':'AI'})
        summary,paired=compare(observations)
        self.assertEqual(summary['shared_texts'],3);self.assertEqual(summary['ai_related_agreements'],3)
        for p in ('GPTZero','Pangram'):
            shared=summary['providers'][p]['matched_subset'];self.assertEqual(shared['machine_produced_texts'],2)
            self.assertEqual(shared['machine_ai_related_flags'],1);self.assertEqual(shared['human_false_positive_flags'],0)
        with self.assertRaises(ValueError):compare(observations+[observations[0]])

    def test_account_provenance_is_complete_unique_and_pdf_bound(self):
        observations=[{'provider':'Pangram','pdf_sha256':'a'},{'provider':'Pangram','pdf_sha256':'b'},
            {'provider':'GPTZero','pdf_sha256':'g'}]
        first={'pdf_sha256':'a','account_batch':'first'};second={'pdf_sha256':'b','account_batch':'second'}
        self.assertEqual(account_batches(observations,{'accounts':[first,second]}),{'first':1,'second':1})
        for entries in ([first],[first,first,second],[first,second,{'pdf_sha256':'g','account_batch':'second'}]):
            with self.assertRaises(ValueError):account_batches(observations,{'accounts':entries})


if __name__=='__main__':unittest.main()
