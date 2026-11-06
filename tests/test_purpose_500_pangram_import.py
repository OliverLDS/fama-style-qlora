import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('purpose_detector_import',ROOT/'scripts/data/import_purpose_500_pangram.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def panel():
    observations=[];coverage=[]
    for i,group in enumerate(['Fama_coauthors','Shiller','Holmstrom','Baker_Wurgler']):
        for route in ['gemini_input','fresh_reconstruction']:
            for condition in ['input','base','epoch_1']:
                failed=group=='Shiller' and route=='fresh_reconstruction' and condition=='epoch_1'
                coverage.append(dict(content_id=str(i),input_condition=route,condition=condition))
                if failed:continue
                human=group=='Fama_coauthors' and route=='gemini_input' and condition!='base'
                observations.append(dict(content_id=str(i),style_group=group,input_condition=route,condition=condition,
                    provider_label='Human Written' if human else 'AI Assisted',entirely_human=human,
                    human_written_document_percent=100. if human else .1,ai_generated_document_percent=.1,
                    ai_assisted_document_percent=.1 if human else 100.,report_date='2026-11-06',provider_version='4.1',
                    full_frozen_text_scanned=not(condition=='epoch_1' and ((group=='Shiller' and route=='gemini_input') or (group=='Baker_Wurgler' and route=='fresh_reconstruction')))))
    return observations,coverage


class PangramImportTests(unittest.TestCase):
    def test_failed_case_has_no_fabricated_score_or_conversion(self):
        observations,coverage=panel();summary=m.compare(observations,coverage)
        self.assertEqual(summary['scored_texts'],23)
        self.assertEqual(summary['paired']['scored_pairs'],7)
        self.assertEqual(summary['paired']['new_entirely_human_conversions'],0)
        failed=next(r for r in summary['pairs'] if r['adapter_task_failed'])
        self.assertIsNone(failed['conditions']['epoch_1'])
        self.assertIsNone(failed['adapter_minus_input_human_share'])

    def test_AI_assisted_is_not_human_and_body_only_sensitivity_is_separate(self):
        observations,coverage=panel();summary=m.compare(observations,coverage)
        fresh=summary['conditions']['fresh_reconstruction']['epoch_1']
        self.assertEqual(fresh['native_labels'],{'AI Assisted':3})
        self.assertEqual(fresh['entirely_human'],0)
        self.assertEqual(summary['strict_full_text_sensitivity']['scored_pairs'],5)
        self.assertEqual(summary['body_only_reports'],2)

    def test_duplicate_observation_is_not_selected_for_a_better_result(self):
        observations,coverage=panel();observations.append(dict(observations[0],human_written_document_percent=100.))
        with self.assertRaisesRegex(ValueError,'duplicate_scan'):m.compare(observations,coverage)

    def test_prefix_only_deviation_is_audited_without_editing_frozen_text(self):
        body='The empirical result distinguishes two groups and reports their different outcomes.'
        original='Rewritten Paragraph:\n\n'+body
        card=dict(scan_id='SCAN_'+'a'*16,condition='epoch_1',text=original,text_sha256=m.packet.review.runtime.sha(original))
        page=dict(text=body,layout_pages=[body])
        matched,method,audit=m.bind_with_prefix_audit(page,[card],'b'*64,card['scan_id']+'.pdf')
        self.assertEqual(matched['text'],body)
        self.assertEqual(card['text'],original)
        self.assertFalse(audit['model_output_edited'])
        self.assertEqual(audit['full_model_output_sha256'],card['text_sha256'])
        self.assertTrue(method.startswith('body_only'))

    def test_substantive_body_changes_cannot_be_accepted_as_prefix_removal(self):
        body='Group one earns higher returns than group two in the reported results.'
        card=dict(scan_id='SCAN_'+'a'*16,condition='epoch_1',text='Rewritten Paragraph:\n\n'+body,text_sha256='f'*64)
        changed=body.replace('higher','lower');page=dict(text=changed,layout_pages=[changed])
        with self.assertRaisesRegex(ValueError,'unique_complete_text_binding'):m.bind_with_prefix_audit(page,[card],'b'*64,card['scan_id']+'.pdf')


if __name__=='__main__':unittest.main()
