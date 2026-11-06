import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('purpose_pangram',ROOT/'scripts/data/prepare_purpose_500_pangram.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def examples():
    items=[];prior=[]
    for i,group in enumerate(['Fama_coauthors','Shiller','Holmstrom','Baker_Wurgler']):
        for route in ['gemini_input','fresh_reconstruction']:
            source=('Synthetic source content %d %s. '%(i,route))*15
            for snapshot in ['base','epoch_1']:
                output=('Synthetic output %d %s %s. '%(i,route,snapshot))*15
                if group=='Shiller' and route=='fresh_reconstruction' and snapshot=='epoch_1':
                    output='Please supply the original passage for rewriting.'
                items.append(dict(content_id='case_'+str(i),style_group=group,paper_id='paper_'+str(i),
                    input_condition=route,prompt_condition='reexpress',snapshot=snapshot,
                    source=source,source_sha256=m.review.runtime.sha(source),reference='Reference '+str(i),
                    prompt_sha256=m.review.runtime.sha('Prompt '+source),output=output))
            if route=='gemini_input':prior.append(dict(text_sha256=m.review.runtime.sha(source),
                provider_label='Human Written' if i==0 else 'AI Generated',human_written_document_percent=100 if i==0 else .1,
                ai_generated_document_percent=.1 if i==0 else 100,ai_assisted_document_percent=.1,
                entirely_human=i==0))
    return items,prior


class PurposePangramTests(unittest.TestCase):
    def test_reuses_inputs_and_retains_failed_case_without_human_rescans(self):
        items,prior=examples();cards=m.select(items,prior)
        self.assertEqual(len(cards),24)
        self.assertEqual(sum(r['status']=='new_scan_required' for r in cards),19)
        self.assertEqual(sum(r['status']=='reused_exact_prior_observation' for r in cards),4)
        failed=[r for r in cards if r['status']=='not_scanned_task_failure']
        self.assertEqual(len(failed),1)
        self.assertEqual(failed[0]['credits_estimate'],0)
        self.assertEqual({r['condition'] for r in cards},{'input','base','epoch_1'})
        ceiling=[r for r in cards if r['prior_observation'] and r['prior_observation']['entirely_human']]
        self.assertEqual(len(ceiling),1)

    def test_exact_text_changes_prevent_reusing_old_input_score(self):
        items,prior=examples();prior[0]['text_sha256']='0'*64
        cards=m.select(items,prior)
        self.assertEqual(sum(r['status']=='new_scan_required' for r in cards),20)

    def test_conflicting_prior_scores_are_not_selected_for_favorable_result(self):
        items,prior=examples();prior.append(dict(prior[0],human_written_document_percent=1))
        with self.assertRaisesRegex(ValueError,'conflicting_historical_scores'):m.select(items,prior)

    def test_html_preserves_full_text_without_script_injection(self):
        cards=[dict(scan_id='SCAN_one',text='Exact <script>content</script> with a heading.',words_estimate=6,credits_estimate=1)]
        page=m.render(cards,1,1)
        self.assertIn('\\u003cscript\\u003e',page)
        self.assertNotIn('Exact <script>content',page)
        self.assertIn('pdf_reports',page)


if __name__=='__main__':unittest.main()
