"""Whole-paper isolation, outcome exclusions and source-only feature contracts."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fama_style.research import verified_merge_selector as module


class VerifiedMergeSelectorTests(unittest.TestCase):
    def rows(self):
        rows=[]
        for p in range(4):
            for i in range(6):
                group='primary' if i<4 else 'faithful_comparable' if i==4 else 'claim_shift'
                rows.append({'item_id':str(p)+'_'+str(i),'paper_id':str(p),'paragraph_id':'p'+str(p)+'_'+str(i//2),
                    'boundary_id':'B'+str(i%2+1),'features':[float(i%2)]*12,'pair_words':20+10*i,
                    'cohort':group,'label':i%2 if group=='primary' else None,
                    'fidelity':'local_shift' if group=='claim_shift' else 'faithful',
                    'readability':'comparable' if group=='faithful_comparable' else 'merged_better' if i%2 else 'separated_better'})
        return rows

    def test_cohort_selection_keeps_ties_and_claim_shifts_out_of_training(self):
        base={'fidelity_verdict':'faithful','readability_in_context':'merged_better'}
        self.assertEqual(module.cohort(base),('primary',1))
        base['readability_in_context']='separated_better';self.assertEqual(module.cohort(base),('primary',0))
        base['readability_in_context']='comparable';self.assertEqual(module.cohort(base),('faithful_comparable',None))
        base['fidelity_verdict']='local_shift';self.assertEqual(module.cohort(base),('claim_shift',None))
        with self.assertRaises(ValueError):module.fit_primary(self.rows())

    def test_held_paper_labels_cannot_change_its_threshold_or_predictions(self):
        rows=self.rows();first,folds=module.evaluate(rows);changed=copy.deepcopy(rows)
        for r in changed:
            if r['paper_id']=='0' and r['cohort']=='primary':r['label']=1-r['label']
        second,other=module.evaluate(changed)
        a=[(r['item_id'],r['method'],r['prediction'],r['probability']) for r in first if r['paper_id']=='0']
        b=[(r['item_id'],r['method'],r['prediction'],r['probability']) for r in second if r['paper_id']=='0']
        self.assertEqual(a,b);self.assertEqual(folds[0],other[0])
        for fold in folds:
            held=fold['outer_paper'];self.assertNotIn(held,fold['training_papers'])
            for trace in fold['selection']['inner_fit_trace']:
                self.assertNotIn(held,trace['training_papers']);self.assertNotIn(trace['inner_paper'],trace['training_papers'])
        self.assertEqual(len(first),24*4)

    def test_side_outcomes_cannot_change_fitted_models(self):
        rows=self.rows();_,first=module.evaluate(rows);changed=copy.deepcopy(rows)
        for r in changed:
            if r['cohort']!='primary':r['readability']='separated_better';r['features']=[999.0]*12
        _,second=module.evaluate(changed);self.assertEqual(first,second)

    def test_metrics_do_not_count_ties_or_shifts_as_negative_labels(self):
        rows=self.rows()
        for r in rows:r['prediction']=1
        score=module.score(rows)
        self.assertEqual((score['tp'],score['fp'],score['fn'],score['tn']),(8,8,0,0))
        self.assertEqual(score['primary_items'],16)
        self.assertEqual(score['side_cohorts']['faithful_comparable']['selected_merges'],4)
        self.assertEqual(score['side_cohorts']['claim_shift']['selected_merges'],4)

    def test_projection_ignores_candidate_reference_and_judge_text(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);verified=root/'verified';sources=root/'sources';verified.mkdir();sources.mkdir()
            source={'paragraph_id':'p','source_paragraph':'Firms invest. Costs fall.',
                'source_sentences':[{'sentence_id':'S1','exact_text':'Firms invest.'},{'sentence_id':'S2','exact_text':'Costs fall.'}],
                'original_paragraph':'POISON','source_claims':['POISON'],'boundaries':['POISON']}
            row={'item_id':'x','source_paragraph':source['source_paragraph'],'left_sentence':'Firms invest.',
                'right_sentence':'Costs fall.','candidate':'POISON','fidelity_verdict':'faithful',
                'readability_in_context':'merged_better','fidelity_rationale':'POISON','source_units':['POISON']}
            (verified/'PRIVATE_MAPPING.json').write_text(json.dumps([{'item_id':'x','paragraph_id':'p','paper_id':'paper','boundary_id':'B1'}]))
            (sources/'annotations.private.jsonl').write_text(json.dumps(source)+'\n')
            file=verified/'annotations.private.jsonl';file.write_text(json.dumps(row)+'\n');a=module.collect(verified,sources)
            row['candidate']='DIFFERENT POISON';row['fidelity_rationale']='DIFFERENT';file.write_text(json.dumps(row)+'\n')
            self.assertEqual(a,module.collect(verified,sources));self.assertNotIn('POISON',str(a))
            row['left_sentence']='Wrong source.';file.write_text(json.dumps(row)+'\n')
            with self.assertRaises(ValueError):module.collect(verified,sources)

    def test_tuning_rejects_held_paper_and_prefers_conservative_ties(self):
        with self.assertRaisesRegex(ValueError,'outer_paper'):module.tune(self.rows(),'0')
        candidates=[{'threshold':t,'metrics':{'macro_paper_accuracy':0.5,'selected_primary_merges':0}} for t in (0.5,0.8,1.0)]
        self.assertEqual(module.choose(candidates,lambda t:t)['threshold'],1.0)

    def test_changed_locked_configuration_stops_before_reading_or_fitting_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);verified=root/'verified';sources=root/'sources';verified.mkdir();sources.mkdir()
            for folder in (verified,sources):(folder/'manifest.json').write_text(json.dumps({'files':{}}))
            locked=copy.deepcopy(module.protocol(verified,sources));locked['config']['penalty']=0.01
            path=root/'protocol.json';path.write_text(json.dumps(locked))
            with self.assertRaisesRegex(ValueError,'locked_protocol_changed'):
                module.run(verified,sources,path,root/'out')
            self.assertFalse((root/'out').exists())
