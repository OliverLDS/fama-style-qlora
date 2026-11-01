import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from fama_style.research.dpo_strength_review import validate_execution,masked_comparison,summarize
from fama_style.training import dpo_strength_pilot as runtime
from test_sft_strength_review import execution_fixture as sft_fixture


def fixture(folder):
    args,refs=sft_fixture(folder)
    report,state,config,pairs,dev,cases,_,gradients,updates=args
    config.update(hyperparameters=runtime.HYPER,runtime_source_sha256=runtime.prior.file_sha(Path(runtime.__file__)))
    report['runtime']['runtime_source_sha256']=config['runtime_source_sha256']
    reference_scores={p['pair_id']:{k:dict(config['initial_chosen_scores'][p['pair_id']]) for k in ('chosen','rejected')} for p in pairs}
    config['reference_scores']=copy.deepcopy(reference_scores)
    cache_hash=runtime.prior.sha(json.dumps(reference_scores,sort_keys=True))
    reference={'scores':reference_scores,'cache_sha256':cache_hash,'initial_adapter_tensor_sha256':'initial',
        'source_adapter_sha256':config['adapter_sha256'],'dataset_sha256':config['dataset_sha256'],
        'model_id':config['model_id'],'model_revision':config['model_revision'],'reference_policy':'unchanged_original_expanded_epoch_2'}
    report['reference_cache_sha256']=cache_hash
    report['initial_policy_reference_identity']=[{'pair_id':p['pair_id'],'max_logp_difference':0.,'dpo_loss':math.log(2)} for p in pairs]
    report['final_preference_scores']=[{'pair_id':p['pair_id'],**copy.deepcopy(reference_scores[p['pair_id']])} for p in pairs]
    report['likelihood_sanity']=[report['likelihood_sanity'][0],copy.deepcopy(report['likelihood_sanity'][0]),report['likelihood_sanity'][1]]
    for update in updates:update['reference_cache_sha256']=cache_hash
    seqs=[(p['pair_id']+'__'+k,p['encoded'][k]) for p in pairs for k in ('chosen','rejected')]+[(r['example_id'],r['encoded']) for r in dev]
    audits=[{'id':i,'input_ids_sha256':runtime.prior.sha(json.dumps(s['input_ids'])),
        'labels_sha256':runtime.prior.sha(json.dumps(s['labels'])),'prompt_tokens':2,'full_tokens':4} for i,s in seqs]
    refs=[{**{k:v for k,v in r.items() if k!='weak_sft'},'strong_sft':r['weak_sft']} for r in refs]
    return [report,state,config,pairs,dev,cases,audits,gradients,updates,reference],refs


class DPOStrengthReviewTests(unittest.TestCase):
    def test_shared_projection_keeps_original_report_and_negative_audits(self):
        with tempfile.TemporaryDirectory() as temp:
            args,_=fixture(Path(temp));original=copy.deepcopy(args);validate_execution(*args)
            self.assertEqual(args,original)
            args[6][1]['labels_sha256']='corrupted rejected mask'
            with self.assertRaises(ValueError):validate_execution(*args)

    def test_reference_identity_and_each_update_cache_are_checked(self):
        mutations=[lambda a:a[0]['initial_policy_reference_identity'][0].update(max_logp_difference=.1),
            lambda a:a[8][0].update(reference_cache_sha256='changed'),
            lambda a:a[9].update(reference_policy='raw_base'),
            lambda a:a[0]['final_preference_scores'][0]['rejected'].update(supervised_tokens=3),
            lambda a:a[0]['likelihood_sanity'][1].update(builtin_mean_nll=9.)]
        with tempfile.TemporaryDirectory() as temp:
            original,_=fixture(Path(temp))
            for i,mutate in enumerate(mutations):
                with self.subTest(mutation=i):
                    args=copy.deepcopy(original);mutate(args)
                    with self.assertRaises(ValueError):validate_execution(*args)

    def test_condition_aliases_restore_actual_comparators_and_equal_texts_tie(self):
        with tempfile.TemporaryDirectory() as temp:
            args,refs=fixture(Path(temp));bundle,key=masked_comparison(args[0],refs)
            self.assertEqual((bundle,key),masked_comparison(args[0],refs))
            judgments={'cases':[{'split':r['split'],'case_id':r['case_id'],'ranking_best_first':[['A','B','C']],
                'outputs':{l:{'output_sha256':runtime.prior.sha(t),'material_claim_change':True,'minor_semantic_shifts':0,
                    'unresolved':False,'notes':'Synthetic persistent error'} for l,t in r['outputs'].items()}} for r in bundle]}
            summary,result=summarize(bundle,key,judgments)
            self.assertEqual(set(summary['training']),{'control','strong_sft','strong_dpo'})
            self.assertEqual(summary['training']['strong_dpo']['generic_expression_vs_strong_sft']['tie'],11)
            self.assertEqual(summary['development']['strong_dpo']['generic_expression_vs_strong_sft']['tie'],8)
            self.assertEqual({r['arm'] for r in result},{'control','strong_sft','strong_dpo'})
            judgments['cases'][0]['ranking_best_first']=[['A'],['B','C']]
            with self.assertRaises(ValueError):summarize(bundle,key,judgments)


if __name__=='__main__':unittest.main()
