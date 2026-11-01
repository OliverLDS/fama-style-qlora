import copy
import json
from pathlib import Path
import tempfile
import unittest

from fama_style.research.sft_strength_review import validate_execution,masked_comparison,summarize
from fama_style.training import sft_strength_pilot as runtime
from test_sft_strength_pilot import fixture


def execution_fixture(folder):
    _,_,dp,cp,_=fixture(folder);pairs,dev,cases,config=runtime.load_inputs(dp,cp)
    config['initial_adapter_tensor_sha256']='initial'
    report={'complete':True,'training_called':True,'config':config,'initial_adapter_tensor_sha256':'initial',
        'final_adapter_tensor_sha256':'final','control_exact_matches':8,'runtime':{k:config[k] for k in
            ('packages','model_id','model_revision','runtime_source_sha256','prior_runtime_sha256','token_helper_sha256')},
        'likelihood_sanity':[{'passed':True,'manual_mean_nll':2.,'builtin_mean_nll':2.} for _ in range(2)],
        'control_generations':[],'final_generations':[],'forward_counts':{'generation_requests':27},
        **{k:True for k in ('optimizer_initially_empty','weights_changed','adapter_reload_weights_match','final_evaluation_frozen')}}
    report['runtime'].update(gpu_count=1,gpu='Synthetic T4')
    for stage in ('initial','final'):
        for split,rows in (('training',pairs),('development',dev)):
            key='pair_id' if split=='training' else 'example_id'
            report[stage+'_'+split+'_scores']=[{key:r[key],'total_log_probability':-4.,'supervised_tokens':2,'mean_nll':2.} for r in rows]
    for row in dev:
        row['expected_generation']={'output':'Synthetic control '+row['example_id'],'output_tokens':5,'empty':False,'output_limited':False,'copied_source':False}
        report['control_generations'].append({'example_id':row['example_id'],**row['expected_generation']})
    references=[]
    for split,rows in (('training',cases),('development',dev)):
        for row in rows:
            cid=row['pair_id'] if split=='training' else row['example_id'];text='Synthetic output '+cid
            report['final_generations'].append({'split':split,'example_id':row['example_id'],'pair_id':row.get('pair_id'),
                'source_sha256':row['source_sha256'],'prompt_sha256':row['prompt_sha256'],'output':text,
                'output_sha256':runtime.prior.sha(text),'output_tokens':5,'empty':False,'output_limited':False,'copied_source':False})
            references.append({'split':split,'case_id':cid,'example_id':row['example_id'],'paper_id':row['paper_id'],
                'source':row['source'],'target':'Synthetic target','control':{'output':text},'weak_sft':{'output':text}})
    sequences=[(r['pair_id'],r['encoded']['chosen']) for r in pairs]+[(r['example_id'],r['encoded']) for r in dev]
    audits=[{'id':i,'input_ids_sha256':runtime.prior.sha(json.dumps(s['input_ids'])),
        'labels_sha256':runtime.prior.sha(json.dumps(s['labels'])),'prompt_tokens':2,'full_tokens':4} for i,s in sequences]
    gradients=[{'step':i,'finite':True,'norm':1.} for i in range(1,23)]
    updates=[{'step':11*(e['epoch']-1)+i+1,'epoch':e['epoch'],'group_id':g['group_id'],'pair_ids':g['pair_ids'],
        'learning_rate':runtime.learning_rate(11*(e['epoch']-1)+i+1),'loss':1.}
        for e in config['group_schedule'] for i,g in enumerate(e['groups'])]
    state={'complete':True,'phase':'complete','training_called':True,'optimizer_steps':22,'elapsed_seconds':100.,'peak_gpu_allocated_bytes':1024**3}
    return [report,state,config,pairs,dev,cases,audits,gradients,updates],references


class SFTStrengthReviewTests(unittest.TestCase):
    def test_execution_rejects_partial_wrong_rate_and_unfrozen_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            original,_=execution_fixture(Path(temp));validate_execution(*original)
            changes=[lambda a:a[1].update(optimizer_steps=21),lambda a:a[0].update(final_evaluation_frozen=False),
                lambda a:a[8][0].update(learning_rate=1e-6),lambda a:a[7][0].update(norm=float('nan')),
                lambda a:a[0]['final_generations'].pop(),lambda a:a[6][0].update(labels_sha256='changed'),
                lambda a:a[0]['initial_training_scores'][0].update(total_log_probability=-3.,mean_nll=1.5)]
            for i,change in enumerate(changes):
                with self.subTest(change=i):
                    altered=copy.deepcopy(original);change(altered)
                    with self.assertRaises(ValueError):validate_execution(*altered)

    def test_masking_and_split_counts_and_equal_text_ranks(self):
        with tempfile.TemporaryDirectory() as temp:
            args,refs=execution_fixture(Path(temp));bundle,key=masked_comparison(args[0],refs)
            self.assertEqual((bundle,key),masked_comparison(args[0],refs))
            judgments={'cases':[{'split':r['split'],'case_id':r['case_id'],'ranking_best_first':[['A','B','C']],
                'outputs':{l:{'output_sha256':runtime.prior.sha(t),'material_claim_change':False,'minor_semantic_shifts':0,
                    'unresolved':False,'notes':'Synthetic faithful output'} for l,t in r['outputs'].items()}} for r in bundle]}
            summary,result=summarize(bundle,key,judgments)
            self.assertEqual(len(result),57);self.assertEqual(summary['training']['strong_sft']['outputs'],11)
            self.assertEqual(summary['development']['strong_sft']['eligible_within_local_budget'],8)
            judgments['cases'][0]['ranking_best_first']=[['A'],['B','C']]
            with self.assertRaises(ValueError):summarize(bundle,key,judgments)

    def test_judgment_hash_must_match_evaluated_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            args,refs=execution_fixture(Path(temp));bundle,key=masked_comparison(args[0],refs)
            judgments={'cases':[{'split':r['split'],'case_id':r['case_id'],'ranking_best_first':[['A','B','C']],
                'outputs':{l:{'output_sha256':'wrong','material_claim_change':False,'minor_semantic_shifts':0,
                    'unresolved':False,'notes':'Synthetic note'} for l in 'ABC'}} for r in bundle]}
            with self.assertRaises(ValueError):summarize(bundle,key,judgments)


if __name__=='__main__':unittest.main()
