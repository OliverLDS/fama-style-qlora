import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from fama_style.research.preference_pilot_review import validate_execution,summarize,tensor_fingerprint
from fama_style.training import preference_pair_pilot as p
from test_preference_pair_pilot import fixture


def execution_fixture(folder):
    data,config,_,_,_=fixture(folder);pairs=data['training_pairs'];dev=data['development']
    config.update(packages={'torch':'synthetic'},adapter_sha256={'adapter_model.safetensors':'synthetic'})
    scores={r['pair_id']:{k:{'total_log_probability':-4.,'mean_nll':2.,'supervised_tokens':2}
        for k in ('chosen','rejected')} for r in pairs}
    cache=p.sha(json.dumps(scores,sort_keys=True))
    reference={'scores':scores,'cache_sha256':cache,'dataset_sha256':config['dataset_sha256'],
        'source_adapter_sha256':config['adapter_sha256'],'model_id':p.MODEL_ID,'model_revision':p.REVISION,
        'token_helper_sha256':config['token_helper_sha256'],'initial_adapter_tensor_sha256':'initial'}
    snapshots=[]
    for arm,epoch,step in [('control',0,0),('sft',1,11),('sft',2,22),('dpo',1,11),('dpo',2,22)]:
        generations=[];losses=[]
        for r in dev:
            generation={k:r[k] for k in ('example_id','source_sha256','prompt_sha256')}
            generation.update(output='Synthetic output '+r['example_id'],output_tokens=5,empty=False,output_limited=False,copied_source=False)
            if arm=='control':r['expected_generation']={k:generation[k] for k in ('output','output_tokens','empty','output_limited','copied_source')}
            generations.append(generation);losses.append({'example_id':r['example_id'],'total_log_probability':-4.,'mean_nll':2.,'supervised_tokens':2})
        snapshots.append({'arm':arm,'epoch':epoch,'optimizer_step':step,'complete':True,
            'generations':generations,'losses':losses,'token_weighted_development_nll':2.})
    arms={a:{'optimizer_initially_empty':True,'reload_weights_match':True,'weights_changed':True,
        'completed_updates':22,'initial_adapter_tensor_sha256':'initial','final_adapter_tensor_sha256':a,
        'reference_identity':[{'pair_id':r['pair_id'],'max_logp_difference':0.,'dpo_loss':math.log(2)} for r in pairs],
        'final_training_scores':[{'pair_id':r['pair_id'],**scores[r['pair_id']]} for r in pairs]} for a in ('sft','dpo')}
    report={'complete':True,'training_called':True,'config':config,'runtime':{
        'packages':config['packages'],'runtime_source_sha256':config['runtime_source_sha256'],
        'token_helper_sha256':config['token_helper_sha256'],'model_id':p.MODEL_ID,'model_revision':p.REVISION,
        'gpu_count':1,'gpu':'Synthetic T4'},'common_control_exact_matches':8,
        'likelihood_sanity':[{'passed':True,'manual_mean_nll':2.,'builtin_mean_nll':2.} for _ in range(2)],
        'snapshots':snapshots,'initial_adapter_tensor_sha256':'initial','frozen_reference_cache_sha256':cache,
        'reference_scores':[{'pair_id':r['pair_id'],**scores[r['pair_id']]} for r in pairs],'arms':arms}
    state={'complete':True,'phase':'complete','optimizer_steps':44,'arm_steps':{'sft':22,'dpo':22},
        'training_called':True,'elapsed_seconds':100.,'peak_gpu_allocated_bytes':1024**3}
    updates=[{'arm':a,'epoch':e['epoch'],'step':11*(e['epoch']-1)+i+1,'group_id':g['group_id'],'rows':len(g['pair_ids']),
        'loss':1.,'learning_rate':1e-6*(1-(11*(e['epoch']-1)+i)/22),'reference_cache_sha256':cache}
        for a in ('sft','dpo') for e in config['group_schedule'] for i,g in enumerate(e['groups'])]
    gradients=[{'arm':r['arm'],'step':r['step'],'finite':True,'norm':1.} for r in updates]
    processor=[{'pair_id':r['pair_id'],'candidate':k,**r['token_audits'][k]} for r in pairs for k in ('chosen','rejected')]
    processor +=[{'example_id':r['example_id'],'candidate':'development_target',
        'input_ids_sha256':p.sha(json.dumps(r['encoded']['input_ids'])),'labels_sha256':p.sha(json.dumps(r['encoded']['labels']))} for r in dev]
    return [report,state,config,pairs,dev,gradients,updates,processor,reference]


class PreferencePilotReviewTests(unittest.TestCase):
    def test_execution_contract_rejects_partial_corrupted_and_unmatched_runs(self):
        changes=[lambda a:a[1].update(complete=False),lambda a:a[1].update(optimizer_steps=43),
            lambda a:a[0]['snapshots'][0]['generations'][0].update(output='Changed control'),
            lambda a:a[0]['snapshots'][1]['losses'].pop(),lambda a:a[5][0].update(norm=float('nan')),
            lambda a:a[6][0].update(rows=99),lambda a:a[7][0].update(labels_sha256='changed'),
            lambda a:a[8]['scores'][next(iter(a[8]['scores']))]['chosen'].update(total_log_probability=-3.),
            lambda a:a[0]['arms']['dpo'].update(initial_adapter_tensor_sha256='sft-final'),
            lambda a:a[0]['arms']['sft']['reference_identity'][0].update(max_logp_difference=.1),
            lambda a:a[0]['likelihood_sanity'][0].update(builtin_mean_nll=3.)]
        with tempfile.TemporaryDirectory() as temp:
            original=execution_fixture(Path(temp));validate_execution(*original)
            for change in changes:
                with self.subTest(change=changes.index(change)):
                    altered=copy.deepcopy(original);change(altered)
                    with self.assertRaises(ValueError):validate_execution(*altered)

    def test_equal_texts_cannot_receive_different_generic_style_ranks(self):
        with tempfile.TemporaryDirectory() as temp:
            report,_,_,_,dev,*_=execution_fixture(Path(temp));key=[];cases=[]
            for row in dev:
                text=row['expected_generation']['output'];digest=p.sha(text)
                for label,arm in zip('ABC',('control','sft','dpo')):
                    key.append({'example_id':row['example_id'],'label':label,'arm':arm,'output_sha256':digest})
                cases.append({'example_id':row['example_id'],'ranking_best_first':[['A','B','C']],
                    'outputs':{l:{'text_sha256':digest,'material_claim_change':False,'unresolved':False,'minor_semantic_shifts':0} for l in 'ABC'}})
            summary,_=summarize(report,dev,key,{'cases':cases})
            self.assertEqual(summary['primary_arms']['dpo']['exact_control_generations'],8)
            self.assertFalse(summary['primary_arms']['dpo']['predeclared_quality_gate_passed'])
            cases[0]['ranking_best_first']=[['A'],['B','C']]
            with self.assertRaises(ValueError):summarize(report,dev,key,{'cases':cases})

    def test_tensor_fingerprint_rejects_truncated_weights_without_ml(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'bad.safetensors';path.write_bytes((100).to_bytes(8,'little')+b'{}')
            with self.assertRaises(ValueError):tensor_fingerprint(path)


if __name__=='__main__':unittest.main()
