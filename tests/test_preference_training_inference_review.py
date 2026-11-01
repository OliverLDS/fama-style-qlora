import copy
import json
from pathlib import Path
import tempfile
import unittest

from fama_style.research.preference_training_inference_review import validate_execution,masked_comparison,summarize
from fama_style.evaluation import preference_training_inference as runtime
from test_preference_training_inference import fixture


def execution_fixture(folder):
    jobs,config,_,_,_=fixture(folder);config['packages']={'torch':'synthetic'}
    snaps=[]
    for arm in ('sft','dpo'):
        generations=[]
        for job in jobs:
            text='Synthetic output '+job['pair_id']
            g={k:job[k] for k in ('pair_id','example_id','anchor_group_id','paper_id','input_kind','source_sha256','prompt_sha256')}
            g.update(output=text,output_sha256=runtime.sha(text),output_tokens=8,empty=False,output_limited=False,copied_source=False)
            generations.append(g)
        snaps.append({'arm':arm,'complete':True,'all_parameters_frozen':True,'checkpoint_epoch':2,'generations':generations})
    report={'complete':True,'mode':'inference_only','config':config,'training_called':False,'optimizer_steps':0,
        'runtime':{k:config[k] for k in ('model_id','model_revision','packages','runtime_source_sha256')},'snapshots':snaps}
    report['runtime'].update(gpu='Synthetic T4',gpu_count=1)
    state={'complete':True,'phase':'complete','training_called':False,'optimizer_steps':0,'completed_generations':22,
        'elapsed_seconds':100.,'peak_gpu_allocated_bytes':1024**3}
    audits=[{'pair_id':j['pair_id'],'prefix_ids_sha256':j['prefix_ids_sha256'],'prompt_tokens':len(j['prefix_ids']),
        'matches_training_prefix':True} for j in jobs]
    refs=[{**j,'chosen':'Synthetic human paragraph','control':{'output':'Synthetic output '+j['pair_id']}} for j in jobs]
    return jobs,config,report,state,audits,refs


class PreferenceTrainingInferenceReviewTests(unittest.TestCase):
    def test_execution_rejects_training_incomplete_outputs_and_changed_bindings(self):
        changes=[lambda x:x[3].update(training_called=True),lambda x:x[3].update(optimizer_steps=1),
            lambda x:x[3].update(completed_generations=21),lambda x:x[2]['snapshots'][0]['generations'].pop(),
            lambda x:x[2]['snapshots'][1].update(all_parameters_frozen=False),
            lambda x:x[4][0].update(prefix_ids_sha256='changed'),
            lambda x:x[2]['snapshots'][0]['generations'][0].update(output='changed'),
            lambda x:x[2]['snapshots'][0]['generations'][0].update(prompt_sha256='changed')]
        with tempfile.TemporaryDirectory() as temp:
            original=execution_fixture(Path(temp));validate_execution(*original[:5])
            for i,change in enumerate(changes):
                with self.subTest(change=i):
                    altered=copy.deepcopy(original);change(altered)
                    with self.assertRaises(ValueError):validate_execution(*altered[:5])

    def test_masking_reproducible_and_equal_outputs_must_tie(self):
        with tempfile.TemporaryDirectory() as temp:
            _,_,report,_,_,refs=execution_fixture(Path(temp));bundle,key=masked_comparison(report,refs)
            self.assertEqual((bundle,key),masked_comparison(report,refs))
            judgments={'cases':[{'pair_id':row['pair_id'],'ranking_best_first':[['A','B','C']],
                'outputs':{l:{'output_sha256':runtime.sha(t),'material_claim_change':True,'unresolved':False,
                    'minor_semantic_shifts':0,'notes':'Synthetic persistent claim error'} for l,t in row['outputs'].items()}}
                for row in bundle]}
            summary,outputs=summarize(bundle,key,judgments)
            self.assertEqual(len(outputs),33)
            self.assertEqual(summary['arms']['dpo']['exact_control_outputs'],11)
            self.assertEqual(summary['arms']['sft']['material_errors_resolved'],0)
            judgments['cases'][0]['ranking_best_first']=[['A'],['B','C']]
            with self.assertRaises(ValueError):summarize(bundle,key,judgments)

    def test_assessments_bind_to_exact_output_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            _,_,report,_,_,refs=execution_fixture(Path(temp));bundle,key=masked_comparison(report,refs)
            judgments={'cases':[{'pair_id':row['pair_id'],'ranking_best_first':[['A','B','C']],
                'outputs':{l:{'output_sha256':'wrong','material_claim_change':False,'unresolved':False,
                    'minor_semantic_shifts':0,'notes':'Synthetic note'} for l in 'ABC'}} for row in bundle]}
            with self.assertRaises(ValueError):summarize(bundle,key,judgments)


if __name__=='__main__':unittest.main()
