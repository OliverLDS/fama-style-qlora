"""Freeze a single SFT learning-rate intervention and archived paired controls."""
import argparse
import json
from pathlib import Path

from fama_style.training import sft_strength_pilot as runtime
from .author_control_replication import verify_files


def rows(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build(root,output):
    if output.exists():raise ValueError('preserve_existing_strength_packet')
    prior=runtime.prior
    old=root/'data/actual_error_pilot_v1';training=root/'data/preference_training_inference_v1'
    old_review=root/'research_runs/actual_error_pilot_v1/review_v1/results'
    seen_review=root/'research_runs/preference_training_inference_v1/review_v1/results'
    old_raw=root/'research_runs/actual_error_pilot_v1/kaggle_v1'
    seen_raw=root/'research_runs/preference_training_inference_v1/kaggle_v1'
    for directory in (old,training,old_review,seen_review):verify_files(directory)
    for review,raw in ((old_review,old_raw),(seen_review,seen_raw)):
        receipt=json.loads((review/'receipt.json').read_text())
        for name,digest in receipt['download_sha256'].items():
            if prior.file_sha(raw/name)!=digest:raise ValueError('reviewed_run_artifact_changed')
    dp=old/'upload/actual_error_pilot_data.json';cp=old/'upload/actual_error_pilot_config.json'
    pairs,development,previous=prior.load_inputs(dp,cp)
    report=json.loads((old_raw/'actual_error_pilot_v1/evaluation.json').read_text())
    seen=json.loads((seen_raw/'preference_training_inference_v1/evaluation.json').read_text())
    if not report['complete'] or not seen['complete']:raise ValueError('completed_comparisons_required')
    cases=rows(training/'upload/preference_training_jobs.jsonl');references=rows(training/'private_key/reference_key.private.jsonl')
    weak_training={g['pair_id']:g for snap in seen['snapshots'] if snap['arm']=='sft' for g in snap['generations']}
    weak_dev={g['example_id']:g for snap in report['snapshots'] if snap['arm']=='sft' and snap['epoch']==2 for g in snap['generations']}
    private=[]
    for row in references:
        private.append({'split':'training','case_id':row['pair_id'],'example_id':row['example_id'],'paper_id':row['paper_id'],
            'family':row['family'],'source':row['source'],'target':row['chosen'],'control':row['control'],
            'weak_sft':weak_training[row['pair_id']],
            'inherited_needs_review':row['inherited_needs_review'],'inherited_needs_external_context':row['inherited_needs_external_context']})
    for row in development:
        private.append({'split':'development','case_id':row['example_id'],'example_id':row['example_id'],'paper_id':row['paper_id'],
            'source':row['source'],'target':row['target'],'control':row['expected_generation'],'weak_sft':weak_dev[row['example_id']],
            'inherited_needs_review':row['needs_review'],'inherited_needs_external_context':row['needs_external_context']})
    output.mkdir(parents=True);upload=output/'upload';upload.mkdir()
    data_path=upload/'sft_strength_data.json';config_path=upload/'sft_strength_config.json'
    prior.write_json(data_path,{'pilot_data_json':dp.read_text(),'pilot_config_json':cp.read_text(),'training_cases':cases})
    config={'protocol':runtime.PROTOCOL,'hyperparameters':runtime.HYPER,'training_authorized':True,'max_runtime_seconds':1800,
        'model_id':previous['model_id'],'model_revision':previous['model_revision'],'adapter_sha256':previous['adapter_sha256'],
        'packages':previous['packages'],'decoding':previous['decoding'],'group_schedule':previous['group_schedule'],
        'expected_generation_config':previous['expected_generation_config'],'initial_adapter_tensor_sha256':report['initial_adapter_tensor_sha256'],
        'dataset_sha256':prior.file_sha(data_path),'prior_data_sha256':prior.file_sha(dp),'prior_config_sha256':prior.file_sha(cp),
        'runtime_source_sha256':prior.file_sha(Path(runtime.__file__)),'prior_runtime_sha256':prior.file_sha(Path(prior.__file__)),
        'token_helper_sha256':prior.file_sha(Path(prior.tokens.__file__)),'training_case_ids':[r['pair_id'] for r in cases],
        'initial_chosen_scores':{r['pair_id']:r['chosen'] for r in report['reference_scores']}}
    prior.write_json(config_path,config);runtime.load_inputs(data_path,config_path)
    private_dir=output/'private_key';private_dir.mkdir()
    (private_dir/'comparisons.private.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in private))
    plan={'protocol':runtime.PROTOCOL,'status':'prepared_not_executed','training_authorized':True,'optimizer_updates':22,
        'training_pairs':17,'paragraph_groups':11,'epochs':2,'learning_rate':1e-5,'previous_learning_rate':1e-6,
        'optimization_change':'Only learning rate; same initial adapter, masks, data, group order, dropout and optimizer settings.',
        'dpo_training':False,'primary_epoch':2,'epoch_1':'checkpoint only; no generation or model selection',
        'training_generation_cases':11,'development_generation_cases':8,'final_generations':19,
        'initial_control_generations':8,'total_new_generations':27,'archived_comparison_outputs':38,
        'maximum_runtime_seconds':1800,'reserved_test_records_read':0,'detector_calls':0,
        'dataset_sha256':config['dataset_sha256'],'config_sha256':prior.file_sha(config_path),
        'runtime_source_sha256':config['runtime_source_sha256'],'prior_runtime_sha256':config['prior_runtime_sha256'],
        'token_helper_sha256':config['token_helper_sha256']}
    prior.write_json(output/'plan.json',plan)
    prior.write_json(output/'authorization.json',{'user_message':'OK go ahead','user_authorized':True,
        'scope':'Single stronger SFT continuation at 1e-5; same 17 pairs, 22 updates, original starting adapter and fixed final evaluation.',
        'maximum_runtime_seconds':1800,'optimizer_updates':22,'dpo_training_authorized':False,'full_training_authorized':False,
        'dataset_sha256':config['dataset_sha256'],'config_sha256':prior.file_sha(config_path),
        'runtime_source_sha256':config['runtime_source_sha256'],'submission_method':'manual Kaggle UI; enable HF_TOKEN before execution'})
    prior.write_json(output/'receipt.json',{'builder_sha256':prior.file_sha(Path(__file__)),
        'source_manifests_sha256':{str(p.relative_to(root)/'manifest.json'):prior.file_sha(p/'manifest.json') for p in (old,training,old_review,seen_review)}})
    prior.write_json(output/'manifest.json',{'files':{str(p.relative_to(output)):prior.file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return plan


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.root,a.output),indent=2))
