"""Freeze a matched DPO-only run against the completed stronger SFT baseline."""
import argparse
import json
from pathlib import Path

from fama_style.training import dpo_strength_pilot as runtime
from .author_control_replication import verify_files


def build(root,output):
    if output.exists():raise ValueError('preserve_existing_DPO_packet')
    prior=runtime.prior;sft=root/'data/sft_strength_v1'
    review=root/'research_runs/sft_strength_v1/review_v1/results';raw=root/'research_runs/sft_strength_v1/kaggle_v1'
    weak_review=root/'research_runs/actual_error_pilot_v1/review_v1/results'
    weak_raw=root/'research_runs/actual_error_pilot_v1/kaggle_v1'
    for folder in (sft,review,weak_review):verify_files(folder)
    for checked,download in ((review,raw),(weak_review,weak_raw)):
        receipt=json.loads((checked/'receipt.json').read_text())
        for name,digest in receipt['download_sha256'].items():
            if prior.file_sha(download/name)!=digest:raise ValueError('reviewed_training_output_changed')
    checked=json.loads((review/'summary.json').read_text());report=json.loads((raw/'sft_strength_v1/evaluation.json').read_text())
    if not checked['execution_complete'] or checked['optimizer_updates']!=22 or not report['complete']:
        raise ValueError('completed_stronger_SFT_required')
    config=json.loads((sft/'upload/sft_strength_config.json').read_text())
    reference=json.loads((weak_raw/'actual_error_pilot_v1/reference_scores.json').read_text())
    if (reference['initial_adapter_tensor_sha256']!=config['initial_adapter_tensor_sha256']
            or reference['source_adapter_sha256']!=config['adapter_sha256']
            or reference['model_id']!=config['model_id'] or reference['model_revision']!=config['model_revision']
            or prior.sha(json.dumps(reference['scores'],sort_keys=True))!=reference['cache_sha256']):
        raise ValueError('original_starting_policy_reference_required')
    if {k:v for k,v in runtime.HYPER.items() if k!='beta'}!=config['hyperparameters']:
        raise ValueError('DPO_settings_do_not_match_strong_SFT')
    output.mkdir(parents=True);upload=output/'upload';upload.mkdir()
    data_path=upload/'dpo_strength_data.json';config_path=upload/'dpo_strength_config.json'
    data_path.write_bytes((sft/'upload/sft_strength_data.json').read_bytes())
    config.update(protocol=runtime.PROTOCOL,hyperparameters=runtime.HYPER,
        runtime_source_sha256=prior.file_sha(Path(runtime.__file__)),dataset_sha256=prior.file_sha(data_path),
        reference_scores=reference['scores'])
    prior.write_json(config_path,config);runtime.load_inputs(data_path,config_path)
    strong={(r['split'],r['pair_id'] if r['split']=='training' else r['example_id']):r for r in report['final_generations']}
    private=[]
    for line in (sft/'private_key/comparisons.private.jsonl').read_text().splitlines():
        row=json.loads(line);row.pop('weak_sft');g=strong[(row['split'],row['case_id'])]
        if (g['source_sha256']!=prior.sha(row['source']) or g['output_sha256']!=prior.sha(g['output'])):
            raise ValueError('strong_SFT_comparison_binding_changed')
        row['strong_sft']=g;private.append(row)
    if len(private)!=19:raise ValueError('nineteen_fixed_comparisons_required')
    private_dir=output/'private_key';private_dir.mkdir()
    (private_dir/'comparisons.private.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in private))
    plan={'protocol':runtime.PROTOCOL,'status':'prepared_not_executed','training_authorized':True,
        'training_pairs':17,'paragraph_groups':11,'epochs':2,'optimizer_updates':22,'learning_rate':1e-5,'beta':.1,
        'reference_policy':'exact unchanged original expanded epoch-2 adapter; frozen before updates',
        'matched_SFT_manifest_sha256':prior.file_sha(sft/'manifest.json'),
        'matched_SFT_review_sha256':prior.file_sha(review/'manifest.json'),
        'data_bytes_identical_to_strong_SFT':True,'new_SFT_training':False,'primary_epoch':2,
        'processor_sequences':42,'reference_sequences':34,'initial_policy_reference_pairs':17,
        'training_generation_cases':11,'development_generation_cases':8,'final_generations':19,
        'initial_control_generations':8,'total_new_generations':27,'cached_comparison_outputs':38,
        'maximum_runtime_seconds':1800,'reserved_test_records_read':0,'detector_calls':0,
        'dataset_sha256':config['dataset_sha256'],'config_sha256':prior.file_sha(config_path),
        'runtime_source_sha256':config['runtime_source_sha256'],'prior_runtime_sha256':config['prior_runtime_sha256'],
        'token_helper_sha256':config['token_helper_sha256']}
    prior.write_json(output/'plan.json',plan)
    prior.write_json(output/'authorization.json',{'user_message':'OK go ahead','user_authorized':True,
        'scope':'Single matched DPO-only continuation, beta 0.1, learning rate 1e-5, same 17 pairs, original adapter and 22 updates.',
        'optimizer_updates':22,'maximum_runtime_seconds':1800,'dpo_training_authorized':True,'full_training_authorized':False,
        'new_SFT_training_authorized':False,'dataset_sha256':config['dataset_sha256'],'config_sha256':prior.file_sha(config_path),
        'runtime_source_sha256':config['runtime_source_sha256'],'submission_method':'manual Kaggle UI; enable HF_TOKEN before execution'})
    prior.write_json(output/'receipt.json',{'builder_sha256':prior.file_sha(Path(__file__)),
        'source_manifests_sha256':{str(p.relative_to(root)/'manifest.json'):prior.file_sha(p/'manifest.json') for p in (sft,review,weak_review)},
        'reference_cache_source_sha256':prior.file_sha(weak_raw/'actual_error_pilot_v1/reference_scores.json'),
        'runtime_derivation_sha256':prior.file_sha(root/'scripts/build_dpo_strength_runtime.py')})
    prior.write_json(output/'manifest.json',{'files':{str(p.relative_to(output)):prior.file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return plan


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.root,a.output),indent=2))
