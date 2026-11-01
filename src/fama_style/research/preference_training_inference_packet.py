"""Prepare eleven fixed training-side prompts and reuse exact archived controls."""
import argparse
from collections import Counter
import json
from pathlib import Path

from fama_style.evaluation import preference_training_inference as runtime
from fama_style.training import preference_pair_pilot as pilot
from .author_control_replication import verify_files


def build(root,output):
    if output.exists():raise ValueError('preserve_existing_inference_packet')
    packet=root/'data/actual_error_pilot_v1'
    review=root/'research_runs/actual_error_pilot_v1/review_v1/results'
    download=root/'research_runs/actual_error_pilot_v1/kaggle_v1'
    controls_review=root/'research_runs/teacher_pair_reverse_pilot_v2/semantic_anchor_inference_v1/review_v1/results'
    controls_download=root/'research_runs/teacher_pair_reverse_pilot_v2/semantic_anchor_inference_v1/kaggle_v1'
    for folder in (packet,review,controls_review):verify_files(folder)
    receipt=json.loads((review/'receipt.json').read_text())
    if receipt['packet_manifest_sha256']!=runtime.file_sha(packet/'manifest.json'):raise ValueError('pilot_review_packet_mismatch')
    for name,digest in receipt['download_sha256'].items():
        if runtime.file_sha(download/name)!=digest:raise ValueError('reviewed_pilot_output_changed')
    control_receipt=json.loads((controls_review/'receipt.json').read_text())
    for name,digest in control_receipt['download_sha256'].items():
        if runtime.file_sha(controls_download/name)!=digest:raise ValueError('archived_control_output_changed')
    pairs,_,previous=pilot.load_inputs(packet/'upload/actual_error_pilot_data.json',packet/'upload/actual_error_pilot_config.json')
    checked=json.loads((review/'summary.json').read_text())
    if not checked['execution_complete'] or checked['optimizer_steps']!=44 or checked['primary_epoch']!=2:
        raise ValueError('completed_fixed_pilot_review_required')
    report=json.loads((download/'actual_error_pilot_v1/evaluation.json').read_text())
    old=json.loads((controls_download/'semantic_anchor_4b_v1/evaluation.json').read_text())
    if (not old['complete'] or old['training_called'] is not False or old['runtime']['packages']!=previous['packages']
            or any(old['config'][k]!=previous[k] for k in ('model_id','model_revision','decoding'))
            or old['config']['seed']!=1729 or old['config']['adapter_sha256']!=previous['adapter_sha256']
            or json.loads((controls_download/'semantic_anchor_4b_v1/generation_config.json').read_text())!=previous['expected_generation_config']):
        raise ValueError('control_environment_or_initial_adapter_differs')
    archived=next(s for s in old['snapshots'] if s['name']=='expanded_epoch_2')
    if not archived['complete']:raise ValueError('archived_generation_incomplete')
    controls={r['job_id']:r for r in archived['generations']}
    selected=runtime.select_rows(pairs);jobs=[];private=[]
    for row in selected:
        g=controls[row['generation_job_id']]
        if (any(g[k]!=row[k] for k in ('source','prompt','paper_id','example_id','input_kind'))
                or g['target_sha256']!=row['chosen_sha256'] or g['output']!=row['rejected']
                or runtime.sha(g['output'])!=row['rejected_sha256'] or g['empty'] or g['output_limited']):
            raise ValueError('reused_control_not_exact_training_rejection')
        seq=row['encoded']['chosen'];n,_=pilot.validate_sequence(seq);prefix=seq['input_ids'][:n]
        job={k:row[k] for k in ('pair_id','anchor_group_id','example_id','paper_id','input_kind','family','source','prompt','source_sha256','prompt_sha256')}
        job.update(prefix_ids=prefix,prefix_ids_sha256=runtime.sha(json.dumps(prefix)));jobs.append(job)
        private.append({k:row[k] for k in ('pair_id','example_id','paper_id','anchor_group_id','source','prompt','chosen','chosen_sha256','rejected_sha256','input_kind','family',
            'inherited_needs_review','inherited_needs_external_context','inherited_review_notes')}|
            {'split':'training','control':g,'control_origin':'archived expanded_epoch_2 / exact included actual rejection'})
    adapter_sha={arm:report['arms'][arm]['final_adapter_files_sha256'] for arm in runtime.ARMS}
    for arm in runtime.ARMS:
        runtime.verify_adapter(download/'actual_error_pilot_v1'/arm/'epoch_2/adapter',adapter_sha[arm])
    output.mkdir(parents=True);upload=output/'upload';upload.mkdir()
    jobs_path=upload/'preference_training_jobs.jsonl';config_path=upload/'preference_training_config.json'
    jobs_path.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in jobs))
    config={'protocol':runtime.PROTOCOL,'model_id':runtime.MODEL_ID,'model_revision':runtime.REVISION,
        'decoding':runtime.DECODING,'arms':list(runtime.ARMS),'seed':1729,'max_length':1024,'maximum_runtime_seconds':1800,
        'training_authorized':False,'optimizer_steps':0,'jobs_sha256':runtime.file_sha(jobs_path),
        'runtime_source_sha256':runtime.file_sha(Path(runtime.__file__)),'pair_ids':[r['pair_id'] for r in jobs],
        'packages':previous['packages'],'adapter_sha256':adapter_sha,'expected_generation_config':previous['expected_generation_config']}
    runtime.write_json(config_path,config);runtime.load_inputs(jobs_path,config_path)
    private_dir=output/'private_key';private_dir.mkdir()
    (private_dir/'reference_key.private.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in private))
    plan={'protocol':runtime.PROTOCOL,'status':'prepared_not_executed','training_authorized':False,'optimizer_steps':0,
        'paragraphs':11,'papers':10,'input_kinds':dict(Counter(r['input_kind'] for r in selected)),
        'families':dict(Counter(r['family'] for r in selected)),'archived_control_outputs':11,'new_generations':22,
        'comparison_outputs':33,'checkpoint_epoch':2,'selection_rule':'One included row per anchor group; original preferred, otherwise variant; sorted group ID then pair ID.',
        'maximum_runtime_seconds':1800,'max_prompt_tokens':max(len(r['prefix_ids']) for r in jobs),
        'reference_wording_in_generator_payload':False,'development_records_read_for_prior_packet_validation':8,
        'development_records_used_for_selection':0,'reserved_test_records_read':0,
        'runtime_source_sha256':config['runtime_source_sha256'],'jobs_sha256':runtime.file_sha(jobs_path),
        'config_sha256':runtime.file_sha(config_path),
        'limits':['Seen training examples selected for actual errors, not a representative or held-out sample.',
                  'Training-side output changes do not establish generalization, detector improvement or Fama-specific style.',
                  'References/control outputs remain private and outside the generator upload payload.']}
    runtime.write_json(output/'plan.json',plan)
    runtime.write_json(output/'receipt.json',{'builder_sha256':runtime.file_sha(Path(__file__)),
        'source_manifests_sha256':{str(p.relative_to(root)/'manifest.json'):runtime.file_sha(p/'manifest.json') for p in (packet,review,controls_review)},
        'selected_pair_ids':[r['pair_id'] for r in selected],
        'control_evaluation_sha256':runtime.file_sha(controls_download/'semantic_anchor_4b_v1/evaluation.json'),
        'completed_pilot_evaluation_sha256':runtime.file_sha(download/'actual_error_pilot_v1/evaluation.json')})
    runtime.write_json(output/'manifest.json',{'files':{str(p.relative_to(output)):runtime.file_sha(p)
        for p in sorted(output.rglob('*')) if p.is_file()}})
    return plan


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(build(a.root,a.output),indent=2))
