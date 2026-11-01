"""Freeze verified preference tokens and the original eight development cases."""
import argparse
import json
from pathlib import Path

from fama_style.training import preference_pair_pilot as pilot
from .author_control_replication import verify_files


def read_rows(path):return [json.loads(line) for line in path.read_text().splitlines()]


def run_training(assets,development,gold_audits):
    """CPU tokenizer packaging only. No model weights or optimization."""
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(assets,local_files_only=True,use_fast=True,trust_remote_code=False)
    tokenizer.chat_template=json.loads((assets/'chat_template.json').read_text())['chat_template']
    eot=tokenizer.convert_tokens_to_ids('<end_of_turn>')
    for row in development:
        messages=pilot.messages(row)
        prefix=pilot.tokens.encode_chat(tokenizer,messages,True)
        full=pilot.tokens.encode_chat(tokenizer,messages+[{'role':'assistant','content':[{'type':'text','text':row['target']}]}],False)
        labels=pilot.tokens.mask(prefix,full,tokenizer.decode(full[len(prefix):],skip_special_tokens=True),
            row['target'],1024,eot,tokenizer.bos_token_id)
        digest=pilot.sha(json.dumps(prefix));gold=gold_audits[row['example_id']+'__adapter_ordinary']
        if digest!=gold['input_ids_sha256'] or len(prefix)!=gold['input_tokens']:
            raise ValueError('archived_development_prompt_not_reproduced')
        row.update(prefix_ids=prefix,prefix_ids_sha256=digest,encoded={'input_ids':full,'labels':labels})


def build(root,output):
    if output.exists():raise ValueError('refuse_overwrite_pilot_packet')
    candidates=root/'data/actual_error_preferences_v1'
    preflight=root/'data/actual_error_preflight_v2'
    run=root/'research_runs/preference_token_preflight_v1/local_recovery_v2'
    review=root/'research_runs/preference_token_preflight_v1/recovery_review_v2'
    milestone=root/'data/milestone_checkpoint_4b_v1'
    export=root/'research_runs/milestone_checkpoint_4b_v1/export_v1'
    raw=root/'research_runs/milestone_checkpoint_4b_v1/kaggle_v1'
    for directory in (candidates,preflight,review,milestone,export):verify_files(directory)
    summary=json.loads((review/'summary.json').read_text());receipt=json.loads((review/'receipt.json').read_text())
    if (summary['input_token_mask_gate_passed'] is not True or summary['verified_pairs']!=17
            or receipt['packet_manifest_sha256']!=pilot.file_sha(preflight/'manifest.json')
            or receipt['fixed_runtime_sha256']!=pilot.file_sha(Path(pilot.tokens.__file__))):
        raise ValueError('verified_tokenizer_recovery_required')
    for name,digest in receipt['run_files_sha256'].items():
        if pilot.file_sha(run/name)!=digest:raise ValueError('recovery_artifact_changed')
    export_receipt=json.loads((export/'receipt.json').read_text())
    if export_receipt['package_manifest_sha256']!=pilot.file_sha(milestone/'manifest.json'):
        raise ValueError('milestone_lineage_changed')
    for name,digest in export_receipt['download_sha256'].items():
        if pilot.file_sha(raw/name)!=digest:raise ValueError('milestone_download_changed')
    encoded={r['pair_id']:r for r in read_rows(run/'encoded_pairs.private.jsonl')}
    audits={(r['pair_id'],r['candidate']):r for r in json.loads((run/'token_audit.json').read_text())}
    pairs=read_rows(candidates/'preferences.private.jsonl')
    for row in pairs:
        row['encoded']={k:encoded[row['pair_id']][k] for k in ('chosen','rejected')}
        row['token_audits']={k:audits[(row['pair_id'],k)] for k in ('chosen','rejected')}
    baseline=json.loads((milestone/'upload/milestone_checkpoint_config.json').read_text())
    refs={r['example_id']:r for r in read_rows(milestone/'private_key/reference_key.private.jsonl')}
    generations={r['example_id']:r for r in read_rows(export/'combined_outputs.private.jsonl') if r['condition']=='adapter_ordinary'}
    development=[]
    for example_id in baseline['example_ids']:
        row=dict(refs[example_id]);gold=generations[example_id]
        row.update(split='development',prompt=pilot.prompt(row['source']))
        row['prompt_sha256']=pilot.sha(row['prompt'])
        if any(row[k]!=gold[k] for k in ('source','source_sha256','prompt_sha256')):
            raise ValueError('development_prompt_changed')
        row['expected_generation']={k:gold[k] for k in ('output','output_tokens','empty','output_limited','copied_source')}
        development.append(row)
    gold_audits={r['job_id']:r for r in json.loads((raw/'milestone_checkpoint_4b_v1/token_audit.json').read_text())}
    run_training(run/'tokenizer_assets',development,gold_audits)
    output.mkdir(parents=True);upload=output/'upload';upload.mkdir()
    data_path=upload/'actual_error_pilot_data.json';config_path=upload/'actual_error_pilot_config.json'
    pilot.write_json(data_path,{'training_pairs':pairs,'development':development})
    config={'protocol':pilot.PROTOCOL,'model_id':pilot.MODEL_ID,'model_revision':pilot.REVISION,
        'hyperparameters':pilot.HYPER,'decoding':pilot.DECODING,'training_authorized':False,'max_runtime_seconds':3600,
        'dataset_sha256':pilot.file_sha(data_path),'token_helper_sha256':pilot.file_sha(Path(pilot.tokens.__file__)),
        'runtime_source_sha256':pilot.file_sha(Path(pilot.__file__)),
        'packages':baseline['expected_packages'],'adapter_sha256':baseline['adapter_sha256'],
        'expected_generation_config':baseline['expected_generation_config'],
        'development_ids':baseline['example_ids'],'group_schedule':pilot.schedule(pairs)}
    pilot.write_json(config_path,config)
    pilot.load_inputs(data_path,config_path)
    plan={'protocol':pilot.PROTOCOL,'status':'prepared_not_executed','training_authorized':False,
        'training_pairs':17,'paragraph_groups':11,'training_papers':len({r['paper_id'] for r in pairs}),
        'development_cases':8,'reserved_test_records_read':0,'updates_per_arm':22,'total_updates':44,
        'snapshots':5,'planned_generations':40,'frozen_reference_sequences':34,'processor_checks':42,
        'primary_checkpoint':'epoch_2','maximum_runtime_seconds':3600,'measured_gpu_fit':False,
        'development_usage':'evaluation packaging only; preference selection already frozen',
        'runtime_source_sha256':config['runtime_source_sha256'],'token_helper_sha256':config['token_helper_sha256'],
        'data_sha256':pilot.file_sha(data_path),'config_sha256':pilot.file_sha(config_path)}
    pilot.write_json(output/'plan.json',plan)
    pilot.write_json(output/'receipt.json',{'source_manifests_sha256':{
        str(p.relative_to(root)/'manifest.json'):pilot.file_sha(p/'manifest.json')
        for p in (candidates,preflight,review,milestone,export)},
        'tokenizer_assets_sha256':json.loads((run/'tokenizer_assets/asset_manifest.json').read_text())['files'],
        'builder_sha256':pilot.file_sha(Path(__file__))})
    pilot.write_json(output/'manifest.json',{'files':{str(p.relative_to(output)):pilot.file_sha(p)
        for p in sorted(output.rglob('*')) if p.is_file()}})
    return plan


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(build(a.root,a.output),indent=2))
