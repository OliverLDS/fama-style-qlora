"""Verify real CPU tokenizer recovery against frozen pairs, hashes and masks."""
import argparse
import json
from pathlib import Path

from fama_style.evaluation import preference_token_preflight as runtime
from .author_control_replication import verify_files
from .neutral_reconstruction import write_json,write_rows
from .neutral_reconstruction_results import file_sha


def validate(rows,config,report,state,audits,encoded,bos_id,eot_id):
    if (not report.get('complete') or not state.get('complete') or state['pairs_completed']!=17
            or any(report.get(k) is not False for k in ('training_called','model_weights_loaded','gpu_used','actual_model_log_probabilities_computed'))
            or any(state.get(k) is not False for k in ('training_called','model_weights_loaded','gpu_used'))
            or report['optimizer_steps']!=0 or report['packages']!=config['packages']
            or report['runtime_source_sha256']!=config['runtime_source_sha256']):
        raise ValueError('complete_tokenizer_only_execution_required')
    pairs={r['pair_id']:r for r in rows};packed={r['pair_id']:r for r in encoded};checks={(r['pair_id'],r['candidate']):r for r in audits}
    if (len(rows)!=17 or len(pairs)!=17 or len(encoded)!=17 or set(packed)!=set(pairs)
            or len(audits)!=34 or len(checks)!=34 or set(checks)!={(p,k) for p in pairs for k in ('chosen','rejected')}):
        raise ValueError('complete_unique_pair_and_response_coverage_required')
    for pid,row in pairs.items():
        item=packed[pid]
        if item['anchor_group_id']!=row['anchor_group_id'] or item['anchor_balanced_weight']!=row['anchor_balanced_weight']:
            raise ValueError('group_weight_changed')
        prefixes=[]
        for kind in ('chosen','rejected'):
            a=checks[(pid,kind)];seq=item[kind];ids=seq['input_ids'];labels=seq['labels'];n=a['prompt_tokens']
            if (a['input_ids_sha256']!=runtime.sha(json.dumps(ids)) or a['labels_sha256']!=runtime.sha(json.dumps(labels))
                    or len(ids)!=a['full_tokens'] or len(ids)>config['max_length'] or not 0<n<len(ids)
                    or labels!=[-100]*n+ids[n:] or ids[0]!=bos_id or ids.count(bos_id)!=1
                    or ids[n:].count(eot_id)!=1 or a['supervised_tokens']!=len(ids)-n
                    or a['decoded_response_matches'] is not True or a['template_end_of_turn_supervised'] is not True):
                raise ValueError('token_sequence_hash_mask_or_marker_changed')
            prefixes.append(ids[:n])
            if kind=='chosen':
                gold=row['archived_chosen']
                if any(a[k]!=gold[k] for k in ('input_ids_sha256','prompt_tokens','full_tokens','supervised_tokens')):
                    raise ValueError('archived_chosen_sequence_not_reproduced')
        if prefixes[0]!=prefixes[1]:raise ValueError('preference_prompt_prefixes_differ')
    if (report['pairs']!=17 or report['response_sequences']!=34 or report['archived_chosen_sequences_reproduced']!=17
            or report['max_full_tokens']!=max(a['full_tokens'] for a in audits)
            or report['min_supervised_tokens']!=min(a['supervised_tokens'] for a in audits)
            or report['response_padding_and_causal_scoring_contract_passed'] is not True
            or report['synthetic_dpo_identity_passed'] is not True):raise ValueError('summary_coverage_or_bounds_changed')
    return {'verified_pairs':17,'verified_response_sequences':34,'verified_archived_chosen_sequences':17,
        'max_full_tokens':report['max_full_tokens'],'min_supervised_tokens':report['min_supervised_tokens']}


def build(packet,run,previous_packet,failed_download,output):
    if output.exists():raise ValueError('refuse_overwrite_tokenizer_recovery_review')
    verify_files(packet);verify_files(previous_packet)
    rows,config=runtime.load_inputs(packet/'upload/actual_error_preflight_pairs.jsonl',packet/'upload/actual_error_preflight_config.json')
    if (file_sha(packet/'upload/actual_error_preflight_pairs.jsonl')!=file_sha(previous_packet/'upload/actual_error_preflight_pairs.jsonl')
            or config['previous_packet_manifest_sha256']!=file_sha(previous_packet/'manifest.json')
            or config['runtime_source_sha256']!=file_sha(Path(runtime.__file__))):raise ValueError('data_or_fixed_runtime_lineage_changed')
    assets=run/'tokenizer_assets';manifest=json.loads((assets/'asset_manifest.json').read_text())
    if (manifest['model_id']!=runtime.MODEL_ID or manifest['model_revision']!=runtime.REVISION
            or manifest['contains_model_weights'] is not False or set(manifest['files'])!=set(runtime.ASSETS)):
        raise ValueError('wrong_tokenizer_asset_set')
    for name,digest in manifest['files'].items():
        if (file_sha(assets/name)!=digest or file_sha(failed_download/'actual_error_preflight_v1/tokenizer_assets'/name)!=digest):
            raise ValueError('exported_tokenizer_asset_changed')
    metadata=json.loads((assets/'tokenizer_config.json').read_text());special={r['content']:int(i) for i,r in metadata['added_tokens_decoder'].items()}
    report=json.loads((run/'summary.json').read_text());state=json.loads((run/'state.json').read_text())
    if report['tokenizer_files_sha256']!=manifest['files']:raise ValueError('runtime_asset_hashes_changed')
    audits=json.loads((run/'token_audit.json').read_text());encoded=[json.loads(l) for l in (run/'encoded_pairs.private.jsonl').read_text().splitlines()]
    checked=validate(rows,config,report,state,audits,encoded,special['<bos>'],special['<end_of_turn>'])
    old_plan=json.loads((previous_packet/'plan.json').read_text());failure=json.loads((failed_download/'actual_error_preflight_v1/state.json').read_text())
    if (file_sha(failed_download/'preference_token_preflight.py')!=old_plan['runtime_source_sha256']
            or failure['complete'] is not False or failure['error_type']!='ValueError' or failure['pairs_completed']!=0):
        raise ValueError('original_failed_run_provenance_changed')
    summary={'status':'real_cpu_tokenizer_gate_passed_after_local_fix','kaggle_version_2_status':'failed_preserved',
        'successful_execution_location':'local CPU, offline using exported tokenizer assets','pair_bytes_unchanged':True,
        'tokenizer_assets_unchanged':True,'model_id':runtime.MODEL_ID,'model_revision':runtime.REVISION,
        **checked,'input_token_mask_gate_passed':True,'actual_model_log_probabilities_computed':False,
        'trainer_collator_equivalence_verified':False,'frozen_reference_scores_available':False,'gpu_memory_fit_measured':False,
        'training_authorized':False,'gpu_training_ready':False,'gpu_used':False,'model_weights_loaded':False,
        'agent_submitted_kaggle_jobs':0,'credentials_needed_for_local_recovery':False,
        'failure_cause':'Transformers 5.20.1 apply_chat_template defaults to BatchEncoding; old code assumed flat token IDs.',
        'fix':'Request return_dict=False explicitly and validate a flat list of nonnegative integer token IDs.',
        'synthetic_loss_tests_are_not_actual_model_scores':True}
    output.mkdir(parents=True);write_json(output/'summary.json',summary);write_rows(output/'verified_audits.private.jsonl',audits)
    write_json(output/'receipt.json',{'packet_manifest_sha256':file_sha(packet/'manifest.json'),
        'previous_packet_manifest_sha256':file_sha(previous_packet/'manifest.json'),
        'fixed_runtime_sha256':file_sha(Path(runtime.__file__)),
        'run_files_sha256':{str(p.relative_to(run)):file_sha(p) for p in sorted(run.rglob('*')) if p.is_file()},
        'failed_download_sha256':{str(p.relative_to(failed_download)):file_sha(p) for p in sorted(failed_download.rglob('*')) if p.is_file()}})
    write_json(output/'manifest.json',{'review_code_sha256':file_sha(Path(__file__)),
        'files':{str(p.relative_to(output)):file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','run','previous-packet','failed-download','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.packet,a.run,a.previous_packet,a.failed_download,a.output)))
