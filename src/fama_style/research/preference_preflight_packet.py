"""Freeze tokenizer-only requests with archived chosen-ID checks; no model calls."""
import argparse
import json
from pathlib import Path

from fama_style.evaluation import preference_token_preflight as runtime
from .author_control_replication import verify_files
from .neutral_reconstruction import write_json,write_rows
from .neutral_reconstruction_results import file_sha


def build(candidates,download,prior_review,output,previous_packet=None):
    if output.exists():raise ValueError('refuse_overwrite_cpu_preflight_packet')
    verify_files(candidates);verify_files(prior_review)
    candidate_receipt=json.loads((candidates/'receipt.json').read_text())
    if candidate_receipt['verified_inference_review_manifest_sha256']!=file_sha(prior_review/'manifest.json'):
        raise ValueError('wrong_archived_review')
    receipt=json.loads((prior_review/'receipt.json').read_text());path=download/'semantic_anchor_4b_v1/label_audit.json'
    if file_sha(path)!=receipt['download_sha256']['semantic_anchor_4b_v1/label_audit.json']:
        raise ValueError('archived_token_audits_changed')
    audits={r['job_id']:r for r in json.loads(path.read_text()) if r['candidate']=='chosen'}
    pairs=[json.loads(l) for l in (candidates/'preferences.private.jsonl').read_text().splitlines()]
    controls=[json.loads(l) for l in (candidates/'sft_control.private.jsonl').read_text().splitlines()]
    if len(pairs)!=17 or len(controls)!=17:raise ValueError('seventeen_matched_candidate_controls_required')
    payload=[]
    for p,c in zip(pairs,controls):
        if (p['pair_id']!=c['pair_id'] or p['prompt']!=c['prompt'] or p['chosen']!=c['completion']
                or p['anchor_balanced_weight']!=c['anchor_balanced_weight'] or p['synthetic_rejection'] is not False):
            raise ValueError('preference_sft_control_binding_failed')
        row={k:p[k] for k in ('pair_id','example_id','anchor_group_id','paper_id','split','prompt','prompt_sha256',
            'chosen','chosen_sha256','rejected','rejected_sha256','anchor_balanced_weight')}
        row['archived_chosen']=audits[p['generation_job_id']];payload.append(row)
    output.mkdir(parents=True);upload=output/'upload';upload.mkdir()
    data_path=upload/'actual_error_preflight_pairs.jsonl';write_rows(data_path,payload)
    if previous_packet is not None:
        verify_files(previous_packet)
        if file_sha(data_path)!=file_sha(previous_packet/'upload/actual_error_preflight_pairs.jsonl'):
            raise ValueError('bugfix_must_preserve_all_seventeen_pair_bytes')
    config={'protocol':runtime.PROTOCOL,'model_id':runtime.MODEL_ID,'model_revision':runtime.REVISION,'max_length':1024,
        'packages':runtime.PACKAGES,'data_sha256':file_sha(data_path),'training_authorized':False,
        'candidate_manifest_sha256':file_sha(candidates/'manifest.json'),'archived_token_audit_sha256':file_sha(path),
        'runtime_source_sha256':file_sha(Path(runtime.__file__))}
    if previous_packet is not None:
        config.update(previous_packet_manifest_sha256=file_sha(previous_packet/'manifest.json'),
            fix='Request flat token IDs explicitly; preserve tokenizer, masks, hashes, sequences and pair data.')
    config_path=upload/'actual_error_preflight_config.json';write_json(config_path,config)
    runtime.load_inputs(data_path,config_path)
    plan={'protocol':runtime.PROTOCOL,'status':'prepared_real_tokenizer_not_executed','pairs':17,'response_sequences':34,
        'expected_archived_chosen_matches':17,'requires_model_weights':False,'requires_gpu':False,'requires_adapter':False,
        'download_allowlist':list(runtime.ASSETS),'data_sha256':file_sha(data_path),'config_sha256':file_sha(config_path),
        'runtime_source_sha256':file_sha(Path(runtime.__file__)),'gpu_jobs_submitted':0,'training_authorized':False,
        'tokenizer_access':'Existing Kaggle Secret for initial assets, or a verified exported local tokenizer bundle.',
        'actual_model_response_scores_computed':False,'synthetic_loss_contract_is_not_model_validation':True}
    write_json(output/'plan.json',plan)
    write_json(output/'manifest.json',{'code_sha256':file_sha(Path(__file__)),
        'files':{str(p.relative_to(output)):file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return plan


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('candidates','download','prior-review','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--previous-packet',type=Path)
    a=p.parse_args();print(json.dumps(build(a.candidates,a.download,a.prior_review,a.output,a.previous_packet)))
