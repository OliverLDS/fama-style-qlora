"""Offline matched-DPO audit; reuse common SFT checks without altering old runs."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from fama_style.training import dpo_strength_pilot as runtime
from . import sft_strength_review as shared
from .author_control_replication import verify_files

prior=runtime.prior


def validate_execution(report,state,config,pairs,development,cases,audits,gradients,updates,reference):
    # The common execution contract covers 22 updates, target scores and 19 outputs.
    # Validate all DPO-only audits first; then project the shared chosen/dev checks.
    sequences=[(r['pair_id']+'__'+k,r['encoded'][k]) for r in pairs for k in ('chosen','rejected')]
    sequences +=[(r['example_id'],r['encoded']) for r in development]
    if len(audits)!=42 or [a['id'] for a in audits]!=[i for i,_ in sequences]:raise ValueError('forty_two_response_audits_required')
    for a,(_,seq) in zip(audits,sequences):
        if (a['input_ids_sha256']!=prior.sha(json.dumps(seq['input_ids'])) or a['labels_sha256']!=prior.sha(json.dumps(seq['labels']))
                or a['prompt_tokens']!=prior.validate_sequence(seq)[0] or a['full_tokens']!=len(seq['input_ids'])):
            raise ValueError('DPO_response_ids_or_masks_changed')
    sanity=report['likelihood_sanity']
    if len(sanity)!=3 or any(not s['passed'] or not math.isclose(s['manual_mean_nll'],s['builtin_mean_nll'],rel_tol=1e-5,abs_tol=1e-5) for s in sanity):
        raise ValueError('chosen_rejected_and_development_sanity_required')
    projected=copy.deepcopy(report);projected['likelihood_sanity']=[sanity[0],sanity[2]]
    chosen=[{**audits[i],'id':r['pair_id']} for i,r in zip(range(0,34,2),pairs)]+audits[34:]
    shared.validate_execution(projected,state,config,pairs,development,cases,chosen,gradients,updates)
    if (reference['cache_sha256']!=prior.sha(json.dumps(reference['scores'],sort_keys=True))
            or reference['cache_sha256']!=report['reference_cache_sha256']
            or reference['initial_adapter_tensor_sha256']!=config['initial_adapter_tensor_sha256']
            or reference['source_adapter_sha256']!=config['adapter_sha256'] or reference['dataset_sha256']!=config['dataset_sha256']
            or reference['model_id']!=config['model_id'] or reference['model_revision']!=config['model_revision']
            or reference['reference_policy']!='unchanged_original_expanded_epoch_2'):
        raise ValueError('fixed_reference_binding_failed')
    ids=[r['pair_id'] for r in pairs]
    if set(reference['scores'])!=set(ids) or any(u['reference_cache_sha256']!=reference['cache_sha256'] for u in updates):
        raise ValueError('immutable_reference_coverage_failed')
    for row in pairs:
        for kind in ('chosen','rejected'):
            measured=reference['scores'][row['pair_id']][kind];gold=config['reference_scores'][row['pair_id']][kind]
            n=prior.validate_sequence(row['encoded'][kind])[1]
            if (measured['supervised_tokens']!=n or not math.isfinite(measured['total_log_probability'])
                    or measured['total_log_probability']>0 or abs(measured['total_log_probability']-gold['total_log_probability'])>1e-3
                    or not math.isclose(measured['mean_nll'],-measured['total_log_probability']/n)):
                raise ValueError('original_reference_score_not_reproduced')
    identity=report['initial_policy_reference_identity']
    if [r['pair_id'] for r in identity]!=ids or any(not math.isfinite(r['max_logp_difference']) or not 0<=r['max_logp_difference']<=1e-3
            or not math.isfinite(r['dpo_loss']) or abs(r['dpo_loss']-math.log(2))>2e-4 for r in identity):
        raise ValueError('initial_policy_reference_identity_failed')
    final=report['final_preference_scores'];chosen_by={r['pair_id']:r for r in report['final_training_scores']}
    if [r['pair_id'] for r in final]!=ids:raise ValueError('seventeen_final_preference_pairs_required')
    for item,row in zip(final,pairs):
        for kind in ('chosen','rejected'):
            s=item[kind];n=prior.validate_sequence(row['encoded'][kind])[1]
            if (s['supervised_tokens']!=n or not math.isfinite(s['total_log_probability']) or s['total_log_probability']>0
                    or not math.isclose(s['mean_nll'],-s['total_log_probability']/n)):raise ValueError('final_preference_score_invalid')
        if any(item['chosen'][k]!=chosen_by[row['pair_id']][k] for k in item['chosen']):raise ValueError('final_chosen_scores_disagree')


def masked_comparison(report,references):
    generated={(g['split'],g['pair_id'] if g['split']=='training' else g['example_id']):g for g in report['final_generations']}
    bundle=[];key=[]
    for ref in references:
        split,cid=ref['split'],ref['case_id'];order=['control','strong_sft','strong_dpo']
        random.Random(int(hashlib.sha256(('dpo-strength-review-v1:'+split+':'+cid).encode()).hexdigest(),16)).shuffle(order)
        row={k:ref[k] for k in ('split','case_id','example_id','paper_id','source','target')};row['outputs']={}
        for label,arm in zip('ABC',order):
            g=generated[(split,cid)] if arm=='strong_dpo' else ref[arm]
            row['outputs'][label]=g['output'];key.append({'split':split,'case_id':cid,'label':label,'arm':arm,'output_sha256':prior.sha(g['output'])})
        bundle.append(row)
    return bundle,key


def summarize(bundle,key,judgments):
    # Reuse the same tested rubric and paired counting with temporary condition aliases.
    aliased=[{**r,'arm':{'control':'control','strong_sft':'weak_sft','strong_dpo':'strong_sft'}[r['arm']]} for r in key]
    summary,result=shared.summarize(bundle,aliased,judgments)
    names={'weak_sft':'strong_sft','strong_sft':'strong_dpo','generic_expression_vs_weak_sft':'generic_expression_vs_strong_sft'}
    def restore(value):
        if isinstance(value,dict):return {names.get(k,k):restore(v) for k,v in value.items()}
        if isinstance(value,list):return [restore(v) for v in value]
        return names.get(value,value) if isinstance(value,str) else value
    return restore(summary),restore(result)


def build(packet,download,review,output):
    if output.exists():raise ValueError('preserve_existing_DPO_review')
    verify_files(packet);pairs,dev,cases,config=runtime.load_inputs(packet/'upload/dpo_strength_data.json',packet/'upload/dpo_strength_config.json')
    folder=download/'dpo_strength_v1'
    def load(name):return json.loads((folder/name).read_text())
    report=load('evaluation.json');state=load('state.json');reference=load('reference_scores.json')
    validate_execution(report,state,config,pairs,dev,cases,load('processor_audit.json'),load('gradient_checks.json'),load('update_log.json'),reference)
    for name,k in (('dpo_strength_pilot.py','runtime_source_sha256'),('preference_pair_pilot.py','prior_runtime_sha256'),('preference_token_preflight.py','token_helper_sha256')):
        if prior.file_sha(download/name)!=config[k]:raise ValueError('executed_runtime_changed')
    if load('generation_config.json')!=config['expected_generation_config']:raise ValueError('decoding_changed')
    if [(c['epoch'],c['step']) for c in report['checkpoints']]!=[(1,11),(2,22)]:raise ValueError('fixed_checkpoints_required')
    for checkpoint in report['checkpoints']:
        adapter=folder/('epoch_'+str(checkpoint['epoch']))/'adapter'
        if any(prior.file_sha(adapter/n)!=digest for n,digest in checkpoint['files_sha256'].items()):raise ValueError('saved_checkpoint_changed')
        fingerprint=shared.tensor_fingerprint(adapter/'adapter_model.safetensors')
        if checkpoint['epoch']==2 and fingerprint!=report['final_adapter_tensor_sha256']:raise ValueError('final_tensor_fingerprint_changed')
    refs=shared.rows(packet/'private_key/comparisons.private.jsonl');bundle,key=masked_comparison(report,refs)
    if bundle!=shared.rows(review/'masked_bundle.private.jsonl') or key!=json.loads((review/'masked_key.private.json').read_text()):raise ValueError('masking_not_reproducible')
    judgments=json.loads((review/'masked_judgments.private.json').read_text());summary,result=summarize(bundle,key,judgments)
    for split in ('training','development'):
        old={r['case_id']:r for r in result if r['split']==split and r['arm']=='strong_sft'}
        current={r['case_id']:r for r in result if r['split']==split and r['arm']=='strong_dpo'}
        summary[split]['strong_dpo']['exact_strong_SFT_outputs']=sum(current[c]['output']==old[c]['output'] for c in old)
    nll={}
    for stage in ('initial','final'):
        for split in ('training','development'):
            scored=report[stage+'_'+split+'_scores'];nll[stage+'_'+split]=-sum(r['total_log_probability'] for r in scored)/sum(r['supervised_tokens'] for r in scored)
    margins=[];losses=[]
    for r in report['final_preference_scores']:
        pc,pr=(r[k]['total_log_probability'] for k in ('chosen','rejected'))
        rc,rr=(reference['scores'][r['pair_id']][k]['total_log_probability'] for k in ('chosen','rejected'))
        margins.append((pc-pr)-(rc-rr));losses.append(prior.tokens.dpo_loss(pc,pr,rc,rr,.1))
    flags={k:sum(g[k] for g in report['final_generations']) for k in ('empty','output_limited','copied_source')}
    dpo=summary['development']['strong_dpo'];pref=dpo['generic_expression_vs_control']
    summary.update(protocol='dpo_strength_review_v1',slug='oliverlds/dpo-strength-4b',user_reported_version=1,
        execution_complete=True,optimizer_updates=22,learning_rate=1e-5,beta=.1,reference_sequences_verified=34,
        initial_identity_pairs=17,processor_sequences_verified=42,control_outputs_reproduced=8,final_generations=19,
        total_new_generations=27,adapter_reload_weights_match=True,target_nll=nll,final_generation_flags=flags,
        mean_training_relative_margin_change_nats=statistics.mean(margins),positive_training_relative_margins=sum(m>0 for m in margins),
        mean_final_training_dpo_loss=statistics.mean(losses),elapsed_seconds=state['elapsed_seconds'],peak_gpu_allocated_bytes=state['peak_gpu_allocated_bytes'],
        development_quality_gate_passed=dpo['material_claim_issues']<=3 and dpo['eligible_within_local_budget']>=4
            and pref['better']>pref['worse'] and dpo['exact_source_copies']==0 and not any(flags.values()),
        reserved_test_records_read=0,detector_calls=0,new_gpu_jobs_submitted=0,additional_training_authorized=False,
        assessment_provisional=True,fama_style_established=False,limits=judgments['provenance']['limits'])
    output.mkdir(parents=True);prior.write_json(output/'summary.json',summary)
    (output/'per_output.private.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result))
    prior.write_json(output/'receipt.json',{'packet_manifest_sha256':prior.file_sha(packet/'manifest.json'),'review_code_sha256':prior.file_sha(Path(__file__)),
        'shared_review_code_sha256':prior.file_sha(Path(shared.__file__)),
        'download_sha256':{str(p.relative_to(download)):prior.file_sha(p) for p in sorted(download.rglob('*')) if p.is_file()},
        'annotation_sha256':{p.name:prior.file_sha(p) for p in sorted(review.glob('*.json*')) if p.is_file()}})
    prior.write_json(output/'manifest.json',{'files':{p.name:prior.file_sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','download','review','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.packet,a.download,a.review,a.output),indent=2))
