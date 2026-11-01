"""Offline audit of bounded SFT/DPO execution and frozen primary generations."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct

from fama_style.training import preference_pair_pilot as pilot
from .author_control_replication import verify_files
from .neutral_reconstruction_results import metrics


def tensor_fingerprint(path):
    with path.open('rb') as stream:
        size=struct.unpack('<Q',stream.read(8))[0]
        if size>path.stat().st_size-8:raise ValueError('truncated_adapter')
        header=json.loads(stream.read(size));base=8+size
        digest=hashlib.sha256();count=0;parameters=0
        for name in sorted(k for k in header if k!='__metadata__'):
            item=header[name];start,end=item['data_offsets'];shape=item['shape']
            if (item['dtype']!='F32' or 'language_model' not in name or '.lora_' not in name
                    or len(shape)!=2 or 8 not in shape or end-start!=4*math.prod(shape)):
                raise ValueError('unexpected_adapter_tensor')
            stream.seek(base+start);data=stream.read(end-start)
            if len(data)!=end-start:raise ValueError('truncated_adapter')
            digest.update(name.encode());digest.update(str(tuple(shape)).encode());digest.update(data)
            count+=1;parameters+=math.prod(shape)
    if count!=136 or parameters!=2228224:raise ValueError('adapter_tensor_coverage')
    return digest.hexdigest()


def validate_execution(report,state,config,pairs,development,gradients,updates,processor,reference):
    if (report.get('complete') is not True or state.get('complete') is not True
            or state.get('phase')!='complete' or state.get('optimizer_steps')!=44
            or state.get('arm_steps')!={'sft':22,'dpo':22} or not state.get('training_called')
            or not report.get('training_called') or report['config']!=config
            or report['runtime']['packages']!=config['packages']
            or report['runtime']['runtime_source_sha256']!=config['runtime_source_sha256']
            or report['runtime']['token_helper_sha256']!=config['token_helper_sha256']
            or report['runtime']['model_id']!=config['model_id'] or report['runtime']['model_revision']!=config['model_revision']
            or report['runtime']['gpu_count']!=1 or 'T4' not in report['runtime']['gpu']
            or not 0<state['elapsed_seconds']<3600 or not 0<state['peak_gpu_allocated_bytes']<16*1024**3):
        raise ValueError('bounded_execution_contract_failed')
    if (report['common_control_exact_matches']!=8 or len(report['likelihood_sanity'])!=2
            or any(not s['passed'] or not math.isclose(s['manual_mean_nll'],s['builtin_mean_nll'],rel_tol=1e-5,abs_tol=1e-5)
                   for s in report['likelihood_sanity'])):raise ValueError('control_or_actual_likelihood_gate_failed')
    expected=[('control',0,0),('sft',1,11),('sft',2,22),('dpo',1,11),('dpo',2,22)]
    if [(s['arm'],s['epoch'],s['optimizer_step']) for s in report['snapshots']]!=expected:
        raise ValueError('fixed_snapshot_boundaries_failed')
    ids=config['development_ids'];by_id={r['example_id']:r for r in development}
    for snap in report['snapshots']:
        if (not snap['complete'] or [r['example_id'] for r in snap['losses']]!=ids
                or [r['example_id'] for r in snap['generations']]!=ids):raise ValueError('eight_case_snapshot_incomplete')
        for loss in snap['losses']:
            count=pilot.validate_sequence(by_id[loss['example_id']]['encoded'])[1]
            if (loss['supervised_tokens']!=count or not math.isfinite(loss['total_log_probability'])
                    or loss['total_log_probability']>0 or not math.isclose(loss['mean_nll'],-loss['total_log_probability']/count)):
                raise ValueError('invalid_development_likelihood')
        weighted=-sum(r['total_log_probability'] for r in snap['losses'])/sum(r['supervised_tokens'] for r in snap['losses'])
        if not math.isclose(weighted,snap['token_weighted_development_nll'],abs_tol=1e-12):raise ValueError('development_loss_aggregation')
        for g in snap['generations']:
            row=by_id[g['example_id']]
            if any(row[k]!=g[k] for k in ('source_sha256','prompt_sha256')):raise ValueError('generation_binding_changed')
            if g['empty']!=(not bool(g['output'].strip())) or g['copied_source']!=(pilot.tokens.norm(g['output'])==pilot.tokens.norm(row['source'])):
                raise ValueError('generation_flags_changed')
            if snap['arm']=='control' and any(g[k]!=row['expected_generation'][k] for k in row['expected_generation']):
                raise ValueError('archived_control_not_reproduced')
    expected_updates=[(arm,e['epoch'],11*(e['epoch']-1)+i+1,g['group_id'],len(g['pair_ids']))
        for arm in pilot.ARMS for e in config['group_schedule'] for i,g in enumerate(e['groups'])]
    if [(u['arm'],u['epoch'],u['step'],u['group_id'],u['rows']) for u in updates]!=expected_updates:
        raise ValueError('independent_grouped_update_coverage')
    if [(g['arm'],g['step']) for g in gradients]!=[(a,s) for a,_,s,_,_ in expected_updates]:raise ValueError('gradient_coverage')
    if any(not g['finite'] or not math.isfinite(g['norm']) or g['norm']<=0 for g in gradients):raise ValueError('invalid_gradient')
    for u in updates:
        if (not math.isfinite(u['loss']) or not math.isclose(u['learning_rate'],1e-6*(1-(u['step']-1)/22))
                or u['reference_cache_sha256']!=reference['cache_sha256']):raise ValueError('update_schedule_or_reference_changed')
    if (reference['dataset_sha256']!=config['dataset_sha256'] or reference['source_adapter_sha256']!=config['adapter_sha256']
            or reference['model_id']!=config['model_id'] or reference['model_revision']!=config['model_revision']
            or reference['token_helper_sha256']!=config['token_helper_sha256']
            or reference['initial_adapter_tensor_sha256']!=report['initial_adapter_tensor_sha256']
            or pilot.sha(json.dumps(reference['scores'],sort_keys=True))!=reference['cache_sha256']
            or report['frozen_reference_cache_sha256']!=reference['cache_sha256']):raise ValueError('reference_cache_binding')
    pids=[r['pair_id'] for r in pairs]
    if set(reference['scores'])!=set(pids) or {r['pair_id']:{k:r[k] for k in ('chosen','rejected')} for r in report['reference_scores']}!=reference['scores']:
        raise ValueError('reference_pair_coverage')
    expected_processor={(r['pair_id'],k):r['token_audits'][k] for r in pairs for k in ('chosen','rejected')}
    for row in development:
        seq=row['encoded'];expected_processor[(row['example_id'],'development_target')]={
            'input_ids_sha256':pilot.sha(json.dumps(seq['input_ids'])),'labels_sha256':pilot.sha(json.dumps(seq['labels']))}
    if len(processor)!=42 or len({(r.get('pair_id',r.get('example_id')),r['candidate']) for r in processor})!=42:
        raise ValueError('processor_coverage')
    for r in processor:
        gold=expected_processor[(r.get('pair_id',r.get('example_id')),r['candidate'])]
        if any(r[k]!=gold[k] for k in ('input_ids_sha256','labels_sha256')):raise ValueError('processor_token_or_mask_changed')
    for arm in pilot.ARMS:
        a=report['arms'][arm]
        if (not a['optimizer_initially_empty'] or not a['reload_weights_match'] or not a['weights_changed']
                or a['completed_updates']!=22 or a['initial_adapter_tensor_sha256']!=report['initial_adapter_tensor_sha256']
                or a['final_adapter_tensor_sha256']==a['initial_adapter_tensor_sha256']
                or [r['pair_id'] for r in a['reference_identity']]!=pids
                or [r['pair_id'] for r in a['final_training_scores']]!=pids):raise ValueError('independent_arm_or_reload_failed')
        if any(not math.isfinite(r['max_logp_difference']) or r['max_logp_difference']>1e-3
               or abs(r['dpo_loss']-math.log(2))>2e-4 for r in a['reference_identity']):raise ValueError('actual_reference_identity_failed')


def summarize(report,development,key,judgments):
    mapping={(r['example_id'],r['label']):r for r in key};judged={r['example_id']:r for r in judgments['cases']}
    ids=[r['example_id'] for r in development]
    if len(mapping)!=24 or len(key)!=24 or len(judged)!=8 or set(judged)!=set(ids):raise ValueError('primary_review_coverage')
    primary={s['arm']:s for s in report['snapshots'] if s['epoch']==2 or s['arm']=='control'}
    outputs={(arm,g['example_id']):g for arm,s in primary.items() for g in s['generations']}
    if {(r['arm'],r['example_id']) for r in key}!=set(outputs):raise ValueError('masked_condition_coverage')
    results=[];ranks={}
    for row in development:
        rid=row['example_id'];case=judged[rid];groups=case['ranking_best_first']
        if any(not g for g in groups) or sorted(l for g in groups for l in g)!=list('ABC'):raise ValueError('invalid_ranks')
        local={l:i for i,g in enumerate(groups) for l in g}
        for label in 'ABC':
            k=mapping[(rid,label)];g=outputs[(k['arm'],rid)];a=case['outputs'][label]
            if (pilot.sha(g['output'])!=k['output_sha256'] or a['text_sha256']!=k['output_sha256']
                    or type(a['material_claim_change']) is not bool or type(a['unresolved']) is not bool
                    or type(a['minor_semantic_shifts']) is not int or a['minor_semantic_shifts']<0):raise ValueError('masked_judgment_binding')
            ranks[(k['arm'],rid)]=local[label]
            results.append({'arm':k['arm'],'example_id':rid,'paper_id':row['paper_id'],'output':g['output'],
                'output_sha256':k['output_sha256'],'assessment':a,'empty':g['empty'],'output_limited':g['output_limited'],
                'copied_source':g['copied_source'],'eligible':not a['material_claim_change'] and not a['unresolved'] and a['minor_semantic_shifts']<=1,
                'source_metrics':metrics(row['source'],g['output']),'reference_metrics':metrics(row['target'],g['output'])})
    for rid in ids:
        for a in primary:
            for b in primary:
                if outputs[(a,rid)]['output']==outputs[(b,rid)]['output'] and ranks[(a,rid)]!=ranks[(b,rid)]:
                    raise ValueError('identical_outputs_must_tie')
    arms={}
    for arm in primary:
        group=[r for r in results if r['arm']==arm]
        counts={'outputs':8,'material_claim_issues':sum(r['assessment']['material_claim_change'] for r in group),
            'eligible_within_local_budget':sum(r['eligible'] for r in group),'exact_source_copies':sum(r['copied_source'] for r in group),
            'empty':sum(r['empty'] for r in group),'output_limited':sum(r['output_limited'] for r in group),
            'token_weighted_development_nll':primary[arm]['token_weighted_development_nll'],
            'median_source_overlap':statistics.median(r['source_metrics']['word_sequence_overlap'] for r in group),
            'median_reference_overlap':statistics.median(r['reference_metrics']['word_sequence_overlap'] for r in group),
            'exact_control_generations':sum(outputs[(arm,rid)]['output']==outputs[('control',rid)]['output'] for rid in ids)}
        if arm!='control':
            pref=Counter('better' if ranks[(arm,rid)]<ranks[('control',rid)] else 'worse' if ranks[(arm,rid)]>ranks[('control',rid)] else 'tie' for rid in ids)
            counts['generic_expression_vs_control']={k:pref[k] for k in ('better','tie','worse')}
            counts['predeclared_quality_gate_passed']=(counts['material_claim_issues']<=3 and counts['eligible_within_local_budget']>=4
                and pref['better']>pref['worse'] and counts['exact_source_copies']==0 and counts['empty']==0 and counts['output_limited']==0)
        arms[arm]=counts
    paired=Counter('better' if ranks[('dpo',rid)]<ranks[('sft',rid)] else 'worse' if ranks[('dpo',rid)]>ranks[('sft',rid)] else 'tie' for rid in ids)
    return {'primary_arms':arms,'dpo_generic_expression_vs_sft':{k:paired[k] for k in ('better','tie','worse')}},results


def build(packet,download,review,approval,output):
    if output.exists():raise ValueError('preserve_existing_review')
    verify_files(packet);pairs,development,config=pilot.load_inputs(packet/'upload/actual_error_pilot_data.json',packet/'upload/actual_error_pilot_config.json')
    approved=json.loads(approval.read_text())
    if approved.get('explicit_user_approval') is not True or approved['packet_manifest_sha256']!=pilot.file_sha(packet/'manifest.json'):
        raise ValueError('approved_packet_changed')
    folder=download/'actual_error_pilot_v1'
    def load(name):return json.loads((folder/name).read_text())
    report=load('evaluation.json');state=load('state.json');reference=load('reference_scores.json')
    validate_execution(report,state,config,pairs,development,load('gradient_checks.json'),load('update_log.json'),load('processor_audit.json'),reference)
    if (pilot.file_sha(download/'preference_pair_pilot.py')!=config['runtime_source_sha256']
            or pilot.file_sha(download/'preference_token_preflight.py')!=config['token_helper_sha256']
            or load('generation_config.json')!=config['expected_generation_config']):raise ValueError('downloaded_source_or_decoding_changed')
    for arm in pilot.ARMS:
        for epoch in (1,2):
            adapter=folder/arm/f'epoch_{epoch}/adapter'
            actual=tensor_fingerprint(adapter/'adapter_model.safetensors')
            cfg=json.loads((adapter/'adapter_config.json').read_text())
            if cfg['r']!=8 or cfg['lora_alpha']!=16 or cfg['base_model_name_or_path']!=config['model_id'] or cfg['target_modules']!=pilot.TARGET_PATTERN:
                raise ValueError('saved_adapter_config_changed')
            if epoch==2:
                if actual!=report['arms'][arm]['final_adapter_tensor_sha256']:raise ValueError('saved_adapter_tensor_fingerprint_changed')
                if any(pilot.file_sha(adapter/name)!=digest for name,digest in report['arms'][arm]['final_adapter_files_sha256'].items()):
                    raise ValueError('saved_adapter_file_changed')
    judgments=json.loads((review/'masked_judgments.private.json').read_text());key=json.loads((review/'masked_key.private.json').read_text())
    summary,results=summarize(report,development,key,judgments)
    summary.update(protocol='preference_pilot_review_v1',user_reported_version=1,slug='oliverlds/preference-pair-pilot-4b',
        execution_complete=True,optimizer_steps=44,processor_sequences_verified=42,reference_sequences_verified=34,
        control_outputs_reproduced=8,adapter_reload_weights_match=True,elapsed_seconds=state['elapsed_seconds'],
        peak_gpu_allocated_bytes=state['peak_gpu_allocated_bytes'],primary_epoch=2,reserved_test_records_read=0,
        detector_calls=0,new_model_calls=0,new_gpu_jobs_submitted=0,additional_training_authorized=False,
        fama_style_established=False,detector_improvement_established=False,assessment_provisional=True,
        limits=judgments['provenance']['limitations'])
    summary['diagnostic_snapshots']=[{'arm':s['arm'],'epoch':s['epoch'],'nll':s['token_weighted_development_nll'],
        'exact_control_generations':sum(g['output']==development[i]['expected_generation']['output'] for i,g in enumerate(s['generations']))}
        for s in report['snapshots']]
    diagnostics={}
    for arm in pilot.ARMS:
        margins=[];losses=[];absolute=[]
        for row in report['arms'][arm]['final_training_scores']:
            rc,rr=(reference['scores'][row['pair_id']][k]['total_log_probability'] for k in ('chosen','rejected'))
            pc,pr=(row[k]['total_log_probability'] for k in ('chosen','rejected'))
            margins.append((pc-pr)-(rc-rr));absolute.append(pc-pr);losses.append(pilot.tokens.dpo_loss(pc,pr,rc,rr))
        diagnostics[arm]={'rows':17,'positive_relative_margin_changes':sum(m>0 for m in margins),
            'mean_relative_margin_change_nats':statistics.mean(margins),'mean_final_dpo_loss':statistics.mean(losses),
            'positive_absolute_chosen_minus_rejected_total_logp':sum(v>0 for v in absolute),
            'mean_absolute_chosen_minus_rejected_total_logp':statistics.mean(absolute),
            'elapsed_seconds':report['arms'][arm]['elapsed_seconds']}
    summary['training_likelihood_diagnostics']=diagnostics
    output.mkdir(parents=True);pilot.write_json(output/'summary.json',summary)
    (output/'per_output.private.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results))
    pilot.write_json(output/'receipt.json',{'packet_manifest_sha256':pilot.file_sha(packet/'manifest.json'),
        'approval_sha256':pilot.file_sha(approval),'download_sha256':{str(p.relative_to(download)):pilot.file_sha(p)
            for p in sorted(download.rglob('*')) if p.is_file()},
        'review_sha256':{p.name:pilot.file_sha(p) for p in sorted(review.glob('*.json*')) if p.is_file()},
        'review_code_sha256':pilot.file_sha(Path(__file__))})
    pilot.write_json(output/'manifest.json',{'files':{p.name:pilot.file_sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','download','review','approval','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.packet,a.download,a.review,a.approval,a.output),indent=2))
