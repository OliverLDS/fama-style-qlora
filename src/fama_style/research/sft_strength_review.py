"""Offline audit of a fixed stronger SFT run and its two archived baselines."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from fama_style.training import sft_strength_pilot as runtime
from .author_control_replication import verify_files
from .preference_pilot_review import tensor_fingerprint
from .neutral_reconstruction_results import metrics

prior=runtime.prior


def rows(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_execution(report,state,config,pairs,development,cases,audits,gradients,updates):
    if (report.get('complete') is not True or state.get('complete') is not True or state.get('phase')!='complete'
            or report.get('training_called') is not True or state.get('training_called') is not True
            or state.get('optimizer_steps')!=22 or report['config']!=config
            or any(report.get(k) is not True for k in ('optimizer_initially_empty','weights_changed','adapter_reload_weights_match','final_evaluation_frozen'))
            or report['initial_adapter_tensor_sha256']!=config['initial_adapter_tensor_sha256']
            or report['final_adapter_tensor_sha256']==report['initial_adapter_tensor_sha256']
            or report['control_exact_matches']!=8 or not 0<state['elapsed_seconds']<1800
            or not 0<state['peak_gpu_allocated_bytes']<16*1024**3):raise ValueError('fixed_strength_execution_contract_failed')
    actual=report['runtime']
    if (any(actual[k]!=config[k] for k in ('packages','model_id','model_revision','runtime_source_sha256','prior_runtime_sha256','token_helper_sha256'))
            or actual['gpu_count']!=1 or 'T4' not in actual['gpu']):raise ValueError('runtime_changed')
    expected=[(e['epoch'],g) for e in config['group_schedule'] for g in e['groups']]
    if len(gradients)!=22 or len(updates)!=22:raise ValueError('twenty_two_gradient_and_update_checks_required')
    for i,((epoch,g),gradient,update) in enumerate(zip(expected,gradients,updates),1):
        if (gradient['step']!=i or gradient['finite'] is not True or not math.isfinite(gradient['norm']) or gradient['norm']<=0
                or update['step']!=i or update['epoch']!=epoch or update['group_id']!=g['group_id']
                or update['pair_ids']!=g['pair_ids'] or not math.isclose(update['learning_rate'],runtime.learning_rate(i))
                or not math.isfinite(update['loss'])):raise ValueError('gradient_group_order_or_learning_rate_changed')
    sequences=[(r['pair_id'],r['encoded']['chosen']) for r in pairs]+[(r['example_id'],r['encoded']) for r in development]
    if len(audits)!=25 or [r['id'] for r in audits]!=[i for i,_ in sequences]:raise ValueError('twenty_five_processor_checks_required')
    for audit,(_,seq) in zip(audits,sequences):
        if (audit['input_ids_sha256']!=prior.sha(json.dumps(seq['input_ids'])) or audit['labels_sha256']!=prior.sha(json.dumps(seq['labels']))
                or audit['prompt_tokens']!=prior.validate_sequence(seq)[0] or audit['full_tokens']!=len(seq['input_ids'])):
            raise ValueError('processor_ids_or_masks_changed')
    if len(report['likelihood_sanity'])!=2 or any(not c['passed'] or not math.isclose(c['manual_mean_nll'],c['builtin_mean_nll'],rel_tol=1e-5,abs_tol=1e-5) for c in report['likelihood_sanity']):
        raise ValueError('actual_likelihood_sanity_failed')
    for stage in ('initial','final'):
        for split,seqs in (('training',sequences[:17]),('development',sequences[17:])):
            scored=report[stage+'_'+split+'_scores'];field='pair_id' if split=='training' else 'example_id'
            if [s[field] for s in scored]!=[i for i,_ in seqs]:raise ValueError('score_coverage_changed')
            for score,(_,seq) in zip(scored,seqs):
                n=prior.validate_sequence(seq)[1]
                if (score['supervised_tokens']!=n or not math.isfinite(score['total_log_probability']) or score['total_log_probability']>0
                        or not math.isclose(score['mean_nll'],-score['total_log_probability']/n)):
                    raise ValueError('likelihood_score_changed')
                if stage=='initial' and split=='training' and abs(score['total_log_probability']-config['initial_chosen_scores'][score[field]]['total_log_probability'])>1e-3:
                    raise ValueError('starting_likelihood_not_reproduced')
    if len(report['control_generations'])!=8:raise ValueError('eight_control_generations_required')
    for g,row in zip(report['control_generations'],development):
        if g['example_id']!=row['example_id'] or any(g[k]!=row['expected_generation'][k] for k in row['expected_generation']):
            raise ValueError('archived_control_changed')
    expected_cases=[('training',r) for r in cases]+[('development',r) for r in development]
    if len(report['final_generations'])!=19 or report['forward_counts']['generation_requests']!=27:
        raise ValueError('fixed_generation_coverage_failed')
    for g,(split,row) in zip(report['final_generations'],expected_cases):
        if (g['split']!=split or g['example_id']!=row['example_id'] or g['pair_id']!=row.get('pair_id')
                or any(g[k]!=row[k] for k in ('source_sha256','prompt_sha256')) or prior.sha(g['output'])!=g['output_sha256']
                or not 0<g['output_tokens']<=512 or g['empty']!=(not bool(g['output'].strip()))
                or g['copied_source']!=(prior.tokens.norm(g['output'])==prior.tokens.norm(row['source']))):
            raise ValueError('final_generation_binding_changed')


def masked_comparison(report,references):
    generated={(g['split'],g['pair_id'] if g['split']=='training' else g['example_id']):g for g in report['final_generations']}
    bundle=[];key=[]
    for ref in references:
        split=ref['split'];cid=ref['case_id'];order=['control','weak_sft','strong_sft']
        random.Random(int(hashlib.sha256(('sft-strength-review-v1:'+split+':'+cid).encode()).hexdigest(),16)).shuffle(order)
        row={k:ref[k] for k in ('split','case_id','example_id','paper_id','source','target')};row['outputs']={}
        for label,arm in zip('ABC',order):
            g=generated[(split,cid)] if arm=='strong_sft' else ref[arm]
            row['outputs'][label]=g['output'];key.append({'split':split,'case_id':cid,'label':label,'arm':arm,'output_sha256':prior.sha(g['output'])})
        bundle.append(row)
    return bundle,key


def summarize(bundle,key,judgments):
    mapping={(r['split'],r['case_id'],r['label']):r for r in key};judged={(r['split'],r['case_id']):r for r in judgments['cases']}
    if (len(bundle)!=19 or len(key)!=57 or len(mapping)!=57 or len(judged)!=19 or len(judgments['cases'])!=19
            or Counter(r['split'] for r in bundle)!=Counter({'training':11,'development':8})):raise ValueError('nineteen_three_way_judgments_required')
    result=[];ranks={}
    for row in bundle:
        split,cid=row['split'],row['case_id'];case=judged[(split,cid)];groups=case['ranking_best_first']
        if any(not g for g in groups) or sorted(l for g in groups for l in g)!=list('ABC'):raise ValueError('invalid_rank_groups')
        rank={l:i for i,g in enumerate(groups) for l in g};texts={}
        for label in 'ABC':
            k=mapping[(split,cid,label)];a=case['outputs'][label];text=row['outputs'][label];arm=k['arm']
            if (arm in texts or prior.sha(text)!=k['output_sha256'] or a['output_sha256']!=k['output_sha256']
                    or type(a['material_claim_change']) is not bool or type(a['unresolved']) is not bool
                    or type(a['minor_semantic_shifts']) is not int or a['minor_semantic_shifts']<0 or not a['notes']):raise ValueError('judgment_byte_binding_failed')
            ranks[(split,cid,arm)]=rank[label];texts[arm]=text
            result.append({'split':split,'case_id':cid,'example_id':row['example_id'],'paper_id':row['paper_id'],'arm':arm,
                'output':text,'output_sha256':k['output_sha256'],'assessment':a,
                'eligible':not a['material_claim_change'] and not a['unresolved'] and a['minor_semantic_shifts']<=1,
                'exact_source_copy':prior.tokens.norm(text)==prior.tokens.norm(row['source']),
                'source_metrics':metrics(row['source'],text),'reference_metrics':metrics(row['target'],text)})
        if set(texts)!={'control','weak_sft','strong_sft'}:raise ValueError('three_conditions_required')
        for a in texts:
            for b in texts:
                if texts[a]==texts[b] and ranks[(split,cid,a)]!=ranks[(split,cid,b)]:raise ValueError('equal_texts_must_tie')
    by={(r['split'],r['case_id'],r['arm']):r for r in result};summary={}
    for split in ('training','development'):
        ids=[r['case_id'] for r in bundle if r['split']==split];arms={}
        for arm in ('control','weak_sft','strong_sft'):
            group=[r for r in result if r['split']==split and r['arm']==arm]
            item={'outputs':len(group),'material_claim_issues':sum(r['assessment']['material_claim_change'] for r in group),
                'eligible_within_local_budget':sum(r['eligible'] for r in group),'exact_source_copies':sum(r['exact_source_copy'] for r in group),
                'exact_control_outputs':sum(by[(split,cid,arm)]['output']==by[(split,cid,'control')]['output'] for cid in ids),
                'median_source_overlap':statistics.median(r['source_metrics']['word_sequence_overlap'] for r in group),
                'median_reference_overlap':statistics.median(r['reference_metrics']['word_sequence_overlap'] for r in group)}
            if arm!='control':
                for baseline in ('control','weak_sft'):
                    if baseline==arm:continue
                    pref=Counter('better' if ranks[(split,cid,arm)]<ranks[(split,cid,baseline)] else 'worse' if ranks[(split,cid,arm)]>ranks[(split,cid,baseline)] else 'tie' for cid in ids)
                    item['generic_expression_vs_'+baseline]={k:pref[k] for k in ('better','tie','worse')}
                transitions=Counter('resolved' if by[(split,cid,'control')]['assessment']['material_claim_change'] and not by[(split,cid,arm)]['assessment']['material_claim_change'] else
                    'new_material_case' if not by[(split,cid,'control')]['assessment']['material_claim_change'] and by[(split,cid,arm)]['assessment']['material_claim_change'] else 'unchanged' for cid in ids)
                item['material_case_transitions_vs_control']={k:transitions[k] for k in ('resolved','new_material_case','unchanged')}
            arms[arm]=item
        summary[split]=arms
    return summary,result


def build(packet,download,review,output):
    if output.exists():raise ValueError('preserve_existing_strength_review')
    verify_files(packet);pairs,dev,cases,config=runtime.load_inputs(packet/'upload/sft_strength_data.json',packet/'upload/sft_strength_config.json')
    folder=download/'sft_strength_v1'
    def load(name):return json.loads((folder/name).read_text())
    report=load('evaluation.json');state=load('state.json')
    validate_execution(report,state,config,pairs,dev,cases,load('processor_audit.json'),load('gradient_checks.json'),load('update_log.json'))
    for name,key in (('sft_strength_pilot.py','runtime_source_sha256'),('preference_pair_pilot.py','prior_runtime_sha256'),('preference_token_preflight.py','token_helper_sha256')):
        if prior.file_sha(download/name)!=config[key]:raise ValueError('executed_source_changed')
    if load('generation_config.json')!=config['expected_generation_config']:raise ValueError('generation_config_changed')
    if [(c['epoch'],c['step']) for c in report['checkpoints']]!=[(1,11),(2,22)]:raise ValueError('checkpoint_coverage_changed')
    for c in report['checkpoints']:
        adapter=folder/('epoch_'+str(c['epoch']))/'adapter'
        if any(prior.file_sha(adapter/name)!=digest for name,digest in c['files_sha256'].items()):raise ValueError('checkpoint_file_changed')
        fingerprint=tensor_fingerprint(adapter/'adapter_model.safetensors')
        if c['epoch']==2 and fingerprint!=report['final_adapter_tensor_sha256']:raise ValueError('final_tensor_fingerprint_changed')
    refs=rows(packet/'private_key/comparisons.private.jsonl');bundle,key=masked_comparison(report,refs)
    if bundle!=rows(review/'masked_bundle.private.jsonl') or key!=json.loads((review/'masked_key.private.json').read_text()):raise ValueError('masking_not_reproducible')
    judgments=json.loads((review/'masked_judgments.private.json').read_text());summary,result=summarize(bundle,key,judgments)
    for stage in ('initial','final'):
        for split in ('training','development'):
            scored=report[stage+'_'+split+'_scores']
            summary.setdefault('target_nll',{})[stage+'_'+split]=-sum(r['total_log_probability'] for r in scored)/sum(r['supervised_tokens'] for r in scored)
    flags={k:sum(g[k] for g in report['final_generations']) for k in ('empty','output_limited','copied_source')}
    strong=summary['development']['strong_sft'];preference=strong['generic_expression_vs_control']
    summary.update(protocol='sft_strength_review_v1',slug='oliverlds/sft-strength-4b',user_reported_version=1,
        execution_complete=True,optimizer_updates=22,learning_rate=1e-5,initial_control_matches=8,initial_training_likelihoods_reproduced=17,
        processor_sequences_verified=25,final_generations=19,total_new_generations=27,final_generation_flags=flags,
        adapter_reload_weights_match=True,elapsed_seconds=state['elapsed_seconds'],peak_gpu_allocated_bytes=state['peak_gpu_allocated_bytes'],
        development_quality_gate_passed=strong['material_claim_issues']<=3 and strong['eligible_within_local_budget']>=4
            and preference['better']>preference['worse'] and strong['exact_source_copies']==0 and not any(flags.values()),
        new_gpu_jobs_submitted=0,additional_training_authorized=False,detector_calls=0,reserved_test_records_read=0,
        assessment_provisional=True,fama_style_established=False,limits=judgments['provenance']['limits'])
    output.mkdir(parents=True);prior.write_json(output/'summary.json',summary)
    (output/'per_output.private.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result))
    prior.write_json(output/'receipt.json',{'packet_manifest_sha256':prior.file_sha(packet/'manifest.json'),'review_code_sha256':prior.file_sha(Path(__file__)),
        'download_sha256':{str(p.relative_to(download)):prior.file_sha(p) for p in sorted(download.rglob('*')) if p.is_file()},
        'annotation_sha256':{p.name:prior.file_sha(p) for p in sorted(review.glob('*.json*')) if p.is_file()}})
    prior.write_json(output/'manifest.json',{'files':{p.name:prior.file_sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','download','review','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.packet,a.download,a.review,a.output),indent=2))
