"""Finalize a complete checkpoint and prepare blind manual detector texts, never scan."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import random
import re

from fama_style.evaluation import milestone_checkpoint_4b as runtime
from .author_control_replication import verify_files
from .neutral_reconstruction import write_json,write_rows
from .neutral_reconstruction_results import file_sha


def rows(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def collect_complete(jobs,reused,report,state,config):
    if (not report.get('complete') or not state.get('complete') or report.get('training_called') is not False
            or state.get('training_called') is not False or state.get('optimizer_steps')!=0
            or state.get('completed_jobs')!=14 or report['config']!=config):
        raise ValueError('all_missing_outputs_must_finish_before_detector_export')
    pending={j['job_id']:j for j in jobs if j['run_required']};archived={j['job_id']:j for j in jobs if not j['run_required']}
    if (len(report['generations'])!=14 or len({r['job_id'] for r in report['generations']})!=14
            or {r['job_id'] for r in report['generations']}!=set(pending)
            or len(reused)!=10 or len({r['job_id'] for r in reused})!=10 or {r['job_id'] for r in reused}!=set(archived)):
        raise ValueError('exact_fourteen_new_plus_ten_archived_required')
    output=[]
    for origin,group,expected in [('archived',reused,archived),('new',report['generations'],pending)]:
        for r in group:
            job=expected[r['job_id']]
            if (any(r[k]!=job[k] for k in ('example_id','condition','source','source_sha256','prompt_sha256'))
                    or not isinstance(r['output'],str) or r['empty']!=(not bool(r['output'].strip()))
                    or r['copied_source']!=(' '.join(r['output'].split())==' '.join(job['source'].split()))):
                raise ValueError('generation_binding_or_flags_changed')
            output.append(r|{'origin':origin,'output_sha256':runtime.sha(r['output'])})
    return sorted(output,key=lambda r:r['job_id'])


def export(package,download,output):
    if output.exists():raise ValueError('refuse_overwrite_detector_checkpoint')
    verify_files(package);jobs,config=runtime.load_inputs(package/'upload/milestone_checkpoint_jobs.jsonl',package/'upload/milestone_checkpoint_config.json')
    folder=download/'milestone_checkpoint_4b_v1'
    report=json.loads((folder/'evaluation.json').read_text());state=json.loads((folder/'state.json').read_text())
    combined=collect_complete(jobs,rows(package/'private_key/reused_outputs.private.jsonl'),report,state,config)
    plan=json.loads((package/'plan.json').read_text());actual=report['runtime']
    if (actual['runtime_source_sha256']!=plan['runtime_source_sha256'] or file_sha(download/'milestone_checkpoint_4b.py')!=plan['runtime_source_sha256']
            or actual['packages']!=config['expected_packages'] or actual['model_revision']!=runtime.REVISION
            or actual['model_id']!=runtime.MODEL_ID or actual['gpu_count']!=1 or 'T4' not in actual['gpu']
            or actual.get('evaluation_references_supplied') is not False):raise ValueError('reuse_runtime_compatibility_failed')
    if json.loads((folder/'generation_config.json').read_text())!=config['expected_generation_config']:
        raise ValueError('generation_settings_changed')
    audits=json.loads((folder/'token_audit.json').read_text());by={r['job_id']:r for r in audits}
    if len(audits)!=24 or len(by)!=24 or set(by)!={j['job_id'] for j in jobs}:raise ValueError('all_twenty_four_prompts_must_be_audited')
    for job in jobs:
        a=by[job['job_id']];limit=4096 if job['condition']=='base_source_structural_prior' else 1024
        if (a['input_tokens']<=0 or a['input_tokens']+512>limit or a['context_guard']!=limit or a['run_required']!=job['run_required']
                or re.fullmatch('[0-9a-f]{64}',a['input_ids_sha256']) is None
                or (not job['run_required'] and a['input_tokens']!=config['reused_prompt_tokens'][job['job_id']])):
            raise ValueError('prompt_context_or_archive_token_count_changed')
    refs=rows(package/'private_key/reference_key.private.jsonl');subset=json.loads((package/'private_key/frozen_subset.json').read_text())
    rounds={r['example_id']:r['round'] for r in subset};ref_by={r['example_id']:r for r in refs}
    items=[]
    for ref in refs:
        texts={'GPT6_input':ref['source'],'historical_human_reference':ref['target']}
        texts.update({r['condition']:r['output'] for r in combined if r['example_id']==ref['example_id']})
        if len(texts)!=5:raise ValueError('complete_matched_five_texts_per_case')
        for condition,text in texts.items():
            text_sha=runtime.sha(text);scan_id='SCAN_'+runtime.sha(runtime.PROTOCOL+'|text|'+text_sha)[:16]
            items.append({'item_id':'ITEM_'+runtime.sha(ref['example_id']+'|'+condition)[:16],
                'scan_id':scan_id,'example_id':ref['example_id'],'paper_id':ref['paper_id'],'round':rounds[ref['example_id']],
                'condition':condition,'text':text,'text_sha256':text_sha,'words_estimate':len(text.split()),
                'empty':not bool(text.strip()),'status':'not_tested'})
    unique={r['scan_id']:r for r in items};output.mkdir(parents=True);(output/'detector/texts').mkdir(parents=True)
    for sid,item in unique.items():(output/'detector/texts'/f'{sid}.txt').write_text(item['text'])
    write_rows(output/'combined_outputs.private.jsonl',combined)
    write_rows(output/'detector/PRIVATE_KEY.jsonl',items)
    with (output/'detector/results_template.csv').open('w',newline='') as stream:
        fields=['scan_id','text_sha256','words_estimate','provider','test_date','provider_version','threshold_rule','status','provider_label','score','score_scale','notes']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for sid in sorted(unique,key=lambda s:runtime.sha('blind-order|'+s)):
            item=unique[sid];writer.writerow({'scan_id':sid,'text_sha256':item['text_sha256'],'words_estimate':item['words_estimate'],'provider':'GPTZero','status':'not_tested'})
    first={r['scan_id'] for r in items if r['round']==1};words=sum(r['words_estimate'] for r in unique.values())
    round1=sum(unique[s]['words_estimate'] for s in first);cal=plan['initial_calibration_pair_words_estimate']
    policy=json.loads((package/'detector_policy.json').read_text());cap=policy['target_max_project_words']
    write_json(output/'detector/budget.json',{'checkpoint_slots':40,'unique_texts':len(unique),
        'actual_checkpoint_words_estimate':words,'round_one_words_estimate':round1,'initial_pair_words_estimate':cal,
        'estimated_total_project_words':words+cal,'target_max_project_words':cap,
        'suggested_complete_rounds':2 if words+cal<=cap else 1 if round1+cal<=cap else 0,
        'actual_provider_charges_and_remaining_balance_known':False,'automatic_provider_calls':0,
        'rule':'Check actual remaining allowance before scanning; use complete four-paper rounds, all five texts per case. Never trim or choose outputs by quality.'})
    write_json(output/'detector/paper_balanced_rounds.private.json',{
        str(i):{'example_ids':[r['example_id'] for r in subset if r['round']==i],
                'scan_ids':sorted({r['scan_id'] for r in items if r['round']<=i})} for i in (1,2)})
    rng=random.Random(34812);bundle=[];key=[];template=[]
    for meta in subset:
        rid=meta['example_id'];outs=[r for r in combined if r['example_id']==rid];rng.shuffle(outs)
        masked={}
        for label,r in zip('ABC',outs):
            masked[label]=r['output'];key.append({'example_id':rid,'label':label,'condition':r['condition'],'output_sha256':r['output_sha256']})
            template.append({'example_id':rid,'label':label,'text_sha256':r['output_sha256'],'status':'unassessed',
                'minor_semantic_shifts':None,'material_claim_change':None,'unresolved':None,'notes':''})
        ref=ref_by[rid];bundle.append({'example_id':rid,'source':ref['source'],'reference':ref['target'],
            'inherited_needs_review':ref['needs_review'],'inherited_needs_external_context':ref['needs_external_context'],'outputs':masked})
    write_rows(output/'masked_quality_bundle.private.jsonl',bundle);write_json(output/'masked_quality_key.private.json',key)
    write_rows(output/'quality_assessment_template.private.jsonl',template)
    summary={'protocol':runtime.PROTOCOL,'status':'all_outputs_frozen_manual_scans_pending','model_outputs':24,
        'new_outputs':14,'archived_outputs':10,'evaluation_cases':8,'detector_slots':40,'unique_detector_texts':len(unique),
        'checkpoint_words_estimate':words,'model_output_flags':{
            'empty':sum(r['empty'] for r in combined),'output_limited':sum(r['output_limited'] for r in combined)},
        'quality_assessed':False,'detector_calls_made':0,'scores_observed':0,'training_called':False}
    write_json(output/'summary.json',summary)
    write_json(output/'receipt.json',{'package_manifest_sha256':file_sha(package/'manifest.json'),
        'download_sha256':{str(p.relative_to(download)):file_sha(p) for p in sorted(download.rglob('*')) if p.is_file()}})
    write_json(output/'manifest.json',{'protocol':runtime.PROTOCOL,'code_sha256':file_sha(Path(__file__)),
        'files':{str(p.relative_to(output)):file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('package','download','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(export(a.package,a.download,a.output)))


if __name__=='__main__':main()
