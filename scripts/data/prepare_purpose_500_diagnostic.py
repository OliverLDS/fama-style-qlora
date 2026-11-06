"""Freeze four held-out contents and reconstruct through two free Gemini calls."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from fama_style.research.content_brief_v4 import canonicalize_note_order, validate_brief_v4
from fama_style.training import gpt6_purpose_500 as training

RUN=ROOT/'research_runs/data_scaling_v1/purpose_500_prompt_input_v1'
PARENT=ROOT/'research_runs/data_scaling_v1/gpt6_purpose_500_4b_v1'
DOCS=ROOT/'docs/data_scaling_v1'
MODEL='gemini-3-flash-preview'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def rows(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def write_rows(path,values):path.write_text(''.join(json.dumps(r,sort_keys=True,ensure_ascii=False)+'\n' for r in values))


def select(report):
    originals=report['snapshots'][0]['generations']
    candidates=[r for r in originals if r['input_condition']=='gemini_input']
    selected=[]
    for group in training.GROUPS:
        # Frozen selection uses identifiers only, not generation/detector outcomes.
        chosen=min((r for r in candidates if r['style_group']==group),
                   key=lambda r:training.sha('purpose500-input-prompt-v1|'+r['content_id']))
        selected.append(dict(content_id=chosen['content_id'],paper_id=chosen['paper_id'],style_group=group,
            reference=chosen['reference'],target_sha256=training.sha(chosen['reference'])))
    if len({r['content_id'] for r in selected})!=4:raise ValueError('four_distinct_contents_required')
    return selected


def job(stage,prompt):
    return dict(stage=stage,prompt=prompt,prompt_sha256=training.sha(prompt))


def prepare():
    if RUN.exists():raise ValueError('refuse_overwrite_diagnostic')
    receipt=read(PARENT/'download_manifest.json')
    if receipt['version']!=1 or not receipt['version_explicitly_verified_via_SDK']:raise ValueError('verified_parent_required')
    for name,h in receipt['files'].items():
        if sha(PARENT/'raw'/name)!=h:raise ValueError('parent_changed')
    parent=PARENT/'raw/gpt6_purpose_500_v1/evaluation.json';report=read(parent)
    if not report['complete'] or report['config']['protocol']!=training.PROTOCOL:raise ValueError('completed_parent_required')
    old=read(ROOT/'research_runs/data_scaling_v1/stage_0500_expansion_v1/config.json')
    if old['model']!=MODEL or not old['free_tier_only'] or not old['free_project_confirmed_by_user'] or old['paid_fallback']:
        raise ValueError('free_only_authorization_required')
    selected=select(report)
    train_papers={r['paper_id'] for r in rows(ROOT/'data/gpt6_purpose_500_v1/training_0500.private.jsonl')}
    if train_papers & {r['paper_id'] for r in selected}:raise ValueError('heldout_paper_leakage')
    RUN.mkdir(parents=True);(RUN/'responses').mkdir()
    write_rows(RUN/'selected_contents.private.jsonl',selected)
    brief_prompt=(DOCS/'BRIEF_ABSTRACTION_V4.md').read_text()
    inputs=[dict(id=r['content_id'],target_sha256=r['target_sha256'],original_paragraph=r['reference']) for r in selected]
    prompt=brief_prompt+'\nAPI transport: return a JSON ARRAY of the four objects instead of JSONL. No Markdown.\nINPUTS:\n'+json.dumps(inputs,ensure_ascii=False)
    write(RUN/'brief_job.private.json',job('brief',prompt))
    write(RUN/'config.json',dict(protocol='purpose_500_prompt_input_v1',model=MODEL,free_tier_only=True,
        free_project_confirmed_by_user=True,paid_fallback=False,max_requests=2,max_output_tokens=16384,
        temperature=.5,top_p=1,top_k=40,thinking_level='low',response_mime_type='application/json',
        training_authorized=False,teacher_match_limit='Gemini reconstruction procedure, not an exact GPT-6 teacher replication',
        parent_report_sha256=sha(parent),selection_sha256=sha(RUN/'selected_contents.private.jsonl'),
        brief_job_sha256=sha(RUN/'brief_job.private.json'),builder_sha256=sha(Path(__file__)),
        provider_adapter_sha256=sha(Path(__file__).with_suffix('.R')),
        abstraction_prompt_sha256=sha(DOCS/'BRIEF_ABSTRACTION_V4.md'),
        reconstruction_prompt_sha256=sha(DOCS/'BRIEF_RECONSTRUCTION_V4.md')))
    print('Prepared four held-out contents; zero provider calls so far.')


def check():
    c=read(RUN/'config.json')
    if c['model']!=MODEL or not c['free_tier_only'] or not c['free_project_confirmed_by_user'] or c['paid_fallback'] or c['training_authorized']:
        raise ValueError('free_only_contract')
    for path,key in [(Path(__file__),'builder_sha256'),(Path(__file__).with_suffix('.R'),'provider_adapter_sha256'),
        (PARENT/'raw/gpt6_purpose_500_v1/evaluation.json','parent_report_sha256'),
        (RUN/'selected_contents.private.jsonl','selection_sha256'),(RUN/'brief_job.private.json','brief_job_sha256'),
        (DOCS/'BRIEF_ABSTRACTION_V4.md','abstraction_prompt_sha256'),(DOCS/'BRIEF_RECONSTRUCTION_V4.md','reconstruction_prompt_sha256')]:
        if sha(path)!=c[key]:raise ValueError('frozen_dependency_changed')
    selected=rows(RUN/'selected_contents.private.jsonl')
    if selected!=select(read(PARENT/'raw/gpt6_purpose_500_v1/evaluation.json')):raise ValueError('selection_changed')
    return c,selected


def call(stage):
    destination=RUN/'responses'/(stage+'.private.json')
    if not destination.exists():
        print('Free Gemini request: '+stage,flush=True)
        p=subprocess.run(['Rscript',str(Path(__file__).with_suffix('.R')),str(RUN),stage,str(destination)],
                         capture_output=True,timeout=660)
        if p.returncode or not destination.exists():
            write(RUN/'state.json',dict(status='stopped',stage=stage,reason='quota_or_provider_failure_details_suppressed',paid_fallback=False))
            raise RuntimeError('free_request_failed_progress_retained_no_retry')
    response=read(destination)
    expected=read(RUN/(stage+'_job.private.json'))
    if response['status']!='generated' or response['prompt_sha256']!=expected['prompt_sha256']:
        raise ValueError('previous_provider_failure_or_prompt_changed_no_retry')
    result=json.loads(response['text'])
    if not isinstance(result,list):raise ValueError('JSON_array_required')
    return result,response


def generate():
    c,selected=check();by={r['content_id']:r for r in selected}
    briefs,response=call('brief')
    if len(briefs)!=4 or {r['id'] for r in briefs}!=set(by):raise ValueError('brief_coverage')
    projected=[];enhanced=[]
    for r in briefs:
        validate_brief_v4(r)
        if r['target_sha256']!=by[r['id']]['target_sha256']:raise ValueError('brief_target_binding')
        r=canonicalize_note_order(r);enhanced.append(r)
        projected.append(dict(id=r['id'],target_sha256=r['target_sha256'],brief_status=r['status'],brief_notes=r['notes'],
            brief={k:r['brief'][k] for k in ['topic','purpose','key_points','anchors']}))
    write_rows(RUN/'enhanced_briefs.private.jsonl',enhanced)
    prompt=(DOCS/'BRIEF_RECONSTRUCTION_V4.md').read_text()+'\nAPI transport: return a JSON ARRAY of the four objects instead of JSONL. No Markdown.\nINPUTS:\n'+json.dumps(projected,ensure_ascii=False)
    spec=job('reconstruction',prompt);path=RUN/'reconstruction_job.private.json'
    if path.exists() and read(path)!=spec:raise ValueError('reconstruction_job_changed')
    write(path,spec)
    outputs,raw=call('reconstruction')
    if len(outputs)!=4 or {r['id'] for r in outputs}!=set(by):raise ValueError('reconstruction_coverage')
    accepted=[]
    for r in outputs:
        if set(r)!={'id','target_sha256','plain_english','rewrite_status','review_notes'} or r['target_sha256']!=by[r['id']]['target_sha256']:
            raise ValueError('reconstruction_binding')
        if not isinstance(r['plain_english'],str) or not r['plain_english'].strip() or '\n\n' in r['plain_english']:
            raise ValueError('complete_single_paragraph_required')
        if r['rewrite_status'] not in ['rewritten','limited_rewrite_room','source_problem'] or not isinstance(r['review_notes'],list):
            raise ValueError('reconstruction_status')
        accepted.append(dict(by[r['id']],source=r['plain_english'],source_sha256=training.sha(r['plain_english']),
            teacher=MODEL,actual_model_version=raw.get('model_version'),
            rewrite_status=r['rewrite_status'],review_notes=r['review_notes'],
            eligibility='four-record diagnostic only; formal checks, no quality filtering or independent semantic certification'))
    accepted.sort(key=lambda r:[x['content_id'] for x in selected].index(r['content_id']))
    write_rows(RUN/'fresh_inputs.private.jsonl',accepted)
    write(RUN/'generation_manifest.json',dict(protocol=c['protocol'],builder_sha256=sha(Path(__file__)),
        files={str(p.relative_to(RUN)):sha(p) for p in sorted(RUN.rglob('*')) if p.is_file() and p.name not in ['state.json','generation_manifest.json']},
        model_calls=2,paid_fallback=False,training_calls=0))
    write(RUN/'state.json',dict(status='four_fresh_inputs_generated',model_calls=2,training_calls=0,paid_fallback=False))
    print('Saved all four fresh reconstructions; no training or detector calls.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','check','generate']);a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='check':check();print('Frozen preparation verified; no API call.')
    else:generate()
