"""Complete matched Gemini-only 200/500 cohorts using free inference; no training."""
import argparse
from collections import Counter
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import time

SPEC=importlib.util.spec_from_file_location('mixed500',Path(__file__).with_name('expand_stage_500.py'))
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)
SPEC2=importlib.util.spec_from_file_location('minimal500',Path(__file__).with_name('review_stage_500_minimal.py'))
q=importlib.util.module_from_spec(SPEC2);SPEC2.loader.exec_module(q)
RUN=m.ROOT/'research_runs/data_scaling_v1/gemini_track_0500_v1'
OUT=m.ROOT/'data/data_scaling_v1/gemini_track_v1'
PARENT=m.OUT/'training.private.jsonl'
ADAPTER=Path(__file__).with_name('expand_stage_500.R')


def verify_files(folder):
    manifest=m.read(folder/'manifest.json')
    for name,h in manifest['files'].items():
        if m.digest(folder/name)!=h:raise ValueError('parent_artifact_changed')


def prepare():
    if RUN.exists():raise ValueError('refuse_overwrite_Gemini_preparation')
    verify_files(PARENT.parent)
    parents=m.lines(PARENT)
    if len(parents)!=500 or Counter(r['style_group'] for r in parents)!=Counter({g:125 for g in m.GROUPS}):raise ValueError('mixed_500_required')
    trial=m.ROOT/'research_runs/gemini_teacher_trial_v1';verify_files(trial)
    prior={r['target_sha256']:r for r in m.lines(trial/'pairs.private.jsonl')}
    records=[];seeds=[];missing=[]
    for i,row in enumerate(parents,1):
        if row['split']!='training' or not row['training_ready'] or row['target_sha256']!=m.sha(row['target']):raise ValueError('approved_training_parent')
        rid=f'G{i:03d}'
        target=dict(id=rid,index=i,mixed_id=row['id'],target=row['target'],target_sha256=row['target_sha256'],paper_id=row['paper_id'],style_group=row['style_group'])
        records.append(target)
        if row['provenance'].get('generator')==m.MODEL:
            seeds.append(dict(id=rid,plain_english=row['source'],source_sha256=row['source_sha256'],origin='approved_mixed_reuse',
                generation_evidence=row['provenance'],review_required=False))
        elif row['target_sha256'] in prior:
            old=prior[row['target_sha256']]
            if old['generator']!=m.MODEL or old['split']!='training' or old['source_sha256']!=m.sha(old['plain_english']) or old['original_paragraph']!=row['target']:raise ValueError('trial_reuse_binding')
            seeds.append(dict(id=rid,plain_english=old['plain_english'],source_sha256=old['source_sha256'],origin='historical_Gemini_trial_draft',
                generation_evidence={k:old[k] for k in ['generator','actual_model_version','prompt_sha256','response_sha256']},review_required=True))
        else:missing.append(target)
    if Counter(x['origin'] for x in seeds)!=Counter(approved_mixed_reuse=214,historical_Gemini_trial_draft=8):raise ValueError('reuse_inventory_changed')
    RUN.mkdir(parents=True);(RUN/'generation').mkdir();(RUN/'verification').mkdir()
    m.write_lines(RUN/'targets.private.jsonl',records);m.write_lines(RUN/'seeds.private.jsonl',seeds)
    jobs=[]
    for start in range(0,len(missing),35):
        chunk=missing[start:start+35];prompt=m.GEN+'\nINPUTS:\n'+json.dumps([dict(id=r['id'],original_paragraph=r['target']) for r in chunk],ensure_ascii=False)
        jobs.append(dict(index=len(jobs)+1,ids=[r['id'] for r in chunk],prompt=prompt,prompt_sha256=m.sha(prompt)))
    m.write_lines(RUN/'generation/jobs.private.jsonl',jobs)
    config={k:m.read(m.RUN/'config.json')[k] for k in ['model','free_tier_only','free_project_confirmed_by_user','paid_fallback','max_output_tokens','temperature','top_p','top_k','thinking_level','response_mime_type']}
    config.update(protocol='gemini_track_0500_v1',training_authorized=False,parent_mixed_sha256=m.digest(PARENT),
        parent_manifest_sha256=m.digest(PARENT.parent/'manifest.json'),trial_manifest_sha256=m.digest(trial/'manifest.json'),
        adapter_sha256=m.digest(ADAPTER),builder_sha256=m.digest(Path(__file__)),
        accepted_sources_reused=214,trial_drafts_recovered=8,missing_sources_to_generate=len(missing),
        new_sources_to_qualify=286,teacher_weights_immutable=False,detector_score_used_for_selection=False)
    m.write(RUN/'config.json',config)
    m.write(RUN/'preparation_manifest.json',dict(files={str(p.relative_to(RUN)):m.digest(p) for p in RUN.rglob('*') if p.is_file()}))
    print('Prepared: 214 accepted inputs reused, 8 earlier Gemini drafts recovered for review, 278 missing inputs in 8 requests. No calls yet.')


def check():
    c=m.read(RUN/'config.json')
    if c['model']!=m.MODEL or not c['free_tier_only'] or not c['free_project_confirmed_by_user'] or c['paid_fallback'] or c['training_authorized']:
        raise ValueError('free_only_scope')
    if m.digest(PARENT)!=c['parent_mixed_sha256'] or m.digest(ADAPTER)!=c['adapter_sha256'] or m.digest(Path(__file__))!=c['builder_sha256']:raise ValueError('frozen_source_changed')
    for name,h in m.read(RUN/'preparation_manifest.json')['files'].items():
        if m.digest(RUN/name)!=h:raise ValueError('prepared_inputs_changed')
    return c


def call_jobs(phase):
    check();jobs=m.lines(RUN/phase/'jobs.private.jsonl')
    for job in jobs:
        path=RUN/phase/f"response_{job['index']:02d}.private.json"
        if path.exists():
            r=m.read(path)
            if r['status']!='generated':print('Prior failure retained; no fallback.');return
        else:
            print(f"{phase} {job['index']}/{len(jobs)} ({len(job['ids'])} records)",flush=True)
            try:result=subprocess.run(['Rscript',str(ADAPTER),str(RUN),phase,str(job['index']),str(path)],capture_output=True,timeout=620)
            except subprocess.TimeoutExpired:print('Paused on request deadline; no fallback.',flush=True);return
            if result.returncode or not path.exists() or m.read(path)['status']!='generated':
                category=m.read(path).get('error_category','incomplete_or_blocked') if path.exists() else 'adapter_failure'
                print('Paused: '+category+'; completed records retained; no paid fallback.',flush=True);return
            r=m.read(path)
        if r['prompt_sha256']!=job['prompt_sha256']:raise ValueError('response_prompt_changed')
        try:
            (m.parse_array if phase=='generation' else q.parse)(r['text'],job['ids'])
        except (ValueError,KeyError,TypeError):print('Paused on schema/ID issue; full raw response retained.',flush=True);return
        print(f"Saved/validated {phase} {job['index']}/{len(jobs)}",flush=True)
        if job['index']<len(jobs):time.sleep(21)
    print(phase+' complete.',flush=True)


def drafts():
    by={r['id']:r for r in m.lines(RUN/'seeds.private.jsonl')}
    for j in m.lines(RUN/'generation/jobs.private.jsonl'):
        p=RUN/'generation'/f"response_{j['index']:02d}.private.json";response=m.read(p)
        if response['status']!='generated' or response['prompt_sha256']!=j['prompt_sha256']:raise ValueError('generation_incomplete')
        for r in m.parse_array(response['text'],j['ids']):
            if r['id'] in by:raise ValueError('duplicate_draft')
            by[r['id']]=dict(r,source_sha256=m.sha(r['plain_english']),origin='new_Gemini_generation',review_required=True,
                generation_evidence=dict(generator=m.MODEL,response_sha256=m.digest(p),prompt_sha256=j['prompt_sha256'],model_version=response.get('model_version'),thinking_level=response['thinking_level']))
    if len(by)!=500:raise ValueError('all_five_hundred_drafts_required')
    return by


def prepare_review():
    check();path=RUN/'verification/jobs.private.jsonl'
    if path.exists():raise ValueError('refuse_overwrite_review_jobs')
    by=drafts();targets=[r for r in m.lines(RUN/'targets.private.jsonl') if by[r['id']]['review_required']];jobs=[]
    for start in range(0,len(targets),48):
        chunk=targets[start:start+48];prompt=q.PROMPT+'\nPAIRS:\n'+json.dumps([dict(id=r['id'],HUMAN=r['target'],NEUTRAL=by[r['id']]['plain_english']) for r in chunk],ensure_ascii=False)
        jobs.append(dict(index=len(jobs)+1,ids=[r['id'] for r in chunk],prompt=prompt,prompt_sha256=m.sha(prompt)))
    m.write_lines(path,jobs);m.write_lines(RUN/'drafts.private.jsonl',[by[r['id']] for r in m.lines(RUN/'targets.private.jsonl')])
    m.write(RUN/'verification/preparation_manifest.json',dict(jobs_sha256=m.digest(path),drafts_sha256=m.digest(RUN/'drafts.private.jsonl'),review_spec_sha256=m.digest(Path(q.__file__))))
    print(f'Prepared {len(targets)} meaning-only reviews in {len(jobs)} requests; accepted 214 inputs remain unchanged.')


def freeze():
    check();parents=m.lines(PARENT);targets=m.lines(RUN/'targets.private.jsonl');by=drafts()
    decisions=m.lines(RUN/'final_decisions.private.jsonl');dv={r['id']:r for r in decisions}
    evidence=m.read(RUN/'final_decisions_manifest.json')
    for name,h in evidence['files'].items():
        if m.digest(RUN/name)!=h:raise ValueError('decision_evidence_changed')
    pending={r['id'] for r in targets if by[r['id']]['review_required']}
    if len(decisions)!=286 or set(dv)!=pending:raise ValueError('all_new_sources_must_be_qualified')
    complete=[]
    for parent,target in zip(parents,targets):
        d=by[target['id']];source=d['plain_english'];decision=None
        if d['review_required']:
            decision=dv[target['id']]
            if decision['status']!='approved_automated_pilot' or decision['draft_sha256']!=m.sha(source) or decision['target_sha256']!=parent['target_sha256']:raise ValueError('unapproved_or_unbound_decision')
            for patch in decision['applied_patches']:
                if not patch['before'] or source.count(patch['before'])!=patch['occurrences']:raise ValueError('patch_binding')
                source=source.replace(patch['before'],patch['after'])
            if source!=decision['final_source'] or m.sha(source)!=decision['source_sha256']:raise ValueError('patch_replay')
            if m.numbers(source)!=m.numbers(parent['target']) and not decision['numeric_difference_explanation']:raise ValueError('numeric_difference_not_adjudicated')
        row=copy.deepcopy(parent);row.update(id=parent['id']+'::gemini_track_v1',paragraph_id=parent['id']+'::gemini_track_v1',source=source,plain_english=source,source_sha256=m.sha(source))
        row['provenance']=dict(generator=m.MODEL,teacher_track='Gemini',matched_mixed_id=parent['id'],parent_mixed_sha256=m.digest(PARENT),
            previous_pair_provenance=parent['provenance'],input_origin=d['origin'],generation_evidence=d['generation_evidence'],
            acceptance='reused prior automated acceptance' if decision is None else 'separate meaning-only review and versioned minimal adjudication',
            final_decision_sha256=None if decision is None else m.digest(RUN/'final_decisions.private.jsonl'),independent_semantic_verification=False)
        row['quality']=dict(parent['quality'],text_pair_status='automated_pilot_approved',semantic_review=row['provenance']['acceptance'],independent_human_verification=False)
        if row['target']!=parent['target'] or row['target_sha256']!=parent['target_sha256']:raise ValueError('human_target_changed')
        complete.append(row)
    if len(complete)!=500 or len({r['parent_target_id'] for r in complete})!=500:raise ValueError('cohort_identity')
    if OUT.exists():raise ValueError('refuse_overwrite_Gemini_track')
    text=''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in complete)
    for size in [200,500]:
        folder=OUT/f'stage_{size:04d}';folder.mkdir(parents=True)
        subset=complete[:size]
        if Counter(r['style_group'] for r in subset)!=Counter({g:size//4 for g in m.GROUPS}):raise ValueError('balanced_nested_cohort')
        path=folder/'training.private.jsonl';path.write_text(text if size==500 else ''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in subset))
        summary=dict(protocol='gemini_track_v1',stage=size,pairs=size,groups=dict(Counter(r['style_group'] for r in subset)),papers=len({r['paper_id'] for r in subset}),teacher=m.MODEL,parent_mixed_sha256=m.digest(PARENT),human_targets_unchanged=True,training_calls=0,free_tier_only=True)
        m.write(folder/'summary.json',summary);m.write(folder/'manifest.json',dict(files={p.name:m.digest(p) for p in folder.iterdir() if p.is_file()},final_decisions_manifest_sha256=m.digest(RUN/'final_decisions_manifest.json')))
    if not (OUT/'stage_0500/training.private.jsonl').read_bytes().startswith((OUT/'stage_0200/training.private.jsonl').read_bytes()):raise ValueError('nested_prefix')
    print('Frozen matched Gemini-only 200/500 cohorts. No training submitted.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','check','generate','prepare-review','review','freeze']);a=p.parse_args()
    {'prepare':prepare,'check':check,'generate':lambda:call_jobs('generation'),'prepare-review':prepare_review,'review':lambda:call_jobs('verification'),'freeze':freeze}[a.action]()
