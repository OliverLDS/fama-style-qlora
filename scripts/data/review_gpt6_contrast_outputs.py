"""Assess brief-only rewrites, replay local repairs, and stage a versioned 500.

No training, detector, provider, development, or reserved-test calls/reads.
"""
import argparse
from collections import Counter
import copy
import importlib.util
import json
from pathlib import Path
import statistics

SPEC=importlib.util.spec_from_file_location('brief_review',Path(__file__).with_name('review_gpt6_contrast_briefs.py'))
b=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(b)
m=b.m
OUT=m.OUT/'rewrite_quality_v1'


def validate_response(row,expected):
    if not isinstance(row,dict) or set(row)!=set(m.REWRITE_SCHEMA['required']):raise ValueError('rewrite_fields')
    if row['id']!=expected['id'] or row['target_sha256']!=expected['target_sha256']:raise ValueError('rewrite_identity')
    if not isinstance(row['plain_english'],str) or not row['plain_english'].strip():raise ValueError('empty_rewrite')
    if row['rewrite_status'] not in {'rewritten','limited_rewrite_room','source_problem'}:raise ValueError('rewrite_status')
    if not isinstance(row['review_notes'],list) or any(not isinstance(x,str) for x in row['review_notes']):raise ValueError('rewrite_notes')
    if row['rewrite_status']!='rewritten' and not row['review_notes']:raise ValueError('unexplained_rewrite_status')


def load():
    b.check();selected={r['id']:r for r in m.lines(m.OUT/'selected_targets.private.jsonl')}
    rows=[];files={}
    for job in m.read(b.STAGE2/'jobs.json'):
        name='responses/stage_2_rewrites/'+job['response'];raw=m.lines(m.OUT/name)
        if len(raw)!=len(job['ids']):raise ValueError('rewrite_batch_count')
        files[name]=m.digest(m.OUT/name)
        for row,rid in zip(raw,job['ids']):
            validate_response(row,selected[rid]);rows.append(row)
    if len(rows)!=len(selected) or len({r['id'] for r in rows})!=len(selected):raise ValueError('rewrite_coverage')
    return rows,files,selected


def apply_patches(source,patches):
    result=source
    for p in patches:
        if not isinstance(p['before'],str) or not p['before'] or result.count(p['before'])!=1 or not isinstance(p['after'],str) or not p.get('reason'):
            raise ValueError('rewrite_patch_binding')
        if p['before']==source or len(p['before'].split())>max(35,.6*len(source.split())):raise ValueError('rewrite_patch_too_broad')
        result=result.replace(p['before'],p['after'],1)
    return result


def prepare():
    if (OUT/'manifest.json').exists():raise ValueError('refuse_overwrite_rewrite_review')
    rows,files,selected=load();decisions=m.lines(OUT/'adjudications.private.jsonl');by={r['id']:r for r in decisions}
    if len(by)!=len(decisions) or set(by)!={r['id'] for r in rows}:raise ValueError('rewrite_decision_coverage')
    assessed=[];effective={};holds=[]
    for row in rows:
        rid=row['id'];d=by[rid];old=selected[rid]
        if d['raw_source_sha256']!=m.sha(row['plain_english']) or d['target_sha256']!=row['target_sha256'] or not d['reason']:raise ValueError('rewrite_decision_text_binding')
        if d['decision'] not in {'accept','repair','source_concern'} or bool(d['patches'])!=(d['decision']=='repair'):raise ValueError('rewrite_decision_schema')
        final=apply_patches(row['plain_english'],d['patches'])
        if d['decision']=='source_concern':holds.append(rid)
        effective[rid]=final
        assessed.append(dict(id=rid,target_sha256=row['target_sha256'],raw_response=row,raw_source_sha256=m.sha(row['plain_english']),
            final_source=final,source_sha256=m.sha(final),old_source_sha256=old['source_sha256'],adjudication=d,
            old_metrics=old['metrics'],raw_metrics=m.contrast(row['plain_english'],old['target']),
            final_metrics=m.contrast(final,old['target']),independent_semantic_verification=False))
    (OUT/'assessed_rewrites.private.jsonl').write_text(m.jsonl(assessed))
    parent=m.lines(m.PARENT/'candidates_0500.private.jsonl');versions=[]
    inherited={'P250','P264','P291','P458','P484'}
    for i,prior in enumerate(parent,1):
        rid=f'P{i:03d}';row=copy.deepcopy(prior);use_new=rid in effective and rid not in holds
        source=effective[rid] if use_new else prior['source']
        row.update(id=rid,paragraph_id=rid,source=source,plain_english=source,source_sha256=m.sha(source),training_ready=False)
        pp=prior['provenance']
        row['provenance']=dict(teacher_track='GPT-6',matched_mixed_id=pp['matched_mixed_id'],canonical_pdf_sha256=pp['canonical_pdf_sha256'],
            parent_dataset='research_runs/data_scaling_v1/gpt6_track_0500_v1/response_quality_v1/candidates_0500.private.jsonl',
            parent_dataset_sha256=m.digest(m.PARENT/'candidates_0500.private.jsonl'),previous_record_id=prior['id'],previous_source_sha256=prior['source_sha256'],
            input_origin='brief_reconstruction_v1' if use_new else 'prior_input_retained',
            reconstruction_id=rid if rid in effective else None,
            assessment_file='assessed_rewrites.private.jsonl' if rid in effective else None,
            actual_model_effort_independently_verified=False,
            source_repair_author='Codex' if use_new and by[rid]['patches'] else None,independent_semantic_verification=False)
        status='Candidate · fresh brief reconstruction' if use_new else 'Candidate · prior GPT-6 input retained'
        if rid in inherited:status='Target cleanup pending · prior input retained'
        if rid in holds:status='Source concern pending · prior input retained'
        row['quality']=dict(prior['quality'],text_pair_status=status,release_status='candidate_only_not_training_ready',
                            target_review_pending=rid in inherited or rid in holds)
        if row['target']!=prior['target'] or row['target_sha256']!=prior['target_sha256']:raise ValueError('human_target_changed')
        versions.append(row)
    if len(versions)!=500 or len({r['parent_target_id'] for r in versions})!=500:raise ValueError('versioned_cohort_identity')
    (OUT/'candidates_0500.private.jsonl').write_text(m.jsonl(versions))
    (OUT/'candidates_0200.private.jsonl').write_text(m.jsonl(versions[:200]))
    summary=dict(protocol='gpt6_brief_reconstruction_quality_v1',responses=175,response_files=len(files),
        decision_counts=dict(Counter(d['decision'] for d in decisions)),
        repaired_ids=[d['id'] for d in decisions if d['patches']],patches=sum(len(d['patches']) for d in decisions),
        source_concern_ids=holds,new_inputs_staged=len(effective)-len(holds),prior_inputs_retained=500-len(effective)+len(holds),
        source_versions_discarded=0,raw_status_counts=dict(Counter(r['rewrite_status'] for r in rows)),
        lower_word_similarity=sum(r['raw_metrics']['word_similarity']<r['old_metrics']['word_similarity'] for r in assessed),
        old_median_word_similarity=statistics.median(r['old_metrics']['word_similarity'] for r in assessed),
        new_median_word_similarity=statistics.median(r['raw_metrics']['word_similarity'] for r in assessed),
        old_median_ordinary_fivegram_coverage=statistics.median(r['old_metrics']['ordinary_shared_fivegram_coverage'] for r in assessed),
        new_median_ordinary_fivegram_coverage=statistics.median(r['raw_metrics']['ordinary_shared_fivegram_coverage'] for r in assessed),
        old_median_longest_ordinary_span=statistics.median(r['old_metrics']['longest_ordinary_words'] for r in assessed),
        new_median_longest_ordinary_span=statistics.median(r['raw_metrics']['longest_ordinary_words'] for r in assessed),
        old_median_word_length_ratio=statistics.median(r['old_metrics']['word_length_ratio'] for r in assessed),
        new_median_word_length_ratio=statistics.median(r['raw_metrics']['word_length_ratio'] for r in assessed),
        still_flagged_by_contrast_rules=sum(m.classify(r['final_metrics'])[0]=='regenerate_brief_first' for r in assessed),
        five_inherited_target_followups=sorted(inherited),all_target_followups=sorted(inherited|set(holds)),
        matched_human_targets_unchanged=True,nested_200_prefix=True,training_ready_release=False,
        provider_calls=0,training_calls=0,development_examples_read=0,reserved_test_examples_read=0,
        assessment_limitations=['Automated reading of the new paragraphs, with targeted reference checks; not independent human/NLI certification.',
            'Ordinary-word exemptions and sentence alignments are surface proxies, not validated author-style or grammar scores.',
            'Minor compression, interpretive additions and emphasis changes are accepted; the audit is not exhaustive claim-by-claim equivalence.',
            'No student-model or Pangram improvement has been measured on these replacements.'])
    m.dump(OUT/'summary.json',summary)
    m.dump(OUT/'manifest.json',dict(raw_response_files=files,parent_brief_manifest_sha256=m.digest(b.AUDIT/'manifest.json'),
        builder_sha256=m.digest(Path(__file__)),parent_candidates_sha256=m.digest(m.PARENT/'candidates_0500.private.jsonl'),
        files={str(f.relative_to(OUT)):m.digest(f) for f in sorted(OUT.iterdir()) if f.is_file()}))
    print(json.dumps(summary,indent=2))


def check():
    rows,files,selected=load();manifest=m.read(OUT/'manifest.json')
    if files!=manifest['raw_response_files'] or m.digest(Path(__file__))!=manifest['builder_sha256'] or m.digest(b.AUDIT/'manifest.json')!=manifest['parent_brief_manifest_sha256'] or m.digest(m.PARENT/'candidates_0500.private.jsonl')!=manifest['parent_candidates_sha256']:raise ValueError('rewrite_dependency_changed')
    for name,h in manifest['files'].items():
        if m.digest(OUT/name)!=h:raise ValueError('rewrite_review_changed')
    assessed=m.lines(OUT/'assessed_rewrites.private.jsonl');new={r['id']:r for r in assessed}
    if len(new)!=175:raise ValueError('assessed_coverage')
    for row in assessed:
        if apply_patches(row['raw_response']['plain_english'],row['adjudication']['patches'])!=row['final_source'] or m.sha(row['final_source'])!=row['source_sha256']:raise ValueError('rewrite_patch_replay')
    current=m.lines(OUT/'candidates_0500.private.jsonl');parent=m.lines(m.PARENT/'candidates_0500.private.jsonl')
    for i,(old,row) in enumerate(zip(parent,current),1):
        rid=f'P{i:03d}';replacement=new.get(rid)
        expected=replacement['final_source'] if replacement and replacement['adjudication']['decision']!='source_concern' else old['source']
        if row['target']!=old['target'] or row['target_sha256']!=old['target_sha256'] or row['source']!=expected or row['source_sha256']!=m.sha(expected) or row['training_ready']:raise ValueError('candidate_version_binding')
    if len(current)!=500 or len({r['parent_target_id'] for r in current})!=500:raise ValueError('candidate_count')
    if not (OUT/'candidates_0500.private.jsonl').read_bytes().startswith((OUT/'candidates_0200.private.jsonl').read_bytes()):raise ValueError('nested_prefix')
    return dict(rewrites=175,candidates=500,training_ready=False,raw_sources_unchanged=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','check']);args=parser.parse_args()
    if args.action=='prepare':prepare()
    else:print(json.dumps(check(),indent=2))
