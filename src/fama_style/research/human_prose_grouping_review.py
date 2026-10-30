"""Evidence-bound review of the five-example explicit grouping diagnostic."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import statistics

from .author_control_replication import verify_files
from .author_comparison_corpus import file_sha, words
from .claims import sentences, tokens
from .signal import overlap
from .human_prose_grouping import CONDITIONS, groups
from .human_prose_review import schema_errors
from .human_prose_realization_review import validate_assessment as validate_claims
from fama_style.rhetoric.experiment import fresh, write_json, write_lines

PROTOCOL='human_prose_grouping_review_v1'


def parse_responses(package,responses):
    verify_files(package)
    jobs=json.loads((package/'private_key.json').read_text())
    if len(jobs)!=15 or len({j['job_id'] for j in jobs})!=15: raise ValueError('job_identity')
    if Counter(j['condition'] for j in jobs)!={c:5 for c in CONDITIONS}: raise ValueError('condition_coverage')
    if any(j['split']!='development' for j in jobs): raise ValueError('development_only')
    schema=json.loads((package/'output.schema.json').read_text())
    accepted={};hashes={}
    for batch in (1,2,3):
        path=responses/('responses_%02d.jsonl'%batch);hashes[path.name]=file_sha(path)
        expected={j['job_id'] for j in jobs if j['batch']==batch}
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():continue
            row=json.loads(line)
            if schema_errors(row,schema) or not row['output'].strip() or row['job_id'] not in expected:
                raise ValueError('response_schema_or_batch')
            if row['job_id'] in accepted: raise ValueError('duplicate_response')
            job=next(j for j in jobs if j['job_id']==row['job_id'])
            plain=job['condition']=='ordinary';status=row['plan_status'];note=row['plan_note']
            if plain!=(status=='not_requested'): raise ValueError('plan_status_condition_mismatch')
            if status in ('followed','not_requested') and note.strip(): raise ValueError('unexpected_plan_note')
            if status not in ('followed','not_requested') and not note.strip(): raise ValueError('missing_deviation_note')
            accepted[row['job_id']]=row
    if set(accepted)!={j['job_id'] for j in jobs}: raise ValueError('response_coverage')
    return jobs,accepted,hashes


def validate_assessment(a,text,review):
    validate_claims(a,text,review)
    if type(review['grouping_ambiguous']) is not bool: raise ValueError('ambiguity_boolean')
    if review['grouping_ambiguous'] and not review['grouping_note'].strip(): raise ValueError('missing_ambiguity_note')
    # Every assigned claim in every sentence must have a corresponding exact anchor.
    for group,sentence in zip(review['output_claim_groups'],sentences(text)):
        for cid in group:
            claim=next(c for c in review['claims'] if c['claim_id']==cid)
            if not any(e['quote'] in sentence for e in claim['output_evidence']):
                raise ValueError('unanchored_group_claim')


def group_status(observed,expected,ambiguous):
    if len(observed)!=len(expected):return 'not_matched'
    if ambiguous:return 'ambiguous_composite_claim'
    return 'matched' if observed==expected else 'not_matched'


def run(package,responses,audit,assessments,output):
    jobs,accepted,hashes=parse_responses(package,responses);verify_files(audit)
    records={r['paragraph_id']:r for r in map(json.loads,(audit/'audits.private.jsonl').read_text().splitlines())}
    spec=json.loads(assessments.read_text())
    if spec['protocol']!=PROTOCOL:raise ValueError('assessment_protocol')
    reviews={r['job_id']:r for r in spec['records']}
    if len(reviews)!=len(spec['records']) or set(reviews)!=set(accepted):raise ValueError('assessment_coverage')
    metrics=[]
    for job in jobs:
        row=records[job['paragraph_id']]
        if row['split']!='development':raise ValueError('development_only')
        a=row['annotation'];response=accepted[job['job_id']];text=response['output'];review=reviews[job['job_id']]
        validate_assessment(a,text,review)
        source_groups,target_groups=groups(a,'source_schema'),groups(a,'target_schema')
        if source_groups==target_groups:raise ValueError('identity_example')
        observed=review['output_claim_groups'];ambiguous=review['grouping_ambiguous']
        intended=source_groups if job['condition']=='source_group_plan' else target_groups
        metrics.append({**job,**response,'source_overlap':overlap(a['plain_english'],text),
            'reference_overlap_evaluation_only':overlap(a['original_paragraph'],text),
            'normalized_source_copy':tokens(a['plain_english'])==tokens(text),
            'output_sentence_count':len(sentences(text)),'output_words':len(words(text)),
            'longest_sentence_words':max(len(words(s)) for s in sentences(text)),
            'target_sentence_count_matches':len(sentences(text))==len(target_groups),
            'source_sentence_count_matches':len(sentences(text))==len(source_groups),
            'target_grouping_status':group_status(observed,target_groups,ambiguous),
            'source_grouping_status':group_status(observed,source_groups,ambiguous),
            'requested_grouping_status':None if job['condition']=='ordinary' else group_status(observed,intended,ambiguous),
            'assessment':review})
    summary={'protocol':PROTOCOL,'status':'exploratory_assistant_evidence_assessed',
        'accepted_outputs':15,'examples':5,'papers':len({j['paper_id'] for j in jobs}),
        'user_reported_model':'GPT-6','user_reported_thinking_effort':'High',
        'current_chat_arrangement':'not_reported','cross_chat_independence_established':False,
        'hidden_decoding_controlled':False,'model_identity_independently_verified':False,
        'assessment_method':spec['assessment_method'],'assessment_blinded':False,
        'semantic_equivalence_certified':False,'distinctive_author_style_established':False,
        'source_only_mapper_tested':False,'detector_performance_tested':False,
        'raw_responses_modified':False,'reserved_rows_used':0,'review_model_calls':0,
        'gpu_jobs_started':0,'training_called':False,'conditions':{},
        'self_reported_plan_statuses':dict(Counter(m['plan_status'] for m in metrics)),
        'all_grouping_changed_cases':True,'source_ambiguity_examples':len(spec['preexisting_ambiguity_ids'])}
    for condition in CONDITIONS:
        subset=[m for m in metrics if m['condition']==condition]
        summary['conditions'][condition]={
            'outputs':len(subset),'median_source_overlap':statistics.median(m['source_overlap'] for m in subset),
            'median_reference_overlap_evaluation_only':statistics.median(m['reference_overlap_evaluation_only'] for m in subset),
            'normalized_source_copies':sum(m['normalized_source_copy'] for m in subset),
            'target_sentence_count_matches':sum(m['target_sentence_count_matches'] for m in subset),
            'source_sentence_count_matches':sum(m['source_sentence_count_matches'] for m in subset),
            'target_grouping_statuses':dict(Counter(m['target_grouping_status'] for m in subset)),
            'source_grouping_statuses':dict(Counter(m['source_grouping_status'] for m in subset)),
            'requested_grouping_statuses':dict(Counter(m['requested_grouping_status'] for m in subset)) if condition!='ordinary' else None,
            'claim_instances':sum(len(m['assessment']['claims']) for m in subset),
            'claim_assessment_statuses':dict(Counter(c['status'] for m in subset for c in m['assessment']['claims']))}
    triplets=[]
    for pid in dict.fromkeys(j['paragraph_id'] for j in jobs):
        triplet={m['condition']:m for m in metrics if m['paragraph_id']==pid}
        plain=triplet['ordinary'];target=triplet['oracle_target_group_plan'];source=triplet['source_group_plan']
        triplets.append({'paragraph_id':pid,'paper_id':plain['paper_id'],
            'target_minus_ordinary_source_overlap':target['source_overlap']-plain['source_overlap'],
            'source_plan_minus_ordinary_source_overlap':source['source_overlap']-plain['source_overlap'],
            'target_ordinary_output_overlap':overlap(target['output'],plain['output']),
            'target_source_plan_output_overlap':overlap(target['output'],source['output']),
            'organization_and_expression_observations':spec['triplet_observations'][pid]})
    summary['paired']={'target_plan_higher_source_overlap_than_ordinary':sum(t['target_minus_ordinary_source_overlap']>0 for t in triplets),
        'source_plan_higher_source_overlap_than_ordinary':sum(t['source_plan_minus_ordinary_source_overlap']>0 for t in triplets)}
    conditions=summary['conditions'];target=conditions['oracle_target_group_plan'];source=conditions['source_group_plan']
    summary['prospective_mechanistic_screen_passed']=(target['requested_grouping_statuses'].get('matched',0)>=4
        and source['requested_grouping_statuses'].get('matched',0)>=4
        and target['target_grouping_statuses'].get('matched',0)>max(
            conditions['ordinary']['target_grouping_statuses'].get('matched',0),source['target_grouping_statuses'].get('matched',0)))
    summary['screen_interpretation']='Limited grouping controllability only; no automatic training or style-quality approval.'
    fresh(output);(output/'raw').mkdir()
    for name,digest in hashes.items():
        shutil.copyfile(responses/name,output/'raw'/name)
        if file_sha(output/'raw'/name)!=digest:raise ValueError('response_changed_during_import')
    shutil.copyfile(assessments,output/'assessments.private.json')
    if (package/'SESSION_NOTES.md').exists():shutil.copyfile(package/'SESSION_NOTES.md',output/'session_notes_as_received.md')
    write_json(output/'summary.json',summary);write_lines(output/'metrics.private.jsonl',metrics)
    write_json(output/'triplets.private.json',triplets)
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'code_sha256':file_sha(Path(__file__)),
        'claim_validator_sha256':file_sha(Path(__file__).with_name('human_prose_realization_review.py')),
        'package_manifest_sha256':file_sha(package/'manifest.json'),'audit_manifest_sha256':file_sha(audit/'manifest.json'),
        'assessment_sha256':file_sha(assessments),'response_sha256':hashes,
        'files':{str(p.relative_to(output)):file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('package','responses','audit','assessments','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.package,a.responses,a.audit,a.assessments,a.output),indent=2))


if __name__=='__main__':main()
