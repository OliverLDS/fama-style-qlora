"""Review the small human-prose oracle diagnostic; no generation or training."""
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
from .human_prose_review import occurrence_span
from fama_style.rhetoric.experiment import fresh, write_json, write_lines

PROTOCOL = 'human_prose_realization_review_v1'
STATUSES = {'no_clear_change_identified', 'minor_shift', 'scope_risk', 'scope_shift'}


def parse_responses(package, responses):
    """Fail closed on malformed, duplicate, misplaced, empty or missing replies."""
    verify_files(package)
    jobs = json.loads((package/'private_key.json').read_text())
    if len(jobs)!=16 or len({j['job_id'] for j in jobs})!=16:
        raise ValueError('expected_16_unique_jobs')
    accepted={};hashes={}
    for batch in (1,2):
        path=responses/('responses_%02d.jsonl'%batch)
        hashes[path.name]=file_sha(path)
        expected={j['job_id'] for j in jobs if j['batch']==batch}
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip(): continue
            row=json.loads(line)
            if (not isinstance(row,dict) or set(row)!={'job_id','output'}
                    or not isinstance(row['job_id'],str) or row['job_id'] not in expected
                    or not isinstance(row['output'],str) or not row['output'].strip()):
                raise ValueError('response_shape_id_or_empty')
            if row['job_id'] in accepted: raise ValueError('duplicate_response')
            accepted[row['job_id']]=row
    if set(accepted)!={j['job_id'] for j in jobs}: raise ValueError('response_coverage')
    return jobs,accepted,hashes


def validate_assessment(annotation, output, review):
    """Check evidence and complete label coverage, not the truth of judgments."""
    claims={c['claim_id'] for c in annotation['semantic_ledger']['claims']}
    entries=review['claims']
    if len(entries)!=len(claims) or {c['claim_id'] for c in entries}!=claims:
        raise ValueError('assessment_claim_coverage')
    units=sentences(output)
    groups=review['output_claim_groups']
    if len(groups)!=len(units) or set().union(*(set(g) for g in groups))!=claims:
        raise ValueError('assessment_sentence_coverage')
    if any(not g or len(set(g))!=len(g) or set(g)-claims for g in groups):
        raise ValueError('assessment_group_claims')
    for claim in entries:
        if claim['status'] not in STATUSES or not claim['note'].strip():
            raise ValueError('assessment_status_or_note')
        evidence=claim['output_evidence']
        if not evidence or any(occurrence_span(output,e) is None for e in evidence):
            raise ValueError('assessment_output_evidence')
        if not any(claim['claim_id'] in group and any(e['quote'] in sentence for e in evidence)
                   for group,sentence in zip(groups,units)):
            raise ValueError('assessment_evidence_outside_group')
    return True


def run(package, responses, audit, assessments, output):
    jobs,accepted,hashes=parse_responses(package,responses)
    verify_files(audit)
    records={r['paragraph_id']:r for r in map(json.loads,(audit/'audits.private.jsonl').read_text().splitlines())}
    spec=json.loads(assessments.read_text())
    if spec['protocol']!=PROTOCOL: raise ValueError('assessment_protocol')
    reviews={r['job_id']:r for r in spec['records']}
    if len(reviews)!=len(spec['records']) or set(reviews)!=set(accepted): raise ValueError('assessment_output_coverage')
    metrics=[]
    for job in jobs:
        record=records[job['paragraph_id']]
        if record['split']!='development': raise ValueError('development_only')
        a=record['annotation'];response=accepted[job['job_id']];text=response['output'];review=reviews[job['job_id']]
        validate_assessment(a,text,review)
        target_groups=[s['claim_ids'] for s in a['target_schema']['sentences']]
        metrics.append({**job,**response,'source_overlap':overlap(a['plain_english'],text),
            'reference_overlap_evaluation_only':overlap(a['original_paragraph'],text),
            'normalized_source_copy':tokens(a['plain_english'])==tokens(text),
            'output_sentences':len(sentences(text)),'output_words':len(words(text)),
            'source_target_claim_grouping_differs':[s['claim_ids'] for s in a['source_schema']['sentences']]!=target_groups,
            'target_sentences':len(target_groups),'sentence_count_matches_target':len(sentences(text))==len(target_groups),
            'claim_grouping_matches_target':review['output_claim_groups']==target_groups,'assessment':review})
    pairs=[]
    for paragraph in dict.fromkeys(j['paragraph_id'] for j in jobs):
        pair={r['condition']:r for r in metrics if r['paragraph_id']==paragraph}
        a,b=pair['ordinary'],pair['oracle_presentation']
        pairs.append({'paragraph_id':paragraph,'paper_id':a['paper_id'],
            'paired_output_overlap':overlap(a['output'],b['output']),
            'guided_minus_ordinary_source_overlap':b['source_overlap']-a['source_overlap'],
            'guided_minus_ordinary_reference_overlap_evaluation_only':b['reference_overlap_evaluation_only']-a['reference_overlap_evaluation_only'],
            'source_target_claim_grouping_differs':a['source_target_claim_grouping_differs'],
            'same_sentence_count':a['output_sentences']==b['output_sentences'],
            'same_claim_grouping':a['assessment']['output_claim_groups']==b['assessment']['output_claim_groups'],
            'organization_and_expression_observation':spec['pair_observations'][paragraph]})
    summary={'protocol':PROTOCOL,'status':'exploratory_assistant_evidence_assessed',
        'accepted_outputs':len(accepted),'examples':len(pairs),'papers':len({j['paper_id'] for j in jobs}),
        'user_reported_model':'GPT-6','user_reported_thinking_effort':'High',
        'current_chat_arrangement':'not_reported','cross_chat_independence_established':False,
        'hidden_model_snapshot_known':False,'hidden_decoding_controlled':False,
        'assessment_method':spec['assessment_method'],'assessment_blinded':False,
        'semantic_equivalence_certified':False,'source_only_mapper_tested':False,
        'distinctive_author_style_established':False,'detector_performance_tested':False,
        'generation_performed_by_user':True,'review_model_calls':0,'gpu_jobs_started':0,'training_called':False,
        'reserved_rows_used':0,'conditions':{},'paired':{
            'source_target_grouping_differs_examples':sum(p['source_target_claim_grouping_differs'] for p in pairs),
            'same_output_sentence_count':sum(p['same_sentence_count'] for p in pairs),
            'same_claim_grouping':sum(p['same_claim_grouping'] for p in pairs),
            'median_paired_output_overlap':statistics.median(p['paired_output_overlap'] for p in pairs),
            'guided_lower_source_overlap':sum(p['guided_minus_ordinary_source_overlap']<0 for p in pairs),
            'guided_higher_reference_overlap_evaluation_only':sum(p['guided_minus_ordinary_reference_overlap_evaluation_only']>0 for p in pairs)},
        'known_input_epistemic_shift_examples':len(spec['known_input_epistemic_shift_ids']),'raw_responses_modified':False}
    for condition in ('ordinary','oracle_presentation'):
        group=[r for r in metrics if r['condition']==condition]
        summary['conditions'][condition]={'outputs':len(group),
            'median_source_overlap':statistics.median(r['source_overlap'] for r in group),
            'median_reference_overlap_evaluation_only':statistics.median(r['reference_overlap_evaluation_only'] for r in group),
            'normalized_source_copies':sum(r['normalized_source_copy'] for r in group),
            'sentence_count_matches_target':sum(r['sentence_count_matches_target'] for r in group),
            'claim_grouping_matches_target':sum(r['claim_grouping_matches_target'] for r in group),
            'reviewed_claim_instances':sum(len(r['assessment']['claims']) for r in group),
            'claim_assessment_statuses':dict(Counter(c['status'] for r in group for c in r['assessment']['claims']))}
    fresh(output);(output/'raw').mkdir()
    for name,digest in hashes.items():
        shutil.copyfile(responses/name,output/'raw'/name)
        if file_sha(output/'raw'/name)!=digest: raise ValueError('response_changed_during_import')
    shutil.copyfile(assessments,output/'assessments.private.json')
    write_json(output/'summary.json',summary);write_json(output/'pairs.private.json',pairs)
    write_lines(output/'metrics.private.jsonl',metrics)
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'code_sha256':file_sha(Path(__file__)),
        'package_manifest_sha256':file_sha(package/'manifest.json'),'audit_manifest_sha256':file_sha(audit/'manifest.json'),
        'assessment_sha256':file_sha(assessments),'response_sha256':hashes,
        'files':{str(p.relative_to(output)):file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('package','responses','audit','assessments','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.package,a.responses,a.audit,a.assessments,a.output),indent=2))


if __name__=='__main__':main()
