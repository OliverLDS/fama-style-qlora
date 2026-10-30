"""CPU import and evidence-bound assessment of the frozen source-repair check.

Span checks validate assessment provenance, not the correctness of judgments.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import statistics

from . import source_repair_comparison as experiment
from fama_style.rhetoric import strong_realizer_check as transport
from fama_style.rhetoric.experiment import fresh, write_json, write_lines
from fama_style.research.signal import overlap
from fama_style.research.claims import tokens

PROTOCOL = 'source_repair_review_v1'


def token_edit_distance(a, b):
    a,b=tokens(a),tokens(b); previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1): current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]


def validate_assessment(card, output, review):
    if review['source_sha256'] != experiment.base.sha(card['record']['source']) or review['output_sha256'] != experiment.base.sha(output):
        raise ValueError('assessment_text_identity_changed')
    claims={c['claim_id']:c for c in card['evidence']['claim_index']}
    if Counter(c['claim_id'] for c in review['claims']) != Counter(claims.keys()): raise ValueError('assessment_claim_coverage')
    for claim in review['claims']:
        cid=claim['claim_id']
        if claim['status'] not in {'retained','minor_relation_weakening','possible_force_shift'}: raise ValueError('unknown_review_status')
        if claim['source_span'] not in claims[cid]['source_spans']: raise ValueError('assessment_source_anchor')
        span=claim['output_span']
        if not 0<=span['start']<span['end']<=len(output) or output[span['start']:span['end']] != span['quote']: raise ValueError('assessment_output_anchor')
    groups=review['output_claim_groups']; flat=[c for g in groups for c in g]
    if Counter(flat)!=Counter(claims.keys()): raise ValueError('review_group_coverage')
    units=experiment.parent.grouping.output_units(output)
    if units['ambiguous'] or len(groups)!=len(units['units']): raise ValueError('review_group_unit_count')
    for group,unit in zip(groups,units['units']):
        for cid in group:
            if next(c['output_span']['quote'] for c in review['claims'] if c['claim_id']==cid) not in unit: raise ValueError('review_group_evidence')
    if review['nonpacking_style_gain'] not in {'none','small','clear'}: raise ValueError('style_grade')
    return True


def run(package, responses, output, assessments=None):
    key=experiment.check_package(package); accepted,coverage=transport.parse_responses(key,responses)
    cards={c['id']:c for c in key['rows']}; jobs={j['job_id']:j for j in key['jobs']}
    reviewed={}
    if assessments:
        spec=json.loads(assessments.read_text())
        if spec['protocol']!=PROTOCOL: raise ValueError('review_protocol')
        reviewed={r['job_id']:r for r in spec['records']}
        if len(reviewed)!=len(spec['records']) or set(reviewed)!={r['job_id'] for r in accepted}: raise ValueError('review_output_coverage')
    metrics=[]
    for response in accepted:
        job=jobs[response['job_id']]; card=cards[job['source_id']]; text=response['output']
        pair={c['input_variant']:c for c in key['rows'] if c['paper_key']==card['paper_key']}
        source=card['record']['source']; reference=card['record']['reference']
        units=experiment.parent.grouping.output_units(text)
        metric={**response,**{k:job[k] for k in ('source_id','paper_key','input_variant','mode','condition')},
            'source_sha256':card['source_sha256'],'output_sha256':experiment.base.sha(text),
            'source_overlap':overlap(source,text),'reference_overlap':overlap(reference,text),
            'input_reference_overlap':overlap(source,reference),
            'reference_overlap_gain_over_input':overlap(reference,text)-overlap(source,reference),
            'overlap_with_each_input':{variant:overlap(c['record']['source'],text) for variant,c in pair.items()},
            'token_edit_distance_with_each_input':{variant:token_edit_distance(c['record']['source'],text) for variant,c in pair.items()},
            'normalized_source_copy':tokens(source)==tokens(text),
            'output_sentences':len(units['units']) if not units['ambiguous'] else None,
            'desired_sentences_evaluation_only':sum(len(e['output_slots']) for e in card['plan']),
            'grouping_matches_evidence_review':None,'semantic_assessment':None}
        if response['job_id'] in reviewed:
            review=reviewed[response['job_id']]; validate_assessment(card,text,review)
            desired=[slot['claim_refs'] for op in card['plan'] for slot in op['output_slots']]
            metric['grouping_matches_evidence_review']=review['output_claim_groups']==desired
            metric['semantic_assessment']=review
        metrics.append(metric)
    summary={'protocol':PROTOCOL,'status':'evidence_assessed' if reviewed else 'import_only_quality_pending',
        'complete':coverage['complete'],'expected_outputs':16,'accepted_outputs':len(accepted),
        'user_reported_model':'GPT-6','user_reported_thinking_effort':'High','model_identity_independently_verified':False,
        'conversation_arrangement':'not_reported','hidden_decoding_controlled':False,'conditions':{},
        'assessment_method':'In-session assistant evidence review; not independent model verification.' if reviewed else None,
        'model_calls_by_importer':0,'gpu_jobs_started':0,'training_called':False,'validation_rows_read':0,'reserved_test_rows_read':0,
        'distinctive_author_style_verified':False}
    for variant,mode in experiment.CONDITIONS:
        condition=variant+'_'+mode; group=[m for m in metrics if m['condition']==condition]
        def mean(field): return statistics.mean(m[field] for m in group) if group else None
        summary['conditions'][condition]={'accepted_outputs':len(group),'mean_source_overlap':mean('source_overlap'),
            'mean_reference_overlap':mean('reference_overlap'),'mean_input_reference_overlap':mean('input_reference_overlap'),
            'mean_reference_overlap_gain_over_input':mean('reference_overlap_gain_over_input'),
            'normalized_copies':sum(m['normalized_source_copy'] for m in group),
            'overlap_at_least_0_9':sum(m['source_overlap']>=.9 for m in group),
            'sentence_count_matches':sum(m['output_sentences']==m['desired_sentences_evaluation_only'] for m in group),
            'reviewed_grouping_matches':sum(m['grouping_matches_evidence_review'] is True for m in group) if reviewed else None,
            'claim_review_statuses':dict(Counter(c['status'] for m in group for c in (m['semantic_assessment']['claims'] if m['semantic_assessment'] else [])))}
    comparisons=[]
    for paper in sorted({m['paper_key'] for m in metrics}):
        group={m['condition']:m for m in metrics if m['paper_key']==paper}
        if len(group)!=4: continue
        contrasts={}
        for mode in ('ordinary','planned'):
            a,b=group['original_'+mode],group['repaired_'+mode]
            contrasts[mode]={'reference_overlap_delta':b['reference_overlap']-a['reference_overlap'],
                'input_reference_overlap_delta':b['input_reference_overlap']-a['input_reference_overlap'],
                'reference_gain_above_input_delta':b['reference_overlap_gain_over_input']-a['reference_overlap_gain_over_input'],
                'own_source_overlap_delta':b['source_overlap']-a['source_overlap']}
        comparisons.append({'paper_key':paper,'repair_contrasts':contrasts,
            'qualification':'context_bound' if paper=='NL2003' else 'recognition_correction_gap' if paper=='ME1997' else 'bounded_diagnostic'})
    summary['per_paper']=comparisons
    fresh(output);(output/'raw').mkdir()
    for name,digest in coverage['response_file_hashes'].items():
        shutil.copyfile(responses/name,output/'raw'/name)
        if experiment.file_sha(output/'raw'/name)!=digest: raise ValueError('response_changed_during_import')
    if (responses/'notes.txt').exists():shutil.copyfile(responses/'notes.txt',output/'raw/notes.txt')
    if assessments:shutil.copyfile(assessments,output/'assessments.private.json')
    write_json(output/'coverage.json',coverage);write_lines(output/'metrics.private.jsonl',metrics);write_json(output/'summary.json',summary)
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'code_sha256':experiment.file_sha(Path(__file__)),
        'package_manifest_sha256':experiment.file_sha(package/'manifest.json'),
        'response_hashes':coverage['response_file_hashes'],'assessment_sha256':experiment.file_sha(assessments) if assessments else None,
        'files':{str(p.relative_to(output)):experiment.file_sha(p) for p in output.rglob('*') if p.is_file()}})
    print('Imported '+str(len(accepted))+'/16 outputs; raw files preserved. Assessment: '+summary['status'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,required=True);parser.add_argument('--responses',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--assessments',type=Path)
    args=parser.parse_args();run(args.package,args.responses,args.output_dir,args.assessments)


if __name__=='__main__': main()
