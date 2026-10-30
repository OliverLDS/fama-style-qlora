"""Preserve six expression-check replies and bind exploratory assessments.

Mechanical evidence checks cannot certify semantic or syntactic judgments.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from . import claim_bound_expression as experiment
from . import source_repair_review as shared
from fama_style.rhetoric import strong_realizer_check as transport
from fama_style.rhetoric.experiment import fresh, write_json, write_lines
from fama_style.research.signal import overlap
from fama_style.research.claims import tokens

PROTOCOL = 'claim_bound_expression_review_v1'


def validate_assessment(card, output, assessment):
    shared.validate_assessment(card,output,assessment)
    op=assessment['operation_assessment']; expected=card['expression_operation']
    if op['control_id']!=expected['control_id'] or op['claim_refs']!=expected['claim_refs']: raise ValueError('operation_identity')
    for field in ('form_present','within_fixed_grouping'):
        if type(op[field]) is not bool: raise ValueError('operation_boolean')
    if not op['output_evidence']: raise ValueError('missing_operation_evidence')
    owner_refs=set(expected['claim_refs']); anchors=[c['output_span'] for c in assessment['claims'] if c['claim_id'] in owner_refs]
    for span in op['output_evidence']:
        if not 0<=span['start']<span['end']<=len(output) or output[span['start']:span['end']]!=span['quote']: raise ValueError('operation_output_anchor')
        if not any(a['start']<=span['start']<span['end']<=a['end'] for a in anchors): raise ValueError('operation_outside_claims')
    fixed=[s['claim_refs'] for p in card['plan'] for s in p['output_slots']]
    if op['within_fixed_grouping']!=(assessment['output_claim_groups']==fixed): raise ValueError('operation_grouping_judgment')
    return True


def run(package,responses,output,assessments=None):
    key=experiment.check_package(package);accepted,coverage=transport.parse_responses(key,responses)
    cards={c['id']:c for c in key['rows']};jobs={j['job_id']:j for j in key['jobs']};reviews={}
    if assessments:
        data=json.loads(assessments.read_text())
        if data['protocol']!=PROTOCOL: raise ValueError('assessment_protocol')
        reviews={r['job_id']:r for r in data['records']}
        if len(reviews)!=len(data['records']) or set(reviews)!={r['job_id'] for r in accepted}: raise ValueError('assessment_coverage')
    metrics=[]
    for r in accepted:
        j=jobs[r['job_id']];card=cards[j['source_id']];source=card['record']['source'];out=r['output'];reference=card['record']['reference']
        units=experiment.parent.parent.grouping.output_units(out)
        a=reviews.get(r['job_id'])
        if a:validate_assessment(card,out,a)
        metrics.append({**r,'paper_key':card['paper_key'],'source_id':card['id'],'condition':j['condition'],
            'source_sha256':card['source_sha256'],'output_sha256':experiment.parent.base.sha(out),
            'source_overlap':overlap(source,out),'reference_overlap_evaluation_only':overlap(reference,out),
            'token_edit_distance_from_source':shared.token_edit_distance(source,out),
            'normalized_source_copy':tokens(source)==tokens(out),
            'output_sentence_count':len(units['units']) if not units['ambiguous'] else None,
            'assessment':a})
    summary={'protocol':PROTOCOL,'status':'evidence_assessed' if reviews else 'import_only_quality_pending',
        'expected_outputs':6,'accepted_outputs':len(accepted),'complete':coverage['complete'],
        'user_reported_model':'GPT-6','user_reported_thinking_effort':'High','model_identity_independently_verified':False,
        'chat_arrangement':'not_reported','hidden_decoding_controlled':False,
        'assessment_method':'In-session assistant evidence review, not independent verification.' if reviews else None,
        'claim_instances_reviewed':sum(len(a['claims']) for a in reviews.values()),
        'claim_review_statuses':dict(Counter(c['status'] for a in reviews.values() for c in a['claims'])),
        'normalized_copies':sum(m['normalized_source_copy'] for m in metrics),'per_paper':[],
        'distinctive_author_style_verified':False,'model_calls_by_review':0,'gpu_jobs_started':0,'training_called':False,
        'validation_rows_read':0,'reserved_test_rows_read':0}
    for card in key['rows']:
        group={m['condition']:m for m in metrics if m['paper_key']==card['paper_key']}
        result={'paper_key':card['paper_key'],'control_id':card['expression_operation']['control_id'],'conditions':{}}
        for condition,m in group.items():
            result['conditions'][condition]={'source_overlap':m['source_overlap'],
                'reference_overlap_evaluation_only':m['reference_overlap_evaluation_only'],
                'token_edit_distance_from_source':m['token_edit_distance_from_source'],
                'output_sentence_count':m['output_sentence_count'],
                'operation_assessment':({k:m['assessment']['operation_assessment'][k] for k in ('form_present','within_fixed_grouping')} if m['assessment'] else None)}
        if all(c in group for c in ('grouping_control','selective_expression')):
            a,b=group['grouping_control'],group['selective_expression']
            result['primary_contrast']={'source_overlap_delta':b['source_overlap']-a['source_overlap'],
                'control_guided_overlap':overlap(a['output'],b['output']),
                'control_guided_token_edit_distance':shared.token_edit_distance(a['output'],b['output']),
                'reference_overlap_delta_evaluation_only':b['reference_overlap_evaluation_only']-a['reference_overlap_evaluation_only']}
        summary['per_paper'].append(result)
    fresh(output);(output/'raw').mkdir()
    for name,digest in coverage['response_file_hashes'].items():
        shutil.copyfile(responses/name,output/'raw'/name)
        if experiment.binding.file_sha(output/'raw'/name)!=digest: raise ValueError('raw_response_changed')
    if (responses/'notes.txt').exists():shutil.copyfile(responses/'notes.txt',output/'raw/notes.txt')
    if assessments:shutil.copyfile(assessments,output/'assessments.private.json')
    write_json(output/'coverage.json',coverage);write_lines(output/'metrics.private.jsonl',metrics);write_json(output/'summary.json',summary)
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'code_sha256':experiment.binding.file_sha(Path(__file__)),
        'shared_assessment_validator_sha256':experiment.binding.file_sha(Path(shared.__file__)),
        'package_manifest_sha256':experiment.binding.file_sha(package/'manifest.json'),
        'response_hashes':coverage['response_file_hashes'],'assessment_sha256':experiment.binding.file_sha(assessments) if assessments else None,
        'files':{str(p.relative_to(output)):experiment.binding.file_sha(p) for p in output.rglob('*') if p.is_file()}})
    print('Preserved '+str(len(accepted))+'/6 responses; '+summary['status']+'. No model calls or training.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True);p.add_argument('--responses',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--assessments',type=Path)
    a=p.parse_args();run(a.package,a.responses,a.output_dir,a.assessments)


if __name__=='__main__':main()
