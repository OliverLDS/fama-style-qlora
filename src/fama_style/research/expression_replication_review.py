"""Import four-paper replication outputs and evidence-bound exploratory judgments."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from . import expression_replication as experiment
from . import claim_bound_expression_review as shared
from fama_style.rhetoric import strong_realizer_check as transport
from fama_style.rhetoric.experiment import fresh, write_json, write_lines
from fama_style.research.signal import overlap
from fama_style.research.claims import tokens

PROTOCOL = 'expression_replication_review_v1'


def validate_assessment(card, output, assessment):
    shared.validate_assessment(card,output,assessment)
    obs=assessment['form_observations'];control=card['expression_operation']['control_id']
    keys=({'focus_as_subject','focus_initial'} if control=='FOREGROUND_SUPPORTED_RELATION_OR_FUNCTION'
          else {'finite_predicate','specified_actor_as_subject'})
    if set(obs)!=keys or any(type(v) is not bool for v in obs.values()):raise ValueError('form_observations')
    if assessment['operation_assessment']['form_present']!=all(obs.values()):raise ValueError('strict_form_observation_mismatch')
    if assessment['incremental_style_benefit'] not in {'not_primary_contrast','none_clear','small','clear'}:raise ValueError('incremental_benefit_label')
    return True


def run(package,responses,output,assessments=None):
    key=experiment.check_package(package);accepted,coverage=transport.parse_responses(key,responses)
    cards={c['id']:c for c in key['rows']};jobs={j['job_id']:j for j in key['jobs']};reviews={}
    if assessments:
        spec=json.loads(assessments.read_text())
        if spec['protocol']!=PROTOCOL:raise ValueError('assessment_protocol')
        reviews={a['job_id']:a for a in spec['records']}
        if len(reviews)!=len(spec['records']) or set(reviews)!={r['job_id'] for r in accepted}:raise ValueError('assessment_coverage')
    metrics=[]
    for response in accepted:
        job=jobs[response['job_id']];card=cards[job['source_id']];source=card['record']['source'];out=response['output']
        units=experiment.annotations.grouping.output_units(out);a=reviews.get(response['job_id'])
        if a:validate_assessment(card,out,a)
        metrics.append({**response,'source_id':card['id'],'paper_key':card['paper_key'],'condition':job['condition'],
            'control_id':card['expression_operation']['control_id'],'source_sha256':card['source_sha256'],
            'output_sha256':experiment.annotations.sha(out),'source_overlap':overlap(source,out),
            'reference_overlap_evaluation_only':overlap(card['record']['reference'],out),
            'source_token_edits':shared.shared.token_edit_distance(source,out),'normalized_source_copy':tokens(source)==tokens(out),
            'output_sentences':len(units['units']) if not units['ambiguous'] else None,'assessment':a})
    summary={'protocol':PROTOCOL,'status':'evidence_assessed' if reviews else 'import_only_quality_pending',
        'expected_outputs':12,'accepted_outputs':len(accepted),'complete':coverage['complete'],
        'user_reported_model':'GPT-6','user_reported_thinking_effort':'High','model_identity_independently_verified':False,
        'chat_arrangement':'not_reported','hidden_decoding_controlled':False,
        'assessment_method':'In-session assistant evidence review; not independent or blinded verification.' if reviews else None,
        'claim_instances_reviewed':sum(len(a['claims']) for a in reviews.values()),
        'claim_review_statuses':dict(Counter(c['status'] for a in reviews.values() for c in a['claims'])),
        'normalized_copies':sum(m['normalized_source_copy'] for m in metrics),'per_paper':[],
        'per_operation':{},'distinctive_author_style_verified':False,'model_calls_by_review':0,'gpu_jobs_started':0,
        'training_called':False,'validation_rows_read':0,'reserved_test_rows_read':0}
    for card in key['rows']:
        group={m['condition']:m for m in metrics if m['paper_key']==card['paper_key']}
        paper={'paper_key':card['paper_key'],'control_id':card['expression_operation']['control_id'],
               'grouping_origin':card['grouping_origin'],'conditions':{}}
        for condition,m in group.items():
            a=m['assessment']
            paper['conditions'][condition]={k:m[k] for k in ('source_overlap','source_token_edits','reference_overlap_evaluation_only','output_sentences')}|{
                'operation_assessment':({k:a['operation_assessment'][k] for k in ('form_present','within_fixed_grouping')} if a else None),
                'form_observations':a['form_observations'] if a else None,
                'incremental_style_benefit':a['incremental_style_benefit'] if a else None}
        if all(c in group for c in ('grouping_control','selective_expression')):
            a,b=group['grouping_control'],group['selective_expression']
            paper['primary_contrast']={'source_overlap_delta':b['source_overlap']-a['source_overlap'],
                'control_guided_overlap':overlap(a['output'],b['output']),
                'control_guided_token_edits':shared.shared.token_edit_distance(a['output'],b['output']),
                'reference_overlap_delta_evaluation_only':b['reference_overlap_evaluation_only']-a['reference_overlap_evaluation_only']}
        summary['per_paper'].append(paper)
    for control in experiment.OPERATIONS:
        group=[m for m in metrics if m['control_id']==control];control_rows=[m for m in group if m['condition']=='grouping_control'];guided=[m for m in group if m['condition']=='selective_expression']
        def count(rows,field):return sum(m['assessment'][field] for m in rows) if reviews else None
        summary['per_operation'][control]={'papers':len({m['paper_key'] for m in group}),
            'grouping_control_strict_form':sum(m['assessment']['operation_assessment']['form_present'] for m in control_rows) if reviews else None,
            'guided_strict_form':sum(m['assessment']['operation_assessment']['form_present'] for m in guided) if reviews else None,
            'guided_grouping_matches':sum(m['assessment']['operation_assessment']['within_fixed_grouping'] for m in guided) if reviews else None,
            'guided_incremental_style_benefits':dict(Counter(m['assessment']['incremental_style_benefit'] for m in guided)) if reviews else None}
    fresh(output);(output/'raw').mkdir()
    for name,digest in coverage['response_file_hashes'].items():
        shutil.copyfile(responses/name,output/'raw'/name)
        if experiment.binding.file_sha(output/'raw'/name)!=digest:raise ValueError('response_changed_during_import')
    if (responses/'notes.txt').exists():shutil.copyfile(responses/'notes.txt',output/'raw/notes.txt')
    if assessments:shutil.copyfile(assessments,output/'assessments.private.json')
    write_json(output/'coverage.json',coverage);write_lines(output/'metrics.private.jsonl',metrics);write_json(output/'summary.json',summary)
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'code_sha256':experiment.binding.file_sha(Path(__file__)),
        'inherited_validator_sha256':experiment.binding.file_sha(Path(shared.__file__)),
        'package_manifest_sha256':experiment.binding.file_sha(package/'manifest.json'),
        'response_hashes':coverage['response_file_hashes'],'assessment_sha256':experiment.binding.file_sha(assessments) if assessments else None,
        'files':{str(p.relative_to(output)):experiment.binding.file_sha(p) for p in output.rglob('*') if p.is_file()}})
    print('Preserved '+str(len(accepted))+'/12 responses; '+summary['status']+'. No model calls or training.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True);p.add_argument('--responses',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--assessments',type=Path)
    a=p.parse_args();run(a.package,a.responses,a.output_dir,a.assessments)


if __name__=='__main__':main()
