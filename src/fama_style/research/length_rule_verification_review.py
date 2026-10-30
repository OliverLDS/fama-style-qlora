"""Audit additional-paper verification and score the frozen cutoff, without fits."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from . import length_rule_replication as preparation
from . import relation_pilot_verification_review as verification_checks
from . import verified_merge_selector as decision_metrics
from .author_comparison_corpus import words
from .author_control_replication import verify_files
from .human_prose_review import schema_errors
from .relation_pilot_review import file_manifest, file_sha, read_rows
from fama_style.rhetoric.experiment import fresh, write_json, write_lines

PROTOCOL='fixed_length_verification_review_v1'


def policy_rows(selection,generation,verification):
    """Fixed decisions on every sampled item, with explicit unavailable outcomes."""
    result=[]
    for item_id,source in sorted(selection.items()):
        available=generation[item_id]['generation_status']=='generated'
        count=len(words(source['left_sentence']))+len(words(source['right_sentence']))
        stratum='at_most_30' if count<=30 else 'over_30'
        if count!=source['pair_words'] or stratum!=source['stratum'] or (count<=30)!=source['rule_merge']:
            raise ValueError('frozen_rule_or_word_count_changed')
        if available:
            r=verification[item_id];group,label=decision_metrics.cohort(r)
            fidelity=r['fidelity_verdict'];readability=r['readability_in_context']
        else:
            group,label='generation_abstention',None;fidelity=readability='unavailable'
        predictions={'keep':0,'merge_all_available':int(available),'fixed_30_available':int(available and count<=30)}
        for method,prediction in predictions.items():
            result.append({'item_id':item_id,'paragraph_id':source['paragraph_id'],'paper_id':source['paper_id'],
                'stratum':stratum,'pair_words':count,'available':available,'cohort':group,'label':label,
                'fidelity':fidelity,'readability':readability,'method':method,'prediction':prediction})
    return result


def policy_score(rows):
    value=decision_metrics.score(rows)
    value['all_items']=len(rows);value['all_selected_merges']=sum(r['prediction'] for r in rows)
    value['side_cohorts']['generation_abstention']={'items':sum(r['cohort']=='generation_abstention' for r in rows),
        'selected_merges':sum(r['cohort']=='generation_abstention' and r['prediction']==1 for r in rows)}
    return value


def run(initial,generation_audit,packet,session_path,output):
    verify_files(initial);gen_manifest=verify_files(generation_audit);packet_manifest=verify_files(packet)
    if gen_manifest['input_packet_manifest_sha256']!=file_sha(initial/'manifest.json'):
        raise ValueError('generation_initial_packet_mismatch')
    if packet_manifest['generation_audit_manifest_sha256']!=file_sha(generation_audit/'manifest.json') or packet_manifest['initial_packet_manifest_sha256']!=file_sha(initial/'manifest.json'):
        raise ValueError('verification_generation_packet_mismatch')
    config=json.loads((initial/'protocol.json').read_text())
    if config!=preparation.CONFIG:raise ValueError('frozen_protocol_changed')
    if json.loads((generation_audit/'summary.json').read_text())['invalid_records']:
        raise ValueError('invalid_generation_audit')
    schema=json.loads((packet/'common/verification_output.schema.json').read_text())
    selection={r['item_id']:r for r in read_rows(initial/'PRIVATE_SELECTION.jsonl')}
    generation={r['item_id']:r for r in read_rows(generation_audit/'annotations.private.jsonl')}
    mapping={r['item_id']:r for r in json.loads((packet/'PRIVATE_MAPPING.json').read_text())}
    if set(selection)!=set(generation):raise ValueError('generation_selection_coverage')
    available={k for k,r in generation.items() if r['generation_status']=='generated'}
    if set(mapping)!=available:raise ValueError('available_candidate_coverage')
    paths=sorted((packet/'responses').glob('responses_*.jsonl'))
    batches=sorted((packet/'batches').glob('batch_*'))
    if [p.name for p in paths]!=['responses_%02d.jsonl'%i for i in range(1,len(batches)+1)]:raise ValueError('response_file_coverage')
    annotations=[];details=[];seen=set()
    for bi,path in enumerate(paths,1):
        expected={r['item_id']:r for r in read_rows(packet/('batches/batch_%02d/items.jsonl'%bi))}
        received=read_rows(path);ids=[r.get('item_id') for r in received]
        if len(ids)!=len(expected) or len(set(ids))!=len(ids) or set(ids)!=set(expected) or seen & set(ids):
            raise ValueError('missing_duplicate_or_wrong_batch_ids')
        seen.update(ids)
        for row in received:
            key=row['item_id']
            if expected[key]!=preparation.verification_payload(generation[key]):raise ValueError('candidate_transport_changed')
            if schema_errors(row,schema):raise ValueError('verification_schema_errors:'+key)
            errors=verification_checks.check_row(row,expected[key],schema)
            details.append({'item_id':key,'batch':bi,'mechanically_valid':not errors,'errors':errors,
                **{k:selection[key][k] for k in ('paper_id','paragraph_id','stratum','rule_merge','pair_words')}})
            annotations.append(row)
    if seen!=available:raise ValueError('verification_selection_coverage')
    session=json.loads(session_path.read_text());fresh(output);(output/'raw_responses').mkdir()
    for path in sorted((packet/'responses').iterdir()):
        if path.is_file():shutil.copyfile(path,output/'raw_responses'/path.name)
    shutil.copyfile(session_path,output/'user_reported_session.json')
    shutil.copyfile(packet/'SESSION_NOTES.md',output/'original_session_notes.md')
    shutil.copyfile(initial/'protocol.json',output/'protocol.json')
    write_lines(output/'annotations.private.jsonl',annotations);write_lines(output/'row_audit.private.jsonl',details)
    valid=[r for r,d in zip(annotations,details) if d['mechanically_valid']]
    invalid=len(annotations)-len(valid)
    summary={'protocol':PROTOCOL,'status':'blocked_mechanical_issues' if invalid else 'frozen_rule_replication_complete',
        'config':config,'response_files':len(paths),'sampled_items':len(selection),'available_candidates':len(available),
        'generation_abstentions':len(selection)-len(available),'mechanically_valid_records':len(valid),'invalid_records':invalid,
        'valid_model_judgments':verification_checks.grouped_summary(valid),
        'candidate_single_sentence':dict(Counter(r['candidate_single_sentence'] for r in valid)),
        'issue_categories':dict(Counter(c for r in valid for c in set(r['issue_categories']))),
        'needs_external_context':sum(r['needs_external_context'] for r in valid),
        'reported_model':session['visible_model_label'],'requested_effort':session['requested_thinking_setting'],
        'effective_effort':session['effective_thinking_setting'],
        'raw_provenance_both_null':sum(r['provenance']['visible_model_label'] is None and r['provenance']['thinking_setting'] is None for r in annotations),
        'selected_preferred_subset':False,'source_length_stratified_sample':True,'prior_project_exposure':True,
        'judgments_independently_verified':False,'cutoff_refitted':False,'classifier_fitted':False,
        'approved_training_rows':0,'texts_or_labels_repaired':0,'new_model_calls':0,
        'training_called':False,'gpu_jobs_started':0,'reserved_content_used':False}
    if not invalid:
        predictions=policy_rows(selection,generation,{r['item_id']:r for r in valid})
        methods=config['methods']
        summary['cohorts']=dict(Counter(r['cohort'] for r in predictions if r['method']=='keep'))
        summary['methods']={m:policy_score([r for r in predictions if r['method']==m]) for m in methods}
        summary['per_paper']={p:{m:policy_score([r for r in predictions if r['paper_id']==p and r['method']==m]) for m in methods} for p in sorted({r['paper_id'] for r in predictions})}
        summary['per_stratum']={s:{'judgments':verification_checks.grouped_summary([r for r in valid if selection[r['item_id']]['stratum']==s]),
            'methods':{m:policy_score([r for r in predictions if r['stratum']==s and r['method']==m]) for m in methods}}
            for s in config['strata']}
        write_lines(output/'policy_predictions.private.jsonl',predictions)
    write_json(output/'summary.json',summary)
    file_manifest(output,protocol=PROTOCOL,code_sha256=file_sha(Path(__file__)),
        checker_code_sha256=file_sha(Path(verification_checks.__file__)),
        metric_code_sha256=file_sha(Path(decision_metrics.__file__)),preparation_code_sha256=file_sha(Path(preparation.__file__)),
        initial_manifest_sha256=file_sha(initial/'manifest.json'),generation_manifest_sha256=file_sha(generation_audit/'manifest.json'),
        verification_packet_manifest_sha256=file_sha(packet/'manifest.json'))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('initial','generation-audit','packet','session','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();s=run(a.initial,a.generation_audit,a.packet,a.session,a.output);print(json.dumps(s.get('methods',s),indent=2))


if __name__=='__main__':main()
