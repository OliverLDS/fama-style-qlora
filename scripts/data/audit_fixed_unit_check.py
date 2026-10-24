"""Audit fixed-cell responses, route by actual input IDs and record design deviations."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

from prepare_fixed_unit_check import check_fixed_response,compare_fixed_responses

ROOT=Path(__file__).resolve().parents[2]


def load(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def js(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def match_input_round(ids,prepared):
    """Do not rewrite IDs to pretend that a file came from its numbered batch."""
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate returned paragraph IDs')
    matches=[number for number,inputs in prepared.items() if set(ids)==set(inputs)]
    if len(matches)!=1:raise ValueError('Returned IDs do not identify one complete prepared batch')
    return matches[0]


def aggregate(comparisons):
    cells=sum(r['cells_compared'] for r in comparisons);matches=sum(r['state_matches'] for r in comparisons)
    informative=sum(r['both_informative_cells'] for r in comparisons);imatches=sum(r['both_informative_matches'] for r in comparisons)
    usable=sum(r['both_usable_and_informative_cells'] for r in comparisons);umatches=sum(r['both_usable_and_informative_matches'] for r in comparisons)
    transitions=collections.Counter()
    for comparison in comparisons:transitions.update(comparison['state_transition_counts'])
    return {'cells':cells,'matching_states':matches,'state_agreement':matches/cells if cells else None,
        'both_informative_cells':informative,'matching_informative_states':imatches,
        'informative_agreement':imatches/informative if informative else None,
        'both_usable_informative_cells':usable,'matching_usable_informative_states':umatches,
        'usable_informative_agreement':umatches/usable if usable else None,
        'state_transitions':dict(transitions),
        'semantic_accuracy':None,'general_reliability_established':False,'style_transfer_improvement_established':False}


def main(output,chat_provenance):
    import jsonschema
    if output.exists() and any(output.iterdir()):raise ValueError('Use a new audit output directory')
    base=ROOT/'research_runs/taxonomy_discovery_v1';packet=base/'fixed_unit_check_v1';responses=base/'responses/task_6_fixed_unit_check'
    prepared_manifest=json.loads((packet/'manifest.json').read_text())
    for name,h in prepared_manifest['files'].items():
        if digest(packet/name)!=h:raise ValueError('Prepared input packet changed')
    source_files=[p for p in responses.iterdir() if p.is_file()]
    before={str(p.relative_to(ROOT)):digest(p) for p in source_files}
    links=json.loads((packet/'PRIVATE_LINKS_DO_NOT_UPLOAD.json').read_text())
    source_by_id={r['paragraph_id']:r['source_id'] for r in links}
    prepared={n:{row['paragraph_id']:row for row in load(packet/f'round_{n}/paragraphs.jsonl')} for n in (1,2)}
    rows_by_file={};findings=[];deviations=[];file_summaries=[]
    for number in (1,2):
        path=responses/f'fixed_unit_annotations_{number:02}.jsonl';rows=load(path)
        actual=match_input_round([r.get('paragraph_id') for r in rows],prepared)
        if actual!=number:deviations.append({'file':path.name,'declared_round':number,'matched_input_round':actual,
            'note':'Content is validated against its actual IDs; original responses are not renamed or rewritten.'})
        validator=jsonschema.Draft202012Validator(json.loads((packet/f'round_{actual}/output.schema.json').read_text()))
        by_source={};states=collections.Counter();qualities=collections.Counter()
        for row in rows:
            ident=row['paragraph_id'];errors=list(validator.iter_errors(row))
            findings.extend({'file':path.name,'paragraph_id':ident,'code':'schema_error','path':list(e.absolute_path),'validator':e.validator} for e in errors)
            if errors:continue
            findings.extend({'file':path.name,'paragraph_id':ident,**error} for error in check_fixed_response(row,prepared[actual][ident]))
            source=source_by_id[ident]
            if source in by_source:findings.append({'file':path.name,'code':'duplicate_underlying_source'})
            by_source[source]=row;states.update(c['state'] for c in row['observations']);qualities.update(u['suitability'] for u in row['unit_assessments'])
        rows_by_file[number]=by_source
        file_summaries.append({'file':path.name,'matched_input_round':actual,'declared_round_matches':actual==number,
            'paragraphs':len(rows),'cells':sum(len(r['observations']) for r in rows),'states':dict(states),'unit_suitability':dict(qualities),
            'report_present':(responses/f'coverage_report_{number:02}.md').exists()})
    if set(rows_by_file[1])!=set(rows_by_file[2]):findings.append({'code':'underlying_source_sets_differ'})
    comparisons=[];details=[];features=collections.defaultdict(collections.Counter)
    feature_names={f['feature_id']:f for f in json.loads((packet/'round_1/feature_checklist.json').read_text())['features']}
    if not findings:
        for source in sorted(rows_by_file[1]):
            left,right=rows_by_file[1][source],rows_by_file[2][source]
            report=compare_fixed_responses(left,right);report['source_id']=source;comparisons.append(report)
            a={(r['unit_id'],r['feature_id']):r for r in left['observations']};b={(r['unit_id'],r['feature_id']):r for r in right['observations']}
            for key in a:
                x,y=a[key],b[key];feature=features[key[1]];feature['cells']+=1;feature['matches']+=x['state']==y['state']
                feature['both_present']+=x['state']==y['state']=='present';feature['both_absent']+=x['state']==y['state']=='absent'
                if x['state']!=y['state']:details.append({'source_id':source,'unit_id':key[0],'feature_id':key[1],
                    'canonical_id':x['canonical_id'],'first_proposal':x,'second_proposal':y,'adjudicated':False})
    feature_summary={fid:{'canonical_id':feature_names[fid]['canonical_id'],'name':feature_names[fid]['category']['name'],**dict(counts)} for fid,counts in sorted(features.items())}
    summary={'response_files':file_summaries,'content_findings':findings,'content_checks_passed':not findings,
        'protocol_deviations':deviations,'chat_session_provenance':chat_provenance,
        'provenance_basis':'User reported two separate new chats, without explicitly showing one response to the other' if chat_provenance=='user_confirmed_separate_chats' else 'Chat provenance not established by files',
        'cross_chat_information_isolation_verified':False,'statistical_independence_established':False,
        'memory_carryover_tested':False,'memory_isolation_rerun_required':False,
        'masked_distinct_ID_rounds_completed':not deviations,'comparison_interpretation':'Two-response-file same-input consistency check; distinct-ID batch design not completed' if deviations else 'Prepared two-ID-batch consistency check; chat isolation is separately unverified',
        'aggregate':aggregate(comparisons),'per_text':comparisons,'per_feature':feature_summary,
        'remaining_feature_disagreements':sorted({r['feature_id'] for r in details}),
        'taxonomy_modified':False,'source_annotations_modified':False,'proposals_adjudicated':False,
        'ready_for_full_mapping_training':False,'new_model_calls':0,'gpu_jobs_submitted':0,'held_out_data_used':False,
        'next_step':'Clarify F03/F04 boundaries and separate observed F12 presentation from its semantic binding/action permissions; use a small fixed-plan realization check before training a mapper.'}
    output.mkdir(parents=True,exist_ok=True);js(output/'audit_summary.json',summary)
    (output/'disagreements.private.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in details),encoding='utf-8')
    for name,h in before.items():
        if digest(ROOT/name)!=h:raise ValueError('Source response changed')
    js(output/'manifest.json',{'response_hashes':before,'responses_unchanged':True,
        'prepared_manifest_sha256':digest(packet/'manifest.json'),'auditor_sha256':digest(Path(__file__).resolve()),
        'files':{p.name:digest(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({'content_checks_passed':not findings,'protocol_deviations':len(deviations),'aggregate':summary['aggregate']}))
    if findings:raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--chat-provenance',choices=['unconfirmed','user_confirmed_separate_chats','same_chat'],default='unconfirmed')
    args=parser.parse_args();main(args.output_dir,args.chat_provenance)
