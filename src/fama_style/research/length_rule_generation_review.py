"""Audit fixed-rule generation and prepare verification of every available item."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from . import length_rule_replication as preparation
from .author_control_replication import verify_files
from .relation_pilot_review import file_manifest, file_sha, read_rows
from .relation_pilot_stage2_review import candidate_diagnostics
from .relation_pilot_verification import allocate
from fama_style.rhetoric.experiment import fresh, write_json, write_lines

PROTOCOL='fixed_length_generation_review_v1'


def audit(packet,output,session_path):
    verify_files(packet)
    if json.loads((packet/'common/generation_output.schema.json').read_text())!=preparation.generation_schema():
        raise ValueError('generation_specification_changed')
    paths=sorted((packet/'responses/generation').glob('responses_*.jsonl'))
    batches=sorted((packet/'generation').glob('batch_*'))
    if [p.name for p in paths]!=['responses_%02d.jsonl'%i for i in range(1,len(batches)+1)]:
        raise ValueError('response_file_coverage')
    selection={r['item_id']:r for r in read_rows(packet/'PRIVATE_SELECTION.jsonl')}
    rows=[];details=[];seen=set()
    for bi,path in enumerate(paths,1):
        expected={r['item_id']:r for r in read_rows(packet/('generation/batch_%02d/items.jsonl'%bi))}
        received=read_rows(path);ids=[r.get('item_id') for r in received]
        if len(ids)!=len(expected) or len(set(ids))!=len(ids) or set(ids)!=set(expected) or seen & set(ids):
            raise ValueError('missing_duplicate_or_wrong_batch_ids')
        seen.update(ids)
        for row in received:
            key=row['item_id'];errors=preparation.check_generation(row,expected[key])
            if preparation.schema_errors(row,preparation.generation_schema()):raise ValueError('schema_errors:'+key)
            source=selection[key]
            diagnostics=candidate_diagnostics(row['left_sentence'],row['right_sentence'],row['candidate']) if not errors and row['generation_status']=='generated' else None
            details.append({'item_id':key,'batch':bi,'mechanically_valid':not errors,'errors':errors,
                **{k:source[k] for k in ('paragraph_id','paper_id','boundary_id','stratum','rule_merge','pair_words')},
                'candidate_diagnostics':diagnostics})
            rows.append(row)
    if seen!=set(selection):raise ValueError('selection_coverage')
    session=json.loads(session_path.read_text());fresh(output);(output/'raw_generation').mkdir()
    for path in sorted((packet/'responses/generation').iterdir()):
        if path.is_file():shutil.copyfile(path,output/'raw_generation'/path.name)
    shutil.copyfile(session_path,output/'user_reported_session.json')
    shutil.copyfile(packet/'SESSION_NOTES.md',output/'original_session_notes.md')
    write_lines(output/'annotations.private.jsonl',rows);write_lines(output/'row_audit.private.jsonl',details)
    valid=[r for r,d in zip(rows,details) if d['mechanically_valid']]
    summary={'protocol':PROTOCOL,'records':len(rows),'response_files':len(paths),
        'mechanically_valid_records':len(valid),'invalid_records':len(rows)-len(valid),
        'generation_status':dict(Counter(r['generation_status'] for r in valid)),
        'by_stratum':{s:dict(Counter(r['generation_status'] for r,d in zip(rows,details) if d['mechanically_valid'] and d['stratum']==s)) for s in preparation.CONFIG['strata']},
        'surface_risk_flags':dict(Counter(f for d in details if d['candidate_diagnostics'] for f in d['candidate_diagnostics']['risk_flags'])),
        'semantic_status':'all_candidates_unverified','candidate_texts_repaired':0,
        'raw_provenance_both_null':sum(r['provenance']['visible_model_label'] is None and r['provenance']['thinking_setting'] is None for r in rows),
        'reported_model':session['visible_model_label'],'requested_effort':session['requested_thinking_setting'],
        'effective_effort':session['effective_thinking_setting'],
        'cutoff_refitted':False,'verification_outcomes_available':False,'new_model_calls':0,
        'training_called':False,'gpu_jobs_started':0,'reserved_content_used':False}
    write_json(output/'summary.json',summary)
    file_manifest(output,protocol=PROTOCOL,code_sha256=file_sha(Path(__file__)),
        input_packet_manifest_sha256=file_sha(packet/'manifest.json'))
    return summary


def prepare_verification(packet,audited,output):
    verify_files(packet);manifest=verify_files(audited)
    if manifest['input_packet_manifest_sha256']!=file_sha(packet/'manifest.json'):raise ValueError('audit_packet_mismatch')
    summary=json.loads((audited/'summary.json').read_text())
    if summary['invalid_records']:raise ValueError('invalid_generation_records_require_resolution')
    details={r['item_id']:r for r in read_rows(audited/'row_audit.private.jsonl')}
    jobs=[];unavailable=[]
    for row in read_rows(audited/'annotations.private.jsonl'):
        detail=details[row['item_id']]
        if row['generation_status']=='abstained':
            unavailable.append({'item_id':row['item_id'],'reason':row['abstention_reason']});continue
        # Phase-specific ordering key changes batch assignment, not item identity.
        jobs.append({'paragraph_id':detail['paragraph_id'],'paper_id':detail['paper_id'],
            'payload':{'item_id':preparation.digest('verification|'+row['item_id'])},
            'actual_payload':preparation.verification_payload(row),'detail':detail})
    batches=allocate(jobs);fresh(output);(output/'common').mkdir();(output/'responses').mkdir()
    prompt=(packet/'common/VERIFICATION_PROMPT_SPECIFICATION.md').read_text()
    (output/'common/PROMPT.md').write_text(prompt)
    shutil.copyfile(packet/'common/verification_output.schema.json',output/'common/verification_output.schema.json')
    mapping=[]
    for bi,batch in enumerate(batches,1):
        folder=output/('batches/batch_%02d'%bi);folder.mkdir(parents=True)
        write_lines(folder/'items.jsonl',[j['actual_payload'] for j in batch])
        (folder/'TASK.md').write_text(prompt+'\nProcess all %d fixed items. Save responses_%02d.jsonl and coverage_%02d.md.\n'%(len(batch),bi,bi))
        for job in batch:
            d=job['detail'];mapping.append({**{k:d[k] for k in ('item_id','paragraph_id','paper_id','boundary_id','stratum','rule_merge','pair_words')},
                'batch':bi,'risk_flags':d['candidate_diagnostics']['risk_flags']})
    write_json(output/'PRIVATE_MAPPING.json',mapping);write_json(output/'unavailable.private.json',unavailable)
    report={'protocol':'fixed_length_verification_package_v1','status':'verification_ready_not_executed',
        'items':len(jobs),'batches':len(batches),'batch_sizes':[len(b) for b in batches],
        'unavailable_generation_abstentions':len(unavailable),
        'strata':dict(Counter(m['stratum'] for m in mapping)),
        'selection':'every_valid_generated_candidate_from_both_source_length_strata',
        'selected_preferred_subset':False,'cutoff_decisions_exported':False,
        'generation_context_notes_or_provenance_exported':False,'source_reference_or_claims_exported':False,
        'fidelity_and_readability_separate':True,'semantic_certification_claimed':False,
        'new_model_calls':0,'training_called':False,'gpu_jobs_started':0,'reserved_content_used':False}
    write_json(output/'summary.json',report)
    (output/'START_HERE.md').write_text('''# Fixed-candidate verification for the additional-paper replication

Verify all available candidates, without choosing or repairing them. There are
%d items in %d batches. For each batches/batch_XX, paste TASK.md and attach ONLY
its items.jsonl plus common/verification_output.schema.json. Use GPT-6 and request
High thinking effort; report effective effort as unknown if unclear.

Save the first complete unedited responses_XX.jsonl and coverage_XX.md in THIS
packet's responses/ directory. Notify Codex when every batch is ready. Do not
overwrite generation responses. Record known retries/settings in your reply or
SESSION_NOTES.md; unknown details can remain unknown. No credentials needed.

Do not upload this file, PRIVATE_MAPPING, other answers, previous instructions,
generation notes, source metadata, originals, protocol, cutoff/rule decisions or
surface warnings. Inputs contain only item ID, full neutral source, adjacent pair
and the fixed candidate. Assess claim fidelity and readability separately using
fresh V-units and exact evidence. Local shifts and abstentions are valid outcomes,
not automatic approvals for training. Do not infer an expected preferred layout.

Every actual generated candidate is included; no candidate was discarded for
length, earlier preference or a surface warning. Generation context notes and
raw model provenance remain offline. New batch allocation may change neighbors,
but same-model self-review, pretraining and possible chat retrieval prevent any
guarantee of independence. These reuse earlier nonreserved training-pool neutral
paragraphs; upstream context/meaning cautions remain in force. Pairwise judgments
do not license chained merges. No classifier, threshold refit or Kaggle run is needed.
'''%(len(jobs),len(batches)))
    (output/'SESSION_NOTES.md').write_text('Visible model:\nRequested effort:\nKnown effective effort or unknown:\nContext arrangement if known:\nRetries/edits if known:\n\nNo credentials.\n')
    file_manifest(output,protocol=report['protocol'],code_sha256=file_sha(Path(__file__)),
        generation_audit_manifest_sha256=file_sha(audited/'manifest.json'),
        initial_packet_manifest_sha256=file_sha(packet/'manifest.json'))
    manifest=json.loads((output/'manifest.json').read_text());manifest['files'].pop('SESSION_NOTES.md');write_json(output/'manifest.json',manifest)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','audit-output','session','verification-output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();s=audit(a.packet,a.audit_output,a.session)
    s['verification']=prepare_verification(a.packet,a.audit_output,a.verification_output);print(json.dumps(s,indent=2))


if __name__=='__main__':main()
