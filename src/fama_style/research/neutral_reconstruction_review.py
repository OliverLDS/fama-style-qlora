"""Audit Stage 1 responses and prepare paired Stage 2 jobs from explicit decisions.

Raw replies and frozen source data remain unchanged. Alignment decisions are
automated qualitative assessments, not independent semantic gold labels.
"""
import argparse
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from .author_control_replication import verify_files
from .neutral_reconstruction import make_handoff, sha, validate_response, write_json, write_rows

PROTOCOL='neutral_reconstruction_stage1_review_v1'


def file_sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def patch_brief(raw,patches):
    result=copy.deepcopy(raw)
    for patch in patches:
        parent=result
        if not patch['path']:raise ValueError('root_brief_replacement_not_allowed')
        for key in patch['path'][:-1]:parent=parent[key]
        key=patch['path'][-1]
        if parent[key]!=patch['before']:raise ValueError('brief_patch_before_value_mismatch')
        parent[key]=copy.deepcopy(patch['after'])
    return result


def clean_markers(original,repairs,evidence_dir,pdf_sha):
    text=original
    for repair in repairs:
        before,after,marker=repair['before'],repair['after'],repair['marker']
        compact=lambda s:re.sub(r'\s+','',s)
        if (not marker.isdigit() or before.count(marker)!=1
                or compact(before).replace(marker,'',1)!=compact(after)):
            raise ValueError('source_repair_is_not_marker_deletion_only')
        image=evidence_dir/repair['evidence_image']
        if file_sha(image)!=repair['evidence_image_sha256'] or repair['paper_pdf_sha256']!=pdf_sha:
            raise ValueError('source_repair_evidence_mismatch')
        if text.count(before)!=1:raise ValueError('source_marker_anchor_not_unique')
        text=text.replace(before,after,1)
    return text


def build(root,package,review,output):
    root,package,review=root.resolve(),package.resolve(),review.resolve()
    if output.exists() or (review/'results').exists():raise ValueError('refuse_to_overwrite_review_or_generation_packet')
    verify_files(package)
    index=json.loads((package/'batch_index.json').read_text())
    keys={r['record_id']:r for r in map(json.loads,(package/'private_key/selection.private.jsonl').read_text().splitlines())}
    decisions=json.loads((review/'review_decisions.private.json').read_text())['decisions']
    if len({d['record_id'] for d in decisions})!=len(decisions) or {d['record_id'] for d in decisions}!=set(keys):
        raise ValueError('decision_coverage_mismatch')
    responses={};mechanical=[];response_files={}
    for batch in index:
        path=package/batch['save_response_as'];response_files[str(path.relative_to(root))]=file_sha(path)
        rows=[json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        ids=[r['record_id'] for r in rows]
        if len(ids)!=len(set(ids)) or set(ids)!=set(batch['record_ids']):raise ValueError('response_batch_coverage_mismatch')
        for row in rows:
            source=keys[row['record_id']]
            mechanical.append({'record_id':row['record_id']}|validate_response(row,source))
            responses[row['record_id']]=row
    if set(responses)!=set(keys):raise ValueError('response_coverage_mismatch')
    approved=[];audit=[]
    for decision in decisions:
        rid=decision['record_id'];source=keys[rid];raw=responses[rid]
        record={'record_id':rid,'batch':source['batch'],'paper_id':source['paper_id'],
                'style_group':source['style_group'],'decision':decision['decision'],
                'source_audit_as_received':raw['source_audit'],'brief_needs_review_as_received':(raw['brief'] or {}).get('needs_review',True),
                'brief_gaps_as_received':(raw['brief'] or {}).get('context_gaps',[]),
                'rationale':decision['rationale'],'brief_patches':decision['brief_patches'],
                'source_repairs':decision['source_repairs'],'raw_original_sha256':source['original_sha256'],
                'raw_brief_sha256':sha(json.dumps(raw['brief'],sort_keys=True)),
                'semantic_gold_verified':False,'training_ready':False}
        record['evidence_images']=decision.get('evidence_images',[])
        for evidence in record['evidence_images']:
            if file_sha(review/evidence['file'])!=evidence['sha256']:raise ValueError('review_image_changed')
        if decision['decision']=='hold':
            if decision['brief_patches'] or decision['source_repairs']:raise ValueError('held_case_has_unapproved_repairs')
            audit.append(record);continue
        if decision['decision']!='approve_provisional_generation' or not decision['rationale']:
            raise ValueError('explicit_alignment_decision_required')
        brief=patch_brief(raw['brief'],decision['brief_patches'])
        text=clean_markers(source['original_paragraph'],decision['source_repairs'],review,source['paper_pdf_sha256'])
        handoff=make_handoff(brief,text)
        record.update(generation_original_sha256=sha(text),reviewed_brief_sha256=sha(json.dumps(brief,sort_keys=True)),
                      handoff_sha256=sha(json.dumps(handoff,sort_keys=True)))
        approved.append({'record_id':rid,'batch':source['batch'],'paper_id':source['paper_id'],'style_group':source['style_group'],
                         'original_paragraph':text,'raw_stage1_original':source['original_paragraph'],
                         'raw_original_sha256':source['original_sha256'],'generation_original_sha256':sha(text),
                         'brief':brief,'handoff':handoff,'source_repairs':decision['source_repairs'],
                         'source_variant':'footnote_markers_removed_v1' if decision['source_repairs'] else 'unchanged',
                         'status':'provisional_generation_only','training_ready':False})
        audit.append(record)
    summary={'protocol':PROTOCOL,'responses_received':len(responses),'mechanically_valid':len(mechanical),
             'source_status_as_received':dict(Counter(r['source_audit']['status'] for r in responses.values())),
             'briefs_needing_review_as_received':sum(r['brief']['needs_review'] for r in responses.values()),
             'mechanically_clear_as_received':sum(r['eligible_for_semantic_assessment'] for r in mechanical),
             'provisional_generation_approvals':len(approved),'held_cases':len(keys)-len(approved),
             'approved_by_group':dict(Counter(r['style_group'] for r in approved)),
             'approved_papers':len({r['paper_id'] for r in approved}),
             'brief_fields_changed':sum(len(r['brief_patches']) for r in audit),
             'derived_references_with_marker_repairs':sum(bool(r['source_repairs']) for r in approved),
             'superscript_markers_removed':sum(len(r['source_repairs']) for r in approved),
             'generation_jobs_prepared':2*len(approved),'generation_outputs_observed':0,
             'original_12_case_denominator_retained':True,'cases_replaced':0,'raw_responses_modified':False,
             'independent_or_blinded_verification':False,'training_ready':False,'training_pairs_created':0,
             'model_calls_by_agent':0,'detector_calls':0,'GPU_jobs':0,'reserved_content_used':False}
    output.mkdir(parents=True);(output/'inputs/direct').mkdir(parents=True);(output/'inputs/reconstruction').mkdir(parents=True)
    (output/'prompts').mkdir();(output/'private_key').mkdir()
    generation_index=[]
    for batch in index:
        number=batch['batch'];rows=[r for r in approved if r['batch']==number]
        if not rows:continue
        for method,folder in [('direct_rewrite','direct'),('neutral_reconstruction','reconstruction')]:
            payload=[{'record_id':r['record_id'],'original_paragraph':r['original_paragraph']} if folder=='direct'
                     else {'record_id':r['record_id'],'content_brief':r['handoff']} for r in rows]
            path=output/'inputs'/folder/f'inputs_{number:02d}.jsonl';write_rows(path,payload)
            generation_index.append({'method':method,'batch':number,'expected_records':len(rows),
                'input_file':str(path.relative_to(output)),'record_ids':[r['record_id'] for r in rows],
                'save_response_as':f'responses/{folder}/{folder}_{number:02d}.jsonl'})
    for folder in ('direct','reconstruction'):
        target=output/'responses'/folder;target.mkdir(parents=True)
        (target/'README.txt').write_text('Save first raw JSONL generation responses here; keep blocked rows and original IDs.\n')
    for name in ('DIRECT_REWRITE_PROMPT.md','RECONSTRUCTION_PROMPT.md'):
        (output/'prompts'/name).write_bytes((package/'prompts'/name).read_bytes())
    (output/'prompts/generation_output.schema.json').write_bytes((root/'docs/neutral_reconstruction_v1/generation_output.schema.json').read_bytes())
    (output/'START_HERE.md').write_bytes((root/'docs/neutral_reconstruction_v1/STAGE2_START_HERE.md').read_bytes())
    write_rows(output/'private_key/reviewed_references.private.jsonl',approved)
    write_rows(output/'private_key/selection_and_holds.private.jsonl',audit)
    write_json(output/'generation_index.json',generation_index);write_json(output/'summary.json',summary)
    write_json(output/'session_notes.json',{'requested_model':'GPT-6','requested_effort':'High',
        'actual_model':None,'actual_effort':None,'conversation_arrangement':None,'cross_chat_retrieval':'uncontrolled',
        'instructions':'Record actual generation settings; do not infer them from Stage 1 provenance. No credentials.'})
    write_json(output/'protocol.json',{'protocol':'neutral_reconstruction_stage2_v1','source_selection_count':12,
        'paired_generation_count':len(approved),'methods':['direct_rewrite','neutral_reconstruction'],
        'source_repairs':'same derived reference for both methods and later fidelity checks; old originals unchanged',
        'brief_review':'versioned Codex automated qualitative assessment; not independent gold labels',
        'analysis':'all approved pairs; report blocked counts by original batch and group; no replacements or output-based selection',
        'detector_feedback':False,'model_and_effort':'match actual settings where possible; record UI limitations',
        'sampling_parameters':'UI hidden; not claimed controlled','generation_status':'prepared_not_executed'})
    result=review/'results';result.mkdir()
    write_json(result/'summary.json',summary);write_rows(result/'mechanical_checks.private.jsonl',mechanical)
    write_rows(result/'alignment_audit.private.jsonl',audit)
    write_rows(result/'derived_briefs.private.jsonl',[{'record_id':r['record_id'],'brief':r['brief']} for r in approved])
    write_json(result/'response_receipt.private.json',{'files':response_files,
        'user_provenance':json.loads((review/'session_provenance.private.json').read_text()),'raw_files_unchanged':True})
    files={str(p.relative_to(output)):file_sha(p) for p in output.rglob('*') if p.is_file()
           and 'responses' not in p.relative_to(output).parts and p.name!='session_notes.json'}
    write_json(output/'manifest.json',{'protocol':'neutral_reconstruction_stage2_v1','files':files,
        'code_sha256':file_sha(Path(__file__)),'decision_sha256':file_sha(review/'review_decisions.private.json'),
        'stage1_manifest_sha256':file_sha(package/'manifest.json'),'raw_response_sha256':response_files,
        'mutable_unhashed':['responses/','session_notes.json']})
    write_json(result/'manifest.json',{'protocol':PROTOCOL,'files':{p.name:file_sha(p) for p in result.iterdir() if p.is_file()},
        'decision_sha256':file_sha(review/'review_decisions.private.json'),'code_sha256':file_sha(Path(__file__))})
    # Ensure receipt writes never touched first raw responses.
    if any(file_sha(root/name)!=digest for name,digest in response_files.items()):raise ValueError('raw_response_changed')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3])
    parser.add_argument('--package',type=Path,required=True);parser.add_argument('--review',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(build(args.root,args.package,args.review,args.output),indent=2))


if __name__=='__main__':main()
