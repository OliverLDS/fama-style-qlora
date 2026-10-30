"""Import fixed-candidate verification without treating judgments as certificates."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from .author_control_replication import verify_files
from .human_prose_review import occurrence_span, schema_errors
from .relation_pilot_review import file_manifest, file_sha, read_rows
from fama_style.rhetoric.experiment import fresh, write_json, write_lines

PROTOCOL = 'relation_pilot_verification_review_v1'
TEXT_FIELDS = ('source_paragraph', 'left_sentence', 'right_sentence', 'candidate')


def check_row(row, expected, schema):
    errors = schema_errors(row, schema)
    if errors: return errors
    def error(code, **detail): errors.append({'code': code, **detail})
    if row['item_id'] != expected['item_id']: error('item_id_changed')
    for key in TEXT_FIELDS:
        if row[key] != expected[key]: error('fixed_text_changed', field=key)
    units = row['source_units']
    if [u['unit_id'] for u in units] != ['V'+str(i+1) for i in range(len(units))]: error('unit_sequence')
    def evidence(anchors, text, required, location):
        if required and not anchors: error('missing_evidence', location=location)
        for anchor in anchors:
            if occurrence_span(text, anchor) is None: error('invalid_exact_evidence', location=location)
    for unit in units:
        uid = unit['unit_id']
        evidence(unit['source_evidence'], row['source_paragraph'], True, uid+'/source')
        evidence(unit['candidate_evidence'], row['candidate'], unit['alignment'] in ('preserved', 'changed'), uid+'/candidate')
    evidence(row['context_evidence'], row['source_paragraph'], False, 'context')
    for added in row['added_content']:
        evidence(added['candidate_evidence'], row['candidate'], True, 'added_content')
    if not row['readability_evidence']: error('missing_readability_evidence')
    for anchor in row['readability_evidence']:
        text = row['source_paragraph'] if anchor['text_origin']=='source' else row['candidate']
        evidence([anchor], text, True, 'readability')
    if row['fidelity_verdict']=='faithful' and (row['added_content'] or any(u['alignment']!='preserved' for u in units)):
        error('faithful_verdict_conflicts_with_units_or_additions')
    if not row['fidelity_rationale'].strip() or not row['readability_rationale'].strip(): error('missing_rationale')
    return errors


def grouped_summary(rows):
    """These are reported model judgments, never verified truth or training labels."""
    return {'items': len(rows), 'fidelity': dict(Counter(r['fidelity_verdict'] for r in rows)),
        'readability': dict(Counter(r['readability_in_context'] for r in rows)),
        'joint': dict(Counter(r['fidelity_verdict']+'/'+r['readability_in_context'] for r in rows)),
        'units': dict(Counter(u['alignment'] for r in rows for u in r['source_units']))}


def audit(packet, output, session_path):
    verify_files(packet)
    schema = json.loads((packet/'common/verification_output.schema.json').read_text())
    mapping = {m['item_id']: m for m in json.loads((packet/'PRIVATE_MAPPING.json').read_text())}
    session = json.loads(session_path.read_text())
    batches = sorted((packet/'batches').glob('batch_*'))
    paths = sorted((packet/'responses').glob('responses_*.jsonl'))
    if [p.name for p in paths] != ['responses_%02d.jsonl'%i for i in range(1,len(batches)+1)]:
        raise ValueError('response_file_coverage')
    rows = []; details = []; seen = set()
    for batch, path in enumerate(paths, 1):
        expected = {r['item_id']: r for r in read_rows(packet/('batches/batch_%02d/items.jsonl'%batch))}
        received = read_rows(path); ids = [r.get('item_id') for r in received]
        if len(ids)!=len(expected) or len(set(ids))!=len(ids) or set(ids)!=set(expected) or seen & set(ids):
            raise ValueError('missing_duplicate_or_wrong_batch_ids')
        seen.update(ids)
        for row in received:
            key = row['item_id']
            if schema_errors(row,schema): raise ValueError('schema_errors:'+key)
            errors = check_row(row,expected[key],schema)
            details.append({'item_id': key, 'batch': batch, 'errors': errors, 'mechanically_valid': not errors,
                'paragraph_id': mapping[key]['paragraph_id'], 'paper_id': mapping[key]['paper_id'],
                'boundary_id': mapping[key]['boundary_id'], 'prior_surface_risk_flags': mapping[key]['risk_flags']})
            rows.append(row)
    if seen != set(mapping): raise ValueError('private_mapping_coverage')
    fresh(output); raw = output/'raw_responses'; raw.mkdir()
    for path in sorted((packet/'responses').iterdir()):
        if path.is_file(): shutil.copyfile(path,raw/path.name)
    shutil.copyfile(packet/'SESSION_NOTES.md',output/'original_session_notes.md')
    shutil.copyfile(session_path,output/'user_reported_session.json')
    shutil.copyfile(packet/'PRIVATE_MAPPING.json',output/'PRIVATE_MAPPING.json')
    write_lines(output/'annotations.private.jsonl',rows)
    write_lines(output/'row_audit.private.jsonl',details)
    valid = [r for r,d in zip(rows,details) if d['mechanically_valid']]
    groups = {}
    for paper in sorted({d['paper_id'] for d in details}):
        groups[paper] = grouped_summary([r for r,d in zip(rows,details) if d['paper_id']==paper and d['mechanically_valid']])
    sensitivity = {label: grouped_summary([r for r,d in zip(rows,details) if d['mechanically_valid'] and bool(d['prior_surface_risk_flags'])==flagged])
                   for label,flagged in [('surface_flagged',True),('surface_unflagged',False)]}
    summary = {'protocol': PROTOCOL, 'response_files': len(paths), 'records': len(rows),
        'mechanically_valid_records': len(valid), 'invalid_records': len(rows)-len(valid),
        'valid_model_judgments': grouped_summary(valid), 'per_paper': groups, 'surface_flag_sensitivity': sensitivity,
        'candidate_single_sentence': dict(Counter(r['candidate_single_sentence'] for r in valid)),
        'issue_categories': dict(Counter(k for r in valid for k in set(r['issue_categories']))),
        'needs_external_context': sum(r['needs_external_context'] for r in valid),
        'faithful_merged_better': sum(r['fidelity_verdict']=='faithful' and r['readability_in_context']=='merged_better' for r in valid),
        'local_shift_merged_better': sum(r['fidelity_verdict']=='local_shift' and r['readability_in_context']=='merged_better' for r in valid),
        'reported_model': session['visible_model_label'], 'reported_effort': session['thinking_setting'],
        'raw_provenance_both_null': sum(r['provenance']['visible_model_label'] is None and r['provenance']['thinking_setting'] is None for r in rows),
        'judgments_independently_verified': False, 'selected_positive_sample': True,
        'automatic_training_acceptance': False, 'approved_training_rows': 0, 'labels_repaired': 0,
        'new_model_calls': 0, 'gpu_jobs_started': 0, 'training_called': False,
        'reserved_content_used': False}
    write_json(output/'summary.json',summary)
    file_manifest(output, protocol=PROTOCOL, code_sha256=file_sha(Path(__file__)),
        input_packet_manifest_sha256=file_sha(packet/'manifest.json'))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('packet','output','session'): p.add_argument('--'+name,type=Path,required=True)
    a = p.parse_args(); print(json.dumps(audit(a.packet,a.output,a.session),indent=2))


if __name__=='__main__': main()
