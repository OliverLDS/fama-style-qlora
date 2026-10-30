"""Audit paired annotations and prepare a strictly source-only second stage on CPU.

Transport/evidence checks do not establish semantic equivalence. Raw responses,
paired claims and uncertainty remain private; no annotation text is repaired.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

from .author_comparison_corpus import file_sha
from .author_control_replication import verify_files
from .human_prose_review import normalized, occurrence_span, schema_errors
from .relation_pilot import stage2_payload
from .signal import distribution, overlap
from fama_style.rhetoric.experiment import fresh, write_json, write_lines

PROTOCOL = 'relation_pilot_stage1_review_v1'


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def check_row(row, expected, schema):
    """Return mechanical errors, independently of the model's self-check labels."""
    errors = schema_errors(row, schema)
    if errors:
        return errors
    def error(code, **details): errors.append({'code': code, **details})
    if row['paragraph_id'] != expected['paragraph_id']:
        error('paragraph_id_mismatch')
    if row['original_paragraph'] != expected['original_paragraph']:
        error('original_changed')
    claims = row['claims']
    ids = [c['claim_id'] for c in claims]
    if not ids or ids != ['C' + str(i + 1) for i in range(len(ids))]:
        error('claim_sequence_or_empty')
    for sk, tk, prefix, ek in (
        ('original_sentences', 'original_paragraph', 'O', 'original_evidence'),
        ('neutral_sentences', 'plain_english', 'S', 'neutral_evidence'),
    ):
        text = row[tk] or ''
        ss = row[sk]
        if [s['sentence_id'] for s in ss] != [prefix + str(i + 1) for i in range(len(ss))]:
            error('sentence_sequence', side=tk)
        if normalized(' '.join(s['exact_text'] for s in ss)) != normalized(text):
            error('sentence_reconstruction', side=tk)
        # Full expression and mere back-reference must not be conflated.
        bound = set()
        for s in ss:
            full, refs = s['claim_ids'], s['reference_only_claim_ids']
            if len(set(full)) != len(full) or len(set(refs)) != len(refs):
                error('duplicate_sentence_binding', sentence=s['sentence_id'])
            if set(full) & set(refs) or (set(full) | set(refs)) - set(ids):
                error('invalid_sentence_binding', sentence=s['sentence_id'])
            bound.update(full)
        for c in claims:
            required = tk == 'original_paragraph' or c['alignment'] == 'preserved'
            if required and (not c[ek] or c['claim_id'] not in bound):
                error('missing_claim_evidence_or_binding', claim=c['claim_id'], side=tk)
            for evidence in c[ek]:
                if occurrence_span(text, evidence) is None:
                    error('invalid_exact_evidence', claim=c['claim_id'], side=tk)
    if row['rewrite_status'] == 'already_clear' and row['plain_english'] != row['original_paragraph']:
        error('already_clear_text_changed')
    return errors


def eligibility(row, errors):
    if errors:
        return 'mechanical_errors'
    if not isinstance(row['plain_english'], str) or not row['plain_english'].strip():
        return 'no_neutral_text'
    if row['rewrite_status'] == 'pending_review':
        return 'pending_rewrite'
    if any(c['alignment'] != 'preserved' for c in row['claims']):
        return 'nonpreserved_claim_annotation'
    return 'eligible_for_source_only_annotation_not_training'


def file_manifest(folder, **metadata):
    write_json(folder / 'manifest.json', {**metadata, 'files': {
        str(p.relative_to(folder)): file_sha(p) for p in sorted(folder.rglob('*'))
        if p.is_file() and p.name != 'manifest.json'}})


def audit(packet, output, session_path):
    """Fail on incomplete/misplaced IDs; retain row-level problems without repairs."""
    verify_files(packet)
    schema = json.loads((packet / 'common/stage1_output.schema.json').read_text())
    session = json.loads(session_path.read_text())
    rows, details, originals = [], [], {}
    paths = sorted((packet / 'responses/stage1').glob('responses_*.jsonl'))
    if [p.name for p in paths] != ['responses_%02d.jsonl' % i for i in range(1, 9)]:
        raise ValueError('eight_response_files_required')
    seen = set()
    for batch, path in enumerate(paths, 1):
        supplied = read_rows(packet / ('stage1/batch_%02d/paragraphs.jsonl' % batch))
        expected = {r['paragraph_id']: r for r in supplied}
        received = read_rows(path)
        returned = [r.get('paragraph_id') for r in received]
        if len(received) != len(expected) or len(set(returned)) != len(returned) or set(returned) != set(expected) or seen & set(returned):
            raise ValueError('missing_duplicate_or_wrong_batch_ids')
        seen.update(returned)
        for row in received:
            pid = row['paragraph_id']; errors = check_row(row, expected[pid], schema)
            # Schema errors are not safe to project or summarize as valid rows.
            if schema_errors(row, schema):
                raise ValueError('response_schema_errors:' + pid)
            details.append({'paragraph_id': pid, 'batch': batch, 'errors': errors,
                'stage2_status': eligibility(row, errors),
                'needs_review': row['needs_review'], 'needs_external_context': row['needs_external_context'],
                'review_notes': row['review_notes'],
                'original_sentences': len(row['original_sentences']),
                'neutral_sentences': len(row['neutral_sentences']),
                'sequence_overlap': overlap(row['plain_english'] or '', row['original_paragraph'])})
            rows.append(row); originals[pid] = expected[pid]
    fresh(output)
    raw = output / 'raw_stage1'; raw.mkdir()
    for path in sorted((packet / 'responses/stage1').iterdir()):
        if path.is_file(): shutil.copyfile(path, raw / path.name)
    shutil.copyfile(packet / 'SESSION_NOTES.md', output / 'original_session_notes.md')
    shutil.copyfile(session_path, output / 'user_reported_session.json')
    shutil.copyfile(packet / 'PRIVATE_SELECTION.json', output / 'PRIVATE_SELECTION.json')
    write_lines(output / 'annotations.private.jsonl', rows)
    write_lines(output / 'row_audit.private.jsonl', details)
    n_original = sum(d['original_sentences'] for d in details)
    n_neutral = sum(d['neutral_sentences'] for d in details)
    summary = {'protocol': PROTOCOL, 'records': len(rows), 'response_files': len(paths),
        'exact_original_matches': sum(r['original_paragraph'] == originals[r['paragraph_id']]['original_paragraph'] for r in rows),
        'mechanically_valid_records': sum(not d['errors'] for d in details),
        'rewrite_status': dict(Counter(r['rewrite_status'] for r in rows)),
        'claim_alignment_self_annotations': dict(Counter(c['alignment'] for r in rows for c in r['claims'])),
        'needs_review': sum(r['needs_review'] for r in rows),
        'needs_external_context': sum(r['needs_external_context'] for r in rows),
        'stage2_status': dict(Counter(d['stage2_status'] for d in details)),
        'original_sentences': n_original, 'neutral_sentences': n_neutral,
        'eligible_source_boundaries': sum(d['neutral_sentences'] - 1 for d in details if d['stage2_status'].startswith('eligible_')),
        'sentence_count_change': dict(Counter('more' if d['neutral_sentences'] > d['original_sentences'] else 'fewer' if d['neutral_sentences'] < d['original_sentences'] else 'same' for d in details)),
        'unchanged_neutral_texts': sum(r['plain_english'] == r['original_paragraph'] for r in rows),
        'lexical_sequence_overlap': distribution([d['sequence_overlap'] for d in details]),
        'exact_evidence_anchors_checked': sum(len(c[k]) for r in rows for c in r['claims'] for k in ('original_evidence', 'neutral_evidence')),
        'raw_provenance_null_records': sum(r['provenance']['visible_model_label'] is None and r['provenance']['thinking_setting'] is None for r in rows),
        'reported_model': session['visible_model_label'], 'reported_effort': session['thinking_setting'],
        'mechanical_check_is_semantic_certificate': False, 'annotations_repaired': 0,
        'reserved_content_used': False, 'gpu_jobs_started': 0, 'training_called': False}
    write_json(output / 'summary.json', summary)
    file_manifest(output, protocol=PROTOCOL, code_sha256=file_sha(Path(__file__)),
        input_packet_manifest_sha256=file_sha(packet / 'manifest.json'),
        input_schema_sha256=file_sha(packet / 'common/stage1_output.schema.json'))
    return summary


def prepare_stage2(packet, audited, output):
    """Freeze mixed-paper batches from audited actual neutral text, never C-claims."""
    verify_files(packet); manifest = verify_files(audited)
    if manifest['input_packet_manifest_sha256'] != file_sha(packet / 'manifest.json'):
        raise ValueError('audit_packet_mismatch')
    rows = {r['paragraph_id']: r for r in read_rows(audited / 'annotations.private.jsonl')}
    details = {r['paragraph_id']: r for r in read_rows(audited / 'row_audit.private.jsonl')}
    selection = json.loads((audited / 'PRIVATE_SELECTION.json').read_text())
    bypaper = {}
    held = []
    for r in sorted(selection, key=lambda r: (r['paper_id'], r['paragraph_id'])):
        pid = r['paragraph_id']
        if details[pid]['stage2_status'].startswith('eligible_'):
            bypaper.setdefault(r['paper_id'], []).append(pid)
        else:
            held.append({'paragraph_id': pid, 'reason': details[pid]['stage2_status']})
    # Round-robin mixes papers, without exporting author, strata or uncertainty hints.
    ordered = []
    for i in range(max(map(len, bypaper.values()), default=0)):
        for paper in sorted(bypaper):
            if i < len(bypaper[paper]): ordered.append(bypaper[paper][i])
    fresh(output); (output / 'common').mkdir(); (output / 'responses').mkdir()
    shutil.copyfile(packet / 'common/stage2_output.schema.json', output / 'common/stage2_output.schema.json')
    prompt = (packet / 'common/STAGE2_PROMPT.md').read_text()
    shutil.copyfile(packet / 'common/STAGE2_PROMPT.md', output / 'common/STAGE2_PROMPT.md')
    mapping = []
    for offset in range(0, len(ordered), 5):
        batch = offset // 5 + 1; ids = ordered[offset:offset+5]
        folder = output / ('batches/batch_%02d' % batch); folder.mkdir(parents=True)
        write_lines(folder / 'paragraphs.jsonl', [stage2_payload(rows[pid]) for pid in ids])
        (folder / 'TASK.md').write_text(prompt + '\n\nProcess all %d attached paragraphs using stage2_output.schema.json. '
            'Save responses_%02d.jsonl and coverage_%02d.md. Extract fresh D-claims from the source alone.\n' % (len(ids), batch, batch))
        for pid in ids:
            mapping.append({'paragraph_id': pid, 'batch': batch,
                'source_sha256': hashlib.sha256(rows[pid]['plain_english'].encode()).hexdigest(),
                'stage1_needs_review': rows[pid]['needs_review'],
                'stage1_needs_external_context': rows[pid]['needs_external_context']})
    write_json(output / 'PRIVATE_MAPPING.json', mapping)
    write_json(output / 'held_back.private.json', held)
    summary = {'protocol': 'relation_pilot_v1_stage2_package', 'paragraphs': len(ordered),
        'batches': (len(ordered) + 4) // 5, 'held_back': len(held),
        'payload_fields': ['paragraph_id', 'source_paragraph'],
        'paired_claims_exported': False, 'reference_fields_exported': False,
        'independent_context_guaranteed': False, 'training_called': False, 'gpu_jobs_started': 0}
    write_json(output / 'summary.json', summary)
    (output / 'START_HERE.md').write_text('''# Stage 2: source-only relation and boundary annotations

Use GPT-6 with High thinking effort, consistent with Stage 1. For each folder in
`batches/`, paste its TASK.md and attach ONLY its paragraphs.jsonl and
common/stage2_output.schema.json. There are eight batches: seven of five records
and one of four. Save the first complete unedited responses_01.jsonl through
responses_08.jsonl and coverage_XX.md in this packet's responses/ folder.

Do not attach this file, private mapping, held-back records, Stage 1 responses,
originals, papers, previous model results, claim alignments or other batch answers.
The payload contains the actual neutral text and opaque ID only. Fresh D-claims
must come from that text. Use BOTH and UNRESOLVED when appropriate; do not force
class balance or assume one reference layout is uniquely correct. MERGE/BOTH
requires a concrete candidate whose fidelity remains unverified.

One pending rewrite is held offline. Other context/review flags remain offline
for later analysis; these paragraphs are eligible for annotation, not approved
training data. Source-only input cannot repair information absent from its text.
One already-clear neutral paragraph is unchanged; it remains a valid source input.

Separate chats may reduce direct carryover, but automatic retrieval/pretraining
can prevent independence. We do not certify hidden context isolation. Report any
changes to model/effort or retries when you return the files; do not share credentials.
Notify Codex after all eight batches are ready. No Kaggle job is required.
''')
    file_manifest(output, protocol=summary['protocol'], code_sha256=file_sha(Path(__file__)),
        audit_manifest_sha256=file_sha(audited / 'manifest.json'))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--audit-output', type=Path, required=True)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--stage2-output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.packet, args.audit_output, args.session)
    result['stage2'] = prepare_stage2(args.packet, args.audit_output, args.stage2_output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
