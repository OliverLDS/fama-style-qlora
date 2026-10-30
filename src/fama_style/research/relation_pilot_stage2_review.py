"""CPU audit of source-only relation judgments; candidates remain unverified.

No reference layouts are scored and no text, claims, verdicts or provenance are
repaired. Surface risk flags prioritize inspection, not semantic accept/reject.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil

from .author_control_replication import verify_files
from .human_prose_review import normalized, occurrence_span, schema_errors
from .relation_pilot_review import file_manifest, file_sha, read_rows
from .signal import distribution, overlap
from fama_style.rhetoric.experiment import fresh, write_json, write_lines

PROTOCOL = 'relation_pilot_stage2_review_v1'
NUMBERS = re.compile(r'(?<!\w)\d+(?:[.,⁄/]\d+)*')
CAUSAL = re.compile(r'\b(?:because|since|therefore|thus|hence|consequently|thereby)\b', re.I)
GUARDS = re.compile(r'\b(?:not|no|never|without|may|might|could|would|must|only|all|every|some|most|if|unless)\b', re.I)


def sentence_spans(text, sentences):
    """Locate exact strings sequentially, allowing only whitespace between them."""
    result = {}; offset = 0
    for sentence in sentences:
        start = text.find(sentence['exact_text'], offset)
        if start < 0 or text[offset:start].strip():
            return None
        end = start + len(sentence['exact_text'])
        result[sentence['sentence_id']] = (start, end); offset = end
    return result if not text[offset:].strip() else None


def check_row(row, expected, schema):
    errors = schema_errors(row, schema)
    if errors: return errors
    def error(code, **detail): errors.append({'code': code, **detail})
    if row['paragraph_id'] != expected['paragraph_id']: error('paragraph_id_mismatch')
    text = row['source_paragraph']
    if text != expected['source_paragraph']: error('source_changed')
    sentences = row['source_sentences']; claims = row['source_claims']; boundaries = row['boundaries']
    sids = [s['sentence_id'] for s in sentences]
    cids = [c['claim_id'] for c in claims]
    if not sentences or sids != ['S'+str(i+1) for i in range(len(sentences))]: error('sentence_sequence')
    if not claims or cids != ['D'+str(i+1) for i in range(len(claims))]: error('claim_sequence')
    spans = sentence_spans(text, sentences)
    if spans is None or normalized(' '.join(s['exact_text'] for s in sentences)) != normalized(text):
        error('sentence_reconstruction')
    def anchored(evidence, location=None):
        anchors = [occurrence_span(text, e) for e in evidence]
        if not evidence or any(a is None for a in anchors): return False
        return location is None or any(a[0] < location[1] and a[1] > location[0] for a in anchors)
    claim_lookup = {c['claim_id']: c for c in claims}; bound = set()
    for claim in claims:
        if not claim['proposition'].strip(): error('empty_claim', claim=claim['claim_id'])
        if not anchored(claim['source_evidence']): error('claim_evidence', claim=claim['claim_id'])
    for s in sentences:
        full, refs = s['claim_ids'], s['reference_only_claim_ids']
        if len(set(full)) != len(full) or len(set(refs)) != len(refs) or set(full) & set(refs) or (set(full) | set(refs)) - set(cids):
            error('invalid_binding', sentence=s['sentence_id'])
        bound.update(full)
        if spans:
            for cid in set(full) & set(cids):
                if not anchored(claim_lookup[cid]['source_evidence'], spans[s['sentence_id']]):
                    error('binding_evidence_outside_sentence', sentence=s['sentence_id'], claim=cid)
    if set(cids) - bound: error('unbound_claims')
    if len(boundaries) != max(0, len(sentences)-1): error('boundary_coverage')
    for i, b in enumerate(boundaries):
        if b['boundary_id'] != 'B'+str(i+1): error('boundary_sequence')
        pair = [b['left_sentence_id'], b['right_sentence_id']]
        if pair != sids[i:i+2] or len(pair) != len(sids[i:i+2]): error('nonadjacent_boundary', boundary=b['boundary_id'])
        if not anchored(b['source_evidence']): error('boundary_evidence', boundary=b['boundary_id'])
        if spans:
            for sid in pair:
                if sid in spans and not anchored(b['source_evidence'], spans[sid]):
                    error('boundary_evidence_missing_side', boundary=b['boundary_id'], sentence=sid)
        candidate = b['merge_candidate']
        if b['permissible'] in ('MERGE', 'BOTH') and (not candidate or not candidate.strip()):
            error('missing_merge_candidate', boundary=b['boundary_id'])
        allowed = {'KEEP': {'KEEP', 'NONE'}, 'MERGE': {'MERGE', 'NONE'},
                   'BOTH': {'KEEP', 'MERGE', 'NONE'}, 'UNRESOLVED': {'UNRESOLVED', 'NONE'}}
        if b['presentation_preference'] not in allowed[b['permissible']]:
            error('preference_not_permitted', boundary=b['boundary_id'])
    return errors


def candidate_diagnostics(left, right, candidate):
    """All flags are risk triage: repeated actors and paraphrases can trigger them."""
    source = left + ' ' + right
    numbers = lambda t: Counter(NUMBERS.findall(t))
    guards = lambda t: Counter(m.group().casefold() for m in GUARDS.finditer(t))
    flags = []
    if numbers(source) != numbers(candidate): flags.append('number_inventory_changed')
    if guards(source) != guards(candidate): flags.append('guard_inventory_changed')
    if not CAUSAL.search(source) and CAUSAL.search(candidate): flags.append('new_explicit_causal_cue')
    # Abbreviations must not manufacture a two-sentence warning.
    masked = re.sub(r'\b(?:Sec|Fig|Dr|Mr|Mrs)\.|\bet al\.|\bi\.e\.|\be\.g\.', '', candidate, flags=re.I)
    if re.search(r'[.!?][”\'"’)]*\s+[A-Z]', masked): flags.append('possible_multiple_sentences')
    words = len(candidate.split())
    if words > 60: flags.append('over_60_words')
    return {'risk_flags': flags, 'word_count': words, 'sequence_overlap_with_pair': overlap(source, candidate),
            'semicolon': ';' in candidate, 'semantic_status': 'unverified'}


def audit(packet, output, session_path):
    verify_files(packet)
    schema = json.loads((packet/'common/stage2_output.schema.json').read_text())
    batches = sorted((packet/'batches').glob('batch_*'))
    paths = sorted((packet/'responses').glob('responses_*.jsonl'))
    if [p.name for p in paths] != ['responses_%02d.jsonl' % i for i in range(1, len(batches)+1)]:
        raise ValueError('response_file_coverage')
    session = json.loads(session_path.read_text())
    rows = []; details = []; candidates = []; seen = set(); batch_summaries = []
    for batch, path in enumerate(paths, 1):
        expected = {r['paragraph_id']: r for r in read_rows(packet/('batches/batch_%02d/paragraphs.jsonl' % batch))}
        received = read_rows(path); ids = [r.get('paragraph_id') for r in received]
        if len(ids) != len(expected) or len(set(ids)) != len(ids) or set(ids) != set(expected) or seen & set(ids):
            raise ValueError('missing_duplicate_or_wrong_batch_ids')
        seen.update(ids)
        for row in received:
            pid = row['paragraph_id']; errors = check_row(row, expected[pid], schema)
            if schema_errors(row, schema): raise ValueError('schema_errors:' + pid)
            details.append({'paragraph_id': pid, 'batch': batch, 'errors': errors,
                'mechanically_valid': not errors, 'needs_review': row['needs_review'],
                'needs_external_context': row['needs_external_context'], 'review_notes': row['review_notes']})
            # Invalid rows are retained, but must not produce derived candidates.
            if not errors:
                ss = {s['sentence_id']: s['exact_text'] for s in row['source_sentences']}
                for b in row['boundaries']:
                    if b['merge_candidate']:
                        candidates.append({'paragraph_id': pid, 'boundary_id': b['boundary_id'],
                            'presentation_preference': b['presentation_preference'],
                            **candidate_diagnostics(ss[b['left_sentence_id']], ss[b['right_sentence_id']], b['merge_candidate'])})
            rows.append(row)
        batch_summaries.append({'batch': batch, 'records': len(received),
            'permissible': dict(Counter(b['permissible'] for r in received for b in r['boundaries'])),
            'preference': dict(Counter(b['presentation_preference'] for r in received for b in r['boundaries']))})
    fresh(output); raw = output/'raw_stage2'; raw.mkdir()
    for path in sorted((packet/'responses').iterdir()):
        if path.is_file(): shutil.copyfile(path, raw/path.name)
    shutil.copyfile(session_path, output/'user_reported_session.json')
    write_lines(output/'annotations.private.jsonl', rows)
    write_lines(output/'row_audit.private.jsonl', details)
    write_lines(output/'candidate_diagnostics.private.jsonl', candidates)
    boundaries = [b for row in rows for b in row['boundaries']]
    valid = [r for r, d in zip(rows, details) if d['mechanically_valid']]
    counters = {key: dict(Counter(b[key] for b in boundaries)) for key in
                ('permissible', 'presentation_preference', 'relation', 'confidence')}
    summary = {'protocol': PROTOCOL, 'records': len(rows), 'response_files': len(paths),
        'mechanically_valid_records': len(valid), 'invalid_records': len(rows)-len(valid),
        'source_text_matches': sum(r['source_paragraph'] == next(e['source_paragraph'] for e in read_rows(
            packet/('batches/batch_%02d/paragraphs.jsonl' % d['batch'])) if e['paragraph_id'] == r['paragraph_id']) for r, d in zip(rows, details)),
        'sentences': sum(len(r['source_sentences']) for r in rows), 'boundaries': len(boundaries),
        'source_claims': sum(len(r['source_claims']) for r in rows), **counters,
        'needs_review': sum(r['needs_review'] for r in rows),
        'needs_external_context': sum(r['needs_external_context'] for r in rows),
        'all_keep_permitted_under_raw_judgments': bool(boundaries) and all(b['permissible'] in ('KEEP', 'BOTH') for b in boundaries),
        'candidate_count_from_valid_rows': len(candidates), 'candidate_semantic_status': 'all_unverified',
        'candidate_risk_flag_counts': dict(Counter(f for c in candidates for f in c['risk_flags'])),
        'candidates_with_surface_risk_flags': sum(bool(c['risk_flags']) for c in candidates),
        'preferred_merge_candidates_with_surface_risk_flags': sum(bool(c['risk_flags']) and c['presentation_preference']=='MERGE' for c in candidates),
        'candidate_words': distribution([c['word_count'] for c in candidates]) if candidates else None,
        'candidate_sequence_overlap': distribution([c['sequence_overlap_with_pair'] for c in candidates]) if candidates else None,
        'candidate_semicolons': sum(c['semicolon'] for c in candidates),
        'raw_thinking_setting_null_records': sum(r['provenance']['thinking_setting'] is None for r in rows),
        'reported_model': session['visible_model_label'], 'requested_effort': session['requested_thinking_setting'],
        'effective_effort': session['effective_thinking_setting'],
        'batch_summaries': batch_summaries,
        'labels_repaired': 0, 'approved_training_rows': 0, 'semantic_fidelity_certified': False,
        'reference_layouts_scored': False, 'reserved_content_used': False,
        'new_model_calls': 0, 'gpu_jobs_started': 0, 'training_called': False}
    write_json(output/'summary.json', summary)
    file_manifest(output, protocol=PROTOCOL, code_sha256=file_sha(Path(__file__)),
        input_packet_manifest_sha256=file_sha(packet/'manifest.json'))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('packet', 'output', 'session'): p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args(); print(json.dumps(audit(args.packet, args.output, args.session), indent=2))


if __name__ == '__main__': main()
