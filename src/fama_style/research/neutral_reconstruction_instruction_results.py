"""Import prompt-only repair outputs; preserve raw data and explicit review limits."""
import argparse
import json
from pathlib import Path
import shutil
import statistics

from .author_control_replication import verify_files
from .neutral_reconstruction import write_json, write_rows
from .neutral_reconstruction_instruction import rows, EXPECTED_IDS
from .neutral_reconstruction_results import file_sha, metrics, validate_output

PROTOCOL = 'neutral_reconstruction_instruction_results_v1'


def validate_received(received, expected):
    ids = [r['record_id'] for r in received]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError('response_coverage_or_duplicates')
    for row in received:
        validate_output(row, 'neutral_reconstruction')


def build(packet, response, review, output):
    if output.exists():
        raise ValueError('refuse_overwrite_results')
    verify_files(packet)
    expected = json.loads((packet/'protocol.json').read_text())['record_ids']
    if set(expected) != EXPECTED_IDS:
        raise ValueError('unexpected_development_cases')
    received = rows(response)
    validate_received(received, expected)
    baseline = {r['record_id']: r for r in rows(packet/'private_key/baseline.private.jsonl')}
    assessments = rows(review/'assessments.private.jsonl')
    judged = {r['record_id']: r for r in assessments}
    if len(judged) != len(assessments) or set(judged) != set(expected):
        raise ValueError('assessment_coverage')
    result = []
    for row in sorted(received, key=lambda r: r['record_id']):
        rid = row['record_id']; old = baseline[rid]; assessment = judged[rid]
        if assessment['response_sha256'] != file_sha(response):
            raise ValueError('review_response_binding_changed')
        generated = row['status'] == 'generated'
        if assessment['assessed_status'] != row['status']:
            raise ValueError('assessment_status_mismatch')
        m = metrics(old['original_paragraph'], row['text']) if generated else None
        result.append(row | {'original_paragraph': old['original_paragraph'],
            'previous_text': old['text'], 'previous_metrics': old['metrics'], 'metrics': m,
            'assessment': assessment, 'training_ready': False,
            'overlap_change': m['word_sequence_overlap']-old['metrics']['word_sequence_overlap'] if generated else None})
    generated = [r for r in result if r['status'] == 'generated']
    summary = {'protocol': PROTOCOL, 'status': 'executed_responses_reviewed',
        'expected': len(expected), 'generated': len(generated), 'blocked': len(result)-len(generated),
        'previous_added_commentary': len(result),
        'new_added_commentary': sum(r['assessment']['added_epistemic_metacommentary'] for r in generated),
        'material_errors_identified': sum(r['assessment']['material_claim_error_identified'] for r in generated),
        'outputs_with_local_review_flags': sum(bool(r['assessment']['local_review_flags']) for r in generated),
        'readable': sum(r['assessment']['readable'] for r in generated),
        'previous_median_overlap': statistics.median(r['previous_metrics']['word_sequence_overlap'] for r in result),
        'new_median_overlap': statistics.median(r['metrics']['word_sequence_overlap'] for r in generated) if generated else None,
        'cases_with_increased_overlap': sum(r['overlap_change'] > 0 for r in generated),
        'added_numeric_values': sum(bool(r['metrics']['added_numeric_values']) for r in generated),
        'missing_numeric_values': sum(bool(r['metrics']['missing_numeric_values']) for r in generated),
        'limitations': ['Three failure-selected development cases, not a new held-out sample.',
            'Qualitative automated review is unblinded and not independent semantic verification.',
            'GPT-6 High effort is user-reported; UI decoding and memory/retrieval uncontrolled.',
            'No detector, human-authorship or Fama-style evaluation; no training-ready certification.'],
        'training_ready': False, 'gpu_jobs_submitted': 0}
    output.mkdir(parents=True)
    write_rows(output/'per_output.private.jsonl', result)
    write_json(output/'summary.json', summary)
    shutil.copyfile(response, output/'raw_response.private.jsonl')
    write_json(output/'receipt.json', {'source_response_name': response.name,
        'response_sha256': file_sha(response), 'packet_manifest_sha256': file_sha(packet/'manifest.json'),
        'assessment_sha256': file_sha(review/'assessments.private.jsonl'),
        'provenance_sha256': file_sha(review/'provenance.private.json')})
    write_json(output/'manifest.json', {'protocol': PROTOCOL, 'code_sha256': file_sha(Path(__file__)),
        'files': {str(p.relative_to(output)): file_sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('packet', 'response', 'review', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.packet, args.response, args.review, args.output)))


if __name__ == '__main__':
    main()
