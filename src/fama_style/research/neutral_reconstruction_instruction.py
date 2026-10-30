"""Prepare a three-case, prompt-only development repair. No generation or training."""
import argparse
import difflib
import json
from pathlib import Path

from .author_control_replication import verify_files
from .neutral_reconstruction import write_json, write_rows
from .neutral_reconstruction_results import file_sha

PROTOCOL = 'neutral_reconstruction_instruction_check_v1'
EXPECTED_IDS = frozenset(('NR_0c1c81ebfc38e817', 'NR_04c5bff0d9cf6484', 'NR_3c16f9b49a6f9e84'))
RULE = '''Use modality, uncertainty, attribution and conditional fields to choose
appropriately scoped verbs and qualifications. These fields constrain expression;
do not turn annotation guardrails into additional narrative assertions about
absent evidence, unproven claims, empirical status or unmeasured quantities unless
the economic propositions themselves assert them. Preserve the actual uncertainty,
attribution and conditions: do not present hypotheses or expectations as findings.
'''
READY = '''Execution authorization: Stage 1 review and input freezing are complete.
The template prerequisites below are satisfied. Process the supplied three inputs
now; no additional prerequisite confirmation is needed.

'''


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def revised_prompt(base):
    anchor = 'If a missing fact, unresolved context gap or inconsistency prevents faithful\n'
    if base.count(anchor) != 1:
        raise ValueError('base_prompt_anchor_changed')
    return READY + base.replace(anchor, RULE + '\n' + anchor)


def select(payloads, observations, expected=EXPECTED_IDS):
    flagged = [r for r in observations if r['method'] == 'neutral_reconstruction'
               and any(f['type'] == 'added_epistemic_metacommentary'
                       for f in r['assessment']['local_review_flags'])]
    ids = [r['record_id'] for r in flagged]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError('recorded_failure_set_changed')
    by = {r['record_id']: r for r in payloads}
    if len(by) != len(payloads) or not set(ids) <= set(by):
        raise ValueError('payload_coverage')
    selected = [by[rid] for rid in sorted(ids)]
    if any(set(r) != {'record_id', 'content_brief'} for r in selected):
        raise ValueError('generator_payload_allowlist')
    return selected, sorted(flagged, key=lambda r: r['record_id'])


def build(root, packet, results, output):
    if output.exists():
        raise ValueError('refuse_overwrite_package')
    verify_files(packet)
    verify_files(results)
    payloads = [r for p in sorted((packet/'inputs/reconstruction').glob('inputs_*.jsonl')) for r in rows(p)]
    selected, baseline = select(payloads, rows(results/'per_output.private.jsonl'))
    base = (packet/'prompts/RECONSTRUCTION_PROMPT.md').read_text()
    revised = revised_prompt(base)
    output.mkdir(parents=True)
    for name in ('prompts', 'private_key', 'responses'):
        (output/name).mkdir()
    write_rows(output/'inputs_01.jsonl', selected)
    write_rows(output/'private_key/baseline.private.jsonl', baseline)
    (output/'prompts/RECONSTRUCTION_PROMPT.md').write_text(revised)
    (output/'prompts/base_prompt.md').write_text(base)
    (output/'prompts/prompt_changes.diff').write_text(''.join(difflib.unified_diff(
        base.splitlines(True), revised.splitlines(True), fromfile='base', tofile='instruction_check')))
    (output/'prompts/generation_output.schema.json').write_bytes((packet/'prompts/generation_output.schema.json').read_bytes())
    protocol = {
        'protocol': PROTOCOL, 'status': 'prepared_not_executed', 'expected_records': 3,
        'record_ids': [r['record_id'] for r in selected],
        'selection': 'All three previously recorded reconstruction commentary failures; post-hoc development repair.',
        'intervention': RULE, 'coordination_only_header': READY,
        'unchanged': ['content brief objects', 'record IDs', 'reference paragraphs', 'first outputs', 'output schema'],
        'requested_model': 'GPT-6', 'requested_effort': 'High thinking effort',
        'controls_limits': 'UI sampling/decoding controls unavailable; settings requested, not guaranteed identical. Memory/retrieval uncontrolled.',
        'comparison': 'Compare first returned outputs with frozen first reconstruction outputs and reviewed references; keep blocked cases.',
        'assessment': ['Added epistemic metacommentary', 'Economic claims, actors, modality, quantities, conditions, attribution and causal direction',
                       'Readability and organization', 'Same word overlap and longest shared run diagnostics'],
        'interpretation': 'Success requires commentary removal without material claim errors or lost qualifications; report copying changes per case. No generalization or Fama-style claim from three selected cases.',
        'no_actions': ['No GPU job', 'No training', 'No detector-based selection', 'No reference or baseline supplied to generator'],
        'save_response_as': 'responses/reconstruction_01.jsonl',
        'source_packet_manifest_sha256': file_sha(packet/'manifest.json'),
        'source_results_manifest_sha256': file_sha(results/'manifest.json'),
        'base_prompt_sha256': file_sha(packet/'prompts/RECONSTRUCTION_PROMPT.md'),
    }
    write_json(output/'protocol.json', protocol)
    write_json(output/'session_notes.json', {'requested_model': 'GPT-6', 'requested_effort': 'High thinking effort',
        'actual_model': None, 'actual_effort': None, 'conversation_arrangement': None,
        'note': 'Record observed settings after generation; separate chats do not guarantee independence.'})
    (output/'START_HERE.md').write_text('''# Three-case reconstruction instruction check

Stage 1 review and freezing are complete. This packet is ready to generate.
Use GPT-6 with High thinking effort, matching the previous reported settings.

In one generation task, supply only:
- prompts/RECONSTRUCTION_PROMPT.md
- inputs_01.jsonl
- prompts/generation_output.schema.json

Save the returned three-record UTF-8 JSONL as responses/reconstruction_01.jsonl.
Keep blocked rows. Return the first outputs unchanged; do not retry for preferred scores.
Do not upload private_key, base_prompt.md, prompt_changes.diff, original paragraphs,
PDFs or previous outputs. A new conversation is practical task separation, not
proof that account memory or retrieval is absent.

This is a targeted development repair, not a fresh held-out evaluation.
No training or Kaggle run is required.
''')
    write_json(output/'manifest.json', {'protocol': PROTOCOL, 'code_sha256': file_sha(Path(__file__)),
        'files': {str(p.relative_to(output)): file_sha(p) for p in sorted(output.rglob('*'))
                  if p.is_file() and p.name != 'session_notes.json'},
        'mutable_unhashed': ['session_notes.json', 'responses/']})
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.root, args.packet, args.results, args.output)
    print(json.dumps({'protocol': result['protocol'], 'expected_records': result['expected_records'], 'status': result['status']}))


if __name__ == '__main__':
    main()
