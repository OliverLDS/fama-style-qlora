"""Project unchanged v4 briefs into two matched manual reconstruction jobs."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from fama_style.research.content_brief_v4 import validate_brief_v4

SPEC = importlib.util.spec_from_file_location('abstraction_pilot', Path(__file__).with_name('prepare_content_abstraction_pilot_v4.py'))
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)
m = p.m
OUT = p.OUT / 'stage_2_reconstruction_v1'
RAW = p.OUT / 'responses/stage_1_briefs/briefs_01.jsonl'
PROMPT = ROOT / 'docs/data_scaling_v1/BRIEF_RECONSTRUCTION_V4.md'


def project_unchanged(row):
    """No repairs, note sorting, edge sorting, label changes or anchor changes."""
    validate_brief_v4(row)
    common = dict(id=row['id'], target_sha256=row['target_sha256'],
                  brief_status=row['status'], brief_notes=copy.deepcopy(row['notes']))
    a = copy.deepcopy(common)
    a['brief'] = {key: copy.deepcopy(row['brief'][key]) for key in ('topic', 'key_points', 'anchors')}
    b = copy.deepcopy(common)
    b['brief'] = copy.deepcopy(row['brief'])
    return dict(content_only=a, purpose_and_relations=b)


def load():
    p.check()
    rows = m.lines(RAW)
    inputs = m.lines(p.OUT / 'stage_1_brief_batches/inputs_01.jsonl')
    if [r.get('id') for r in rows] != [r['id'] for r in inputs]:
        raise ValueError('brief_identity_or_order')
    for row, original in zip(rows, inputs):
        validate_brief_v4(row)
        if row['target_sha256'] != original['target_sha256']:
            raise ValueError('brief_target_hash')
    return rows, inputs


def prepare():
    rows, originals = load()
    if OUT.exists():
        raise ValueError('refuse_overwrite_reconstruction_packet')
    prompt = PROMPT.read_text()
    OUT.mkdir()
    jobs = []
    for letter, condition, label in [('A', 'content_only', 'content_only'), ('B', 'purpose_and_relations', 'guided')]:
        payloads = [project_unchanged(r)[condition] for r in rows]
        data_name = f'inputs_{letter}_01.jsonl'
        batch_name = f'batch_{letter}_{label}.md'
        response = f'rewrites_{letter}_01.jsonl'
        (OUT / data_name).write_text(m.jsonl(payloads))
        (OUT / batch_name).write_text(prompt + f'\nOutput data filename: {response}\nExpected records: 5\n\nINPUTS:\n```jsonl\n' + m.jsonl(payloads) + '```\n')
        jobs.append(dict(condition=condition, records=len(rows), ids=[r['id'] for r in rows],
                         inputs=data_name, attachment=batch_name, response=response))
    m.dump(OUT / 'jobs.json', jobs)
    (OUT / 'START_HERE.md').write_text('''# Reconstruct the five pairs in two conditions

The user's five briefs are used unchanged. No suggested P484 adjustments, note
sorting, edge reordering or purpose-label changes have been applied.

Upload each job separately to GPT-6 with requested High thinking effort:

1. batch_A_content_only.md -> rewrites_A_01.jsonl (five paragraphs).
2. batch_B_guided.md -> rewrites_B_01.jsonl (the same five paragraphs).

Each attachment contains the identical base reconstruction prompt and its inputs.
A contains the original topic, notes and anchors. B adds the exact original purpose
and relationship fields. All shared values and their ordering are identical.

Save both returned data files in:
../responses/stage_2_rewrites/
That is the content_abstraction_pilot_v4/responses/stage_2_rewrites/ folder.

Use separate requests, preferably fresh chats without Stage 1 content, human
paragraphs, previous rewrites or the other condition's outputs. No additional
attachments are needed. Cross-chat memory isolation is not assumed or certified.
Keep the first result unchanged while collecting the second. Both jobs use the
same requested model/effort; UI decoding settings cannot be pinned or verified.

This generates ten outputs for five matched cases. Compare communicative function,
organization, copying and material claim changes after both return. Five examples
provide a diagnostic, not a general performance estimate or author-style claim.
No inference calls or GPU jobs have been executed by preparing these files.
''')
    m.dump(OUT / 'summary.json', dict(brief_records=len(rows), reconstruction_jobs=2, expected_outputs=10,
        brief_output_edits=0, brief_order_changes=0, shared_content_identical=True,
        purpose_labels_unchanged=True, relationship_labels_unchanged=True,
        user_requested_no_output_edits=True, raw_briefs_sha256=m.digest(RAW),
        generated_outputs=0, provider_calls=0, training_calls=0))
    dependencies = [p.OUT / 'manifest.json', RAW, PROMPT,
        ROOT / 'src/fama_style/research/content_brief.py',
        ROOT / 'src/fama_style/research/content_brief_v3.py',
        ROOT / 'src/fama_style/research/content_brief_v4.py']
    m.dump(OUT / 'manifest.json', dict(builder_sha256=m.digest(Path(__file__)),
        dependencies={str(f.relative_to(ROOT)): m.digest(f) for f in dependencies},
        files={str(f.relative_to(OUT)): m.digest(f) for f in sorted(OUT.iterdir()) if f.is_file()}))
    return check()


def check():
    rows, originals = load()
    manifest = m.read(OUT / 'manifest.json')
    if m.digest(Path(__file__)) != manifest['builder_sha256']:
        raise ValueError('reconstruction_builder_changed')
    for name, digest in manifest['dependencies'].items():
        if m.digest(ROOT / name) != digest:
            raise ValueError('reconstruction_dependency_changed')
    for name, digest in manifest['files'].items():
        if m.digest(OUT / name) != digest:
            raise ValueError('reconstruction_packet_changed')
    for job in m.read(OUT / 'jobs.json'):
        payloads = m.lines(OUT / job['inputs'])
        if payloads != [project_unchanged(r)[job['condition']] for r in rows]:
            raise ValueError('unchanged_projection_failed')
        packet = (OUT / job['attachment']).read_text()
        if not packet.startswith(PROMPT.read_text()):
            raise ValueError('base_prompt_changed')
        for payload in payloads:
            if set(payload) != {'id', 'target_sha256', 'brief', 'brief_status', 'brief_notes'}:
                raise ValueError('reconstruction_payload_allowlist')
        if any(r['original_paragraph'] in packet for r in originals):
            raise ValueError('original_paragraph_leaked')
    return m.read(OUT / 'summary.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'check'])
    args = parser.parse_args()
    print(json.dumps(prepare() if args.action == 'prepare' else check(), indent=2))
