"""Five repaired-target pilot before approved full-cohort recollection; no calls."""
import argparse
import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location('shared_repairs', Path(__file__).with_name('repair_shared_targets_v2.py'))
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)
m = v.m
OUT = m.ROOT / 'research_runs/data_scaling_v1/purpose_brief_pilot_v3'
DOCS = m.ROOT / 'docs/data_scaling_v1'


def prepare():
    v.check()
    if OUT.exists():
        raise ValueError('refuse_overwrite_pilot')
    rows = [x for x in m.lines(v.OUT / 'targets_0500.private.jsonl') if x['id'] in v.PENDING]
    inputs = [dict(id=x['id'], target_sha256=x['target_sha256'], original_paragraph=x['target']) for x in rows]
    OUT.mkdir()
    jobs = OUT / 'stage_1_brief_batches'
    jobs.mkdir()
    (jobs / 'inputs_01.jsonl').write_text(m.jsonl(inputs))
    prompt = (DOCS / 'BRIEF_EXTRACTION_V3.md').read_text()
    (jobs / 'batch_01.md').write_text(prompt + '\nOutput filename: briefs_01.jsonl\nExpected records: 5\n\nINPUTS:\n```jsonl\n' + m.jsonl(inputs) + '```\n')
    m.dump(jobs / 'jobs.json', [dict(batch=1, ids=[x['id'] for x in inputs], records=5,
        attachment='batch_01.md', response='briefs_01.jsonl', brief_schema_version=3)])
    (OUT / 'brief_v3.schema.json').write_bytes((DOCS / 'brief_v3.schema.json').read_bytes())
    for name in ('stage_1_briefs', 'stage_2_rewrites'):
        (OUT / 'responses' / name).mkdir(parents=True)
    (OUT / 'START_HERE.md').write_text('''# Purpose-aware brief v3 pilot

Use this pilot instead of the earlier shared_target_repairs_v2 five-record packet.
Upload ONLY stage_1_brief_batches/batch_01.md to GPT-6, requesting High effort.
It includes all five corrected targets and the revised prompt with nuanced links.
Save briefs_01.jsonl in responses/stage_1_briefs/ under this pilot folder.
No model-setting confirmation or separate schema upload is required.

This is a pilot before recollecting briefs AND fresh reconstructions for ALL 500
targets, including the 495 already collected. Those earlier results remain baselines,
not substitutes for the approved new annotation round. Do not upload the full 500
yet; first check this prompt and its consequent reconstruction on the five cases.

After the briefs return, review the point content, purpose and relationship labels.
Then prepare two reconstruction conditions with identical task, points and anchors:
A includes those content fields only; B also includes purpose and relationships.
Both receive the same base reconstruction instruction and requested model settings.
This requires no second extraction job. The common fields must contain all substantive
content, so the comparison does not give hidden new claims to condition B.

Do not use old prose as the control: P264 was incomplete and P291 was a different
paragraph. Do not attach human targets or Stage 1 files to reconstruction requests.
Cross-chat memory isolation and independent generation are not assumed.

Stage 2 jobs will be made only after actual reviewed briefs are available. Five
cases can diagnose whether guidance is usable, not establish a general benefit.
If the protocol is retained, reuse the five pilot responses in the new 500-target
round and collect the remaining 495; otherwise record the prompt change first.
No teacher API calls, paid inference, detector scans or training are started here.
''')
    m.dump(OUT / 'summary.json', dict(pilot_records=5, pilot_jobs=1, planned_full_cohort_recollection=500,
        existing_495_recollection_required=True, old_datasets_preserved=True, brief_schema_version=3,
        stage_2_status='awaiting_actual_reviewed_briefs', expected_stage_2_conditions=2,
        comparison='same_content_with_vs_without_explicit_purpose_and_relations',
        provider_calls=0, training_calls=0))
    deps = [v.OUT / 'manifest.json', v.OUT / 'targets_0500.private.jsonl',
            DOCS / 'BRIEF_EXTRACTION_V3.md', DOCS / 'BRIEF_RECONSTRUCTION_V3.md',
            DOCS / 'brief_v3.schema.json', DOCS / 'brief_relations_v3.json',
            m.ROOT / 'src/fama_style/research/content_brief.py',
            m.ROOT / 'src/fama_style/research/content_brief_v3.py']
    m.dump(OUT / 'manifest.json', dict(builder_sha256=m.digest(Path(__file__)),
        dependencies={str(p.relative_to(m.ROOT)): m.digest(p) for p in deps},
        files={str(p.relative_to(OUT)): m.digest(p) for p in sorted(OUT.rglob('*')) if p.is_file()}))
    return check()


def check():
    manifest = m.read(OUT / 'manifest.json')
    if manifest['builder_sha256'] != m.digest(Path(__file__)):
        raise ValueError('pilot_builder_changed')
    for name, digest in manifest['dependencies'].items():
        if m.digest(m.ROOT / name) != digest:
            raise ValueError('pilot_dependency_changed')
    for name, digest in manifest['files'].items():
        if m.digest(OUT / name) != digest:
            raise ValueError('pilot_packet_changed')
    targets = {x['id']: x for x in m.lines(v.OUT / 'targets_0500.private.jsonl')}
    rows = m.lines(OUT / 'stage_1_brief_batches/inputs_01.jsonl')
    if len(rows) != 5 or {x['id'] for x in rows} != v.PENDING:
        raise ValueError('pilot_coverage')
    for x in rows:
        if set(x) != {'id', 'target_sha256', 'original_paragraph'} or x['original_paragraph'] != targets[x['id']]['target'] or x['target_sha256'] != m.sha(x['original_paragraph']):
            raise ValueError('pilot_target_binding')
    prompt = (OUT / 'stage_1_brief_batches/batch_01.md').read_text().split('INPUTS:')[0]
    if 'hope' in prompt.lower():
        raise ValueError('removed_example_reintroduced')
    return m.read(OUT / 'summary.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'check'])
    args = parser.parse_args()
    print(json.dumps(prepare() if args.action == 'prepare' else check(), indent=2))
