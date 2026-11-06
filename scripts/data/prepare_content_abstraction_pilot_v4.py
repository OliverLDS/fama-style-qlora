"""Retry the same five human targets with compact, non-sequential brief abstraction."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from fama_style.research.content_brief_v3 import validate_brief_v3

SPEC = importlib.util.spec_from_file_location('purpose_pilot_v3', Path(__file__).with_name('prepare_purpose_brief_pilot_v3.py'))
p3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p3)
m = p3.m
OUT = ROOT / 'research_runs/data_scaling_v1/content_abstraction_pilot_v4'
DOCS = ROOT / 'docs/data_scaling_v1'
BASELINE = p3.OUT / 'responses/stage_1_briefs/brief_v3.schema.json'


def prepare():
    p3.check()
    if OUT.exists():
        raise ValueError('refuse_overwrite_abstraction_pilot')
    inputs_path = p3.OUT / 'stage_1_brief_batches/inputs_01.jsonl'
    inputs = m.lines(inputs_path)
    targets = {r['id']: r for r in inputs}
    baseline = m.lines(BASELINE)
    if len(baseline) != len(inputs) or [r['id'] for r in baseline] != [r['id'] for r in inputs]:
        raise ValueError('baseline_coverage')
    for row in baseline:
        validate_brief_v3(row)
        if row['target_sha256'] != targets[row['id']]['target_sha256']:
            raise ValueError('baseline_target_binding')
    OUT.mkdir()
    jobs = OUT / 'stage_1_brief_batches'
    jobs.mkdir()
    (jobs / 'inputs_01.jsonl').write_bytes(inputs_path.read_bytes())
    prompt = (DOCS / 'BRIEF_ABSTRACTION_V4.md').read_text()
    (jobs / 'batch_01.md').write_text(prompt + '\nOutput data filename: briefs_01.jsonl\nExpected records: 5\n\nINPUTS:\n```jsonl\n' + m.jsonl(inputs) + '```\n')
    m.dump(jobs / 'jobs.json', [dict(batch=1, ids=[r['id'] for r in inputs], records=5,
        attachment='batch_01.md', response='briefs_01.jsonl', brief_schema_version=4)])
    (OUT / 'brief_v4.schema.json').write_bytes((DOCS / 'brief_v4.schema.json').read_bytes())
    (OUT / 'baselines').mkdir()
    # The user's file is JSONL despite its schema-like name. Preserve bytes,
    # including the original at its original path, under an unambiguous copy name.
    (OUT / 'baselines/v3_briefs.private.jsonl').write_bytes(BASELINE.read_bytes())
    for name in ('stage_1_briefs', 'stage_2_rewrites'):
        (OUT / 'responses' / name).mkdir(parents=True)
    (OUT / 'START_HERE.md').write_text('''# Retry the five pairs: content abstraction v4

Upload ONLY stage_1_brief_batches/batch_01.md to GPT-6, requesting High effort.
It includes the revised prompt, all 36 relation definitions and the same five
corrected target texts/hashes used in v3. No prior briefs or schema attachment
are needed. Save the actual five data records as briefs_01.jsonl in:
responses/stage_1_briefs/ under this folder.

Prefer a fresh request without earlier brief responses. Cross-chat memory isolation
is not asserted or required. All original files and prior responses remain intact.
The earlier response named brief_v3.schema.json contains valid JSONL briefs; an
unchanged-byte copy is preserved under baselines/v3_briefs.private.jsonl here.

The new format removes repeated task/takeaway prose, uses subject-headed fragments
in alphabetical indexing order, and stores factual details once in an anchor
dictionary. Relationships retain content dependencies, not the original outline.
There is no rigid length or wording-change threshold.

After the five new briefs return, assess redundancy, anchor completeness, sentence
alignment and direction of content links. Local note sorting can renumber point
links without changing their meaning. Then prepare the two planned reconstruction
conditions from identical topic/points/anchors, with versus without purpose/links.
Stage 2 receives reviewed briefs only; no originals, old rewrites or this batch.

The eventual goal remains fresh briefs AND reconstructions for all 500 targets,
including the existing 495. Freeze the protocol after this pilot before preparing
the bulk round. Reuse these five if the retained protocol is unchanged.
No provider calls, GPU jobs, detector scans or fabricated outputs are made here.
''')
    m.dump(OUT / 'summary.json', dict(pilot_records=5, pilot_jobs=1, brief_schema_version=4,
        target_texts_unchanged_from_v3=True, baseline_briefs_preserved=5,
        planned_full_cohort_recollection=500, existing_495_recollection_required=True,
        stage_2_status='awaiting_actual_reviewed_v4_briefs', provider_calls=0, training_calls=0))
    dependencies = [p3.OUT / 'manifest.json', inputs_path, BASELINE,
        DOCS / 'BRIEF_ABSTRACTION_V4.md', DOCS / 'BRIEF_RECONSTRUCTION_V4.md', DOCS / 'brief_v4.schema.json',
        ROOT / 'src/fama_style/research/content_brief.py', ROOT / 'src/fama_style/research/content_brief_v3.py',
        ROOT / 'src/fama_style/research/content_brief_v4.py']
    m.dump(OUT / 'manifest.json', dict(builder_sha256=m.digest(Path(__file__)),
        dependencies={str(p.relative_to(ROOT)): m.digest(p) for p in dependencies},
        files={str(p.relative_to(OUT)): m.digest(p) for p in sorted(OUT.rglob('*')) if p.is_file()}))
    return check()


def check():
    manifest = m.read(OUT / 'manifest.json')
    if manifest['builder_sha256'] != m.digest(Path(__file__)):
        raise ValueError('abstraction_builder_changed')
    for name, digest in manifest['dependencies'].items():
        if m.digest(ROOT / name) != digest:
            raise ValueError('abstraction_dependency_changed')
    for name, digest in manifest['files'].items():
        if m.digest(OUT / name) != digest:
            raise ValueError('abstraction_packet_changed')
    inputs = OUT / 'stage_1_brief_batches/inputs_01.jsonl'
    if inputs.read_bytes() != (p3.OUT / 'stage_1_brief_batches/inputs_01.jsonl').read_bytes():
        raise ValueError('pilot_targets_changed')
    if (OUT / 'baselines/v3_briefs.private.jsonl').read_bytes() != BASELINE.read_bytes():
        raise ValueError('raw_baseline_changed')
    rows = m.lines(inputs)
    if len(rows) != 5 or {r['id'] for r in rows} != p3.v.PENDING:
        raise ValueError('pilot_identity')
    if any(set(r) != {'id', 'target_sha256', 'original_paragraph'} or m.sha(r['original_paragraph']) != r['target_sha256'] for r in rows):
        raise ValueError('pilot_fields')
    return m.read(OUT / 'summary.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'check'])
    args = parser.parse_args()
    print(json.dumps(prepare() if args.action == 'prepare' else check(), indent=2))
