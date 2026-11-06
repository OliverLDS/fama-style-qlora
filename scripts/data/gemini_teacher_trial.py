"""Freeze eight training targets and call a free-tier Gemini project via inferencer."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'research_runs/gemini_teacher_trial_v1'
MODEL = 'gemini-3-flash-preview'
INSTRUCTION = ('Rewrite this paragraph in clear, neutral academic English for an economically literate reader. '
               'Simplify vocabulary and sentence structure where useful. Preserve the substantive claims, actors, '
               'quantities, conditions, uncertainty, negation, attribution and causal direction. '
               'Do not add explanations or examples. Return only the rewritten paragraph.\n\nParagraph:\n')


def sha(text): return hashlib.sha256(text.encode()).hexdigest()
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text())
def write(path, value): path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def select(rows):
    if len(rows) != 200 or any(r['split'] != 'training' or not r['training_ready'] for r in rows):
        raise ValueError('frozen_200_training_rows_required')
    groups = sorted({r['style_group'] for r in rows})
    if len(groups) != 4: raise ValueError('four_groups_required')
    selected = []
    for group in groups:
        candidates = sorted((r for r in rows if r['style_group'] == group), key=lambda r: sha('gemini-teacher-trial-v1|' + r['id']))
        # Two different papers per group; no selection by detector score.
        papers = set()
        for row in candidates:
            if row['paper_id'] in papers: continue
            papers.add(row['paper_id'])
            target = row['original_paragraph']
            selected.append(dict(example_id=row['id'], paper_id=row['paper_id'], style_group=group,
                split='training', original_paragraph=target, target_sha256=sha(target),
                previous_source_sha256=sha(row['plain_english']), prompt=INSTRUCTION + target,
                prompt_sha256=sha(INSTRUCTION + target)))
            if len(papers) == 2: break
        if len(papers) != 2: raise ValueError('two_papers_per_group_required')
    return selected


def prepare(output):
    if output.exists(): raise ValueError('refuse_overwrite_packet')
    source = ROOT / 'data/data_scaling_v1/stage_0200_salvage_v1/training.private.jsonl'
    manifest = read(source.parent / 'manifest.json')
    if digest(source) != manifest['outputs'][source.name]: raise ValueError('cohort_hash_changed')
    items = select([json.loads(line) for line in source.read_text().splitlines()])
    output.mkdir(parents=True)
    packet = output / 'inputs.private.jsonl'
    packet.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in items))
    write(output / 'config.json', dict(protocol='gemini_teacher_trial_v1', model=MODEL,
        packet_sha256=digest(packet), source_sha256=digest(source), input_rows=8,
        runner_sha256=digest(Path(__file__)), adapter_sha256=digest(Path(__file__).with_suffix('.R')),
        free_tier_only=True, free_project_confirmed_by_user=True, paid_fallback=False,
        max_requests=8, max_output_tokens=4096, temperature=.7, top_p=1, top_k=40,
        user_reported_GPT6_effort='High', existing_GPT6_data_preserved=True,
        training_authorized=False, detector_selection=False))
    print('Prepared eight training-side targets from eight papers; no calls made.')


def check(output):
    config = read(output / 'config.json'); packet = output / 'inputs.private.jsonl'
    if (config['model'] != MODEL or not config['free_tier_only'] or config['paid_fallback']
            or not config['free_project_confirmed_by_user'] or config['max_requests'] != 8
            or digest(packet) != config['packet_sha256']
            or digest(Path(__file__)) != config['runner_sha256']
            or digest(Path(__file__).with_suffix('.R')) != config['adapter_sha256']):
        raise ValueError('frozen_free_only_configuration_required')
    items = [json.loads(line) for line in packet.read_text().splitlines()]
    if len(items) != 8: raise ValueError('eight_inputs_required')
    if sorted(Counter(r['style_group'] for r in items).values()) != [2, 2, 2, 2]: raise ValueError('group_balance')
    for row in items:
        if row['split'] != 'training' or sha(row['original_paragraph']) != row['target_sha256']:
            raise ValueError('target_changed')
        if row['prompt'] != INSTRUCTION + row['original_paragraph'] or sha(row['prompt']) != row['prompt_sha256']:
            raise ValueError('prompt_changed')
    return config, items


def generate(output):
    config, items = check(output); responses = output / 'responses'; responses.mkdir(exist_ok=True)
    for i, row in enumerate(items):
        path = responses / f'response_{i+1:02d}.private.json'
        if path.exists():
            previous = read(path)
            if previous['example_id'] != row['example_id']: raise ValueError('resume_binding_changed')
            if previous['status'] == 'generated': continue
            raise ValueError('previous_failure_retained_inspect_before_new_version')
        print(f'Request {i+1}/8: free-tier Gemini via inferencer', flush=True)
        # No credential in command arguments; capture provider errors without printing URLs.
        result = subprocess.run(['Rscript', str(Path(__file__).with_suffix('.R')),
            str(output), str(i+1), str(path)], capture_output=True, timeout=180)
        if result.returncode or not path.exists():
            write(output / 'state.json', dict(status='paused', reason='adapter_failed_details_suppressed',
                  completed=sum(read(p)['status'] == 'generated' for p in responses.glob('*.json'))))
            print('Paused; completed responses retained. No fallback attempted.', flush=True); return
        response = read(path)
        if response['status'] != 'generated':
            write(output / 'state.json', dict(status='paused', reason=response['status'], last_request=i+1))
            print('Paused on provider or completion failure; no fallback attempted.', flush=True); return
        print(f'Saved {i+1}/8', flush=True)
        if i < len(items)-1: time.sleep(21)
    pairs = []
    for i, row in enumerate(items):
        response = read(responses / f'response_{i+1:02d}.private.json')
        pairs.append(dict(example_id=row['example_id'], paper_id=row['paper_id'], style_group=row['style_group'],
            split='training', plain_english=response['text'], original_paragraph=row['original_paragraph'],
            source_sha256=sha(response['text']), target_sha256=row['target_sha256'],
            generator=MODEL, actual_model_version=response['model_version'], prompt_sha256=row['prompt_sha256'],
            response_sha256=digest(responses / f'response_{i+1:02d}.private.json'),
            pair_status='draft_pending_alignment_review', training_ready=False))
    (output / 'pairs.private.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in pairs))
    write(output / 'state.json', dict(status='generation_complete', generated=8, training_calls=0,
        approved_training_pairs=0, detector_calls=0, paid_fallback=False))
    print('Saved eight draft pairs. No training or detector calls.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    for name in ['prepare', 'check', 'generate']: mode.add_argument('--' + name, action='store_true')
    p.add_argument('--output', type=Path, default=DEFAULT); args = p.parse_args()
    if args.prepare: prepare(args.output)
    elif args.check: check(args.output); print('Validated frozen eight-row free-only trial; no API call.')
    else: generate(args.output)
