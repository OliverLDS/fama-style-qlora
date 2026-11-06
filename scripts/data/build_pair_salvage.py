#!/usr/bin/env python3
"""Assemble reviewed text pairs without conflating optional structural QA.

Decisions and before/after prose stay private. This script does not perform a
semantic review, invoke models, read reserved splits or launch training.
"""
import argparse
from collections import Counter, defaultdict
import difflib
import hashlib
import json
from pathlib import Path
import unicodedata


def sha(value):
    return hashlib.sha256(value).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n')


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in rows))


def build(args):
    audit_dir = args.audit.parent
    parent_manifest = json.loads((audit_dir / 'manifest.json').read_text())
    # Verify all frozen response/audit artifacts, not just the selected rows.
    verified = 0
    for name, expected in parent_manifest['files'].items():
        path = audit_dir / name
        if sha(path.read_bytes()) != expected:
            raise ValueError('Changed parent artifact: ' + name)
        verified += 1
    rows = read_rows(args.audit)
    decisions = [r for p in args.decisions for r in read_rows(p)]
    if sorted(r['index'] for r in decisions) != list(range(len(rows))):
        raise ValueError('Every parent needs exactly one decision')
    decisions.sort(key=lambda r: r['index'])
    selections = {}
    for path in args.selection:
        for row in json.loads(path.read_text()):
            if row['paragraph_id'] in selections:
                raise ValueError('Duplicate selection ID')
            selections[row['paragraph_id']] = row
    inventory = {r['paper_id']: r for r in json.loads(args.inventory.read_text())['papers']}
    output = defaultdict(list)
    audit = []
    patches = []
    group_counts = defaultdict(Counter)
    seen = {}
    paper_roles = {}
    pdf_hashes = {}
    for parent, decision in zip(rows, decisions):
        if decision['decision'] not in {'accept', 'repair', 'exclude'}:
            raise ValueError('Unresolved decision')
        role = parent['role']
        if role not in {'training', 'development'}:
            raise ValueError('Reserved/unknown role refused')
        paper = parent['paper_id']
        if paper in paper_roles and paper_roles[paper] != role:
            raise ValueError('Paper split leakage')
        paper_roles[paper] = role
        key = selections[parent['paragraph_id']]
        if paper not in pdf_hashes:
            pdf_hashes[paper] = sha(Path(inventory[paper]['pdf']).read_bytes())
        if pdf_hashes[paper] != key['pdf_sha256']:
            raise ValueError('Changed source PDF: ' + paper)
        original = parent['annotation']
        if sha(original['original_paragraph'].encode()) != key['original_sha256']:
            raise ValueError('Original target differs from frozen selection')
        views = []
        for name in decision.get('pdf_views', []):
            path = args.evidence_dir / name
            views.append({'path': str(path), 'sha256': sha(path.read_bytes())})
        record = dict(decision)
        record.update(paper_id=paper, paragraph_id=parent['paragraph_id'], role=role,
                      style_group=parent['style_group'], before_source=original['plain_english'],
                      before_target=original['original_paragraph'], source_selection=key,
                      pdf_evidence=views, original_annotation=original,
                      old_hold_reasons=parent['pair_hold_reasons'])
        audit.append(record)
        if decision['decision'] == 'exclude':
            output['excluded'].append(record)
            continue
        children = decision.get('children', [decision])
        for n, child in enumerate(children, 1):
            source, target = child['source'], child['target']
            if not isinstance(source, str) or not source.strip() or not isinstance(target, str) or not target.strip():
                raise ValueError('Missing prose in approved pair')
            if any(unicodedata.category(c) == 'Co' for c in source + target):
                raise ValueError('Unresolved private-use symbol in approved pair')
            fingerprint = sha(' '.join(target.casefold().split()).encode())
            if fingerprint in seen:
                raise ValueError('Duplicate target: ' + parent['paragraph_id'])
            seen[fingerprint] = parent['paragraph_id']
            changed = source != original['plain_english'] or target != original['original_paragraph']
            supplied = original['plain_english'] is None
            example = {
                'id': parent['paragraph_id'] + ('::child_' + str(n) if len(children) > 1 else '') + '::salvage_v1',
                'parent_id': parent['paragraph_id'], 'paper_id': paper,
                'style_group': parent['style_group'], 'authors': inventory[paper]['authors'],
                'split': role, 'plain_english': source, 'original_paragraph': target,
                'source_sha256': sha(source.encode()), 'target_sha256': sha(target.encode()),
                'target_fingerprint': fingerprint,
                'quality': {
                    'text_pair_status': 'automated_pilot_approved',
                    'decision': decision['decision'], 'semantic_review': decision['reviewer'],
                    'full_visual_transcription_verified': False,
                    'pdf_crops_checked': bool(views), 'independent_human_verification': False,
                    'reason': decision['reason'], 'source_caveat': decision.get('source_caveat'),
                    'structural_annotations_status': 'needs_rebinding_after_text_change' if changed else 'retained_separate_not_certified',
                },
                'provenance': {
                    'parent_audit': str(args.audit), 'parent_audit_sha256': sha(args.audit.read_bytes()),
                    'parent_index': decision['index'], 'package': parent['package'], 'batch_id': parent['batch_id'],
                    'parent_original_sha256': key['original_sha256'], 'pdf_sha256': key['pdf_sha256'],
                    'line_refs': key['line_refs'], 'page_location': key['page_location'],
                    'raw_generation': original.get('provenance'),
                    'user_reported_generation': 'GPT-6; requested/observed High thinking effort; raw nulls retained',
                    'source_repair_author': decision.get('source_author', 'Codex' if changed else None),
                    'new_neutral_input_author': 'Codex' if supplied else None,
                    'pdf_evidence': views,
                },
            }
            output[role].append(example)
            group_counts[parent['style_group']][role] += 1
            if changed:
                patches.append({
                    'id': example['id'], 'parent_id': parent['paragraph_id'], 'reason': decision['reason'],
                    'source_diff': list(difflib.unified_diff((original['plain_english'] or '').splitlines(), source.splitlines(), lineterm='')),
                    'target_diff': list(difflib.unified_diff(original['original_paragraph'].splitlines(), target.splitlines(), lineterm='')),
                    'before_source_sha256': sha((original['plain_english'] or '').encode()),
                    'before_target_sha256': key['original_sha256'],
                    'after_source_sha256': example['source_sha256'], 'after_target_sha256': example['target_sha256'],
                    'pdf_evidence': views,
                })
    summary = {
        'protocol': 'pair_salvage_v1', 'parent_pairs_reviewed': len(rows),
        'parent_decisions': dict(Counter(r['decision'] for r in decisions)),
        'usable_examples': len(output['training']) + len(output['development']),
        'training_examples': len(output['training']), 'development_examples': len(output['development']),
        'unique_training_parents': len({r['parent_id'] for r in output['training']}),
        'usable_parent_pairs': sum(r['decision'] != 'exclude' for r in decisions),
        'by_style_group': {k: dict(v) for k, v in sorted(group_counts.items())},
        'neutral_inputs_supplied_by_codex': sum(r['provenance']['new_neutral_input_author'] == 'Codex' for role in ['training','development'] for r in output[role]),
        'changed_examples': len(patches),
        'pdf_evidence_parent_count': sum(bool(r.get('pdf_views')) for r in decisions),
        'parent_artifact_hashes_verified': verified,
        'source_pdf_hashes_verified': len(pdf_hashes),
        'training_papers': len({r['paper_id'] for r in output['training']}),
        'development_papers': len({r['paper_id'] for r in output['development']}),
        'new_model_calls': 0, 'GPU_jobs': 0, 'reserved_rows_read': 0,
        'limitations': [
            'Codex semantic adjudication is automated assessment, not independent human ground truth.',
            'PDF crops resolve targeted transcription issues; full visual verification of every target is not claimed.',
            'Optional ledgers/taxonomies are retained in the audit; changed text requires rebinding before structural training.',
            'Printed author-draft grammatical errors and internal research-specification inconsistencies can remain explicitly flagged.',
            'Earlier datasets remain unchanged and are not implicitly requalified by this release.',
            'No 200-example balanced cohort or training result is claimed.'
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for role in ['training', 'development', 'excluded']:
        write_rows(args.output / (role + '.private.jsonl'), output[role])
    write_rows(args.output / 'repair_audit.private.jsonl', patches)
    write_rows(args.output / 'adjudications.private.jsonl', audit)
    write_json(args.output / 'summary.json', summary)
    inputs = [args.audit, args.inventory] + args.decisions + args.selection
    manifest = {'protocol': 'pair_salvage_v1', 'code_sha256': sha(Path(__file__).read_bytes()),
                'inputs': {str(p): sha(p.read_bytes()) for p in inputs},
                'files': {p.name: sha(p.read_bytes()) for p in sorted(args.output.iterdir()) if p.name != 'manifest.json' and p.is_file()}}
    write_json(args.output / 'manifest.json', manifest)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--decisions', nargs='+', type=Path, required=True)
    parser.add_argument('--selection', nargs='+', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--evidence-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args())
