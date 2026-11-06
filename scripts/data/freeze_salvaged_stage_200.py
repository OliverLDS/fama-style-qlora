#!/usr/bin/env python3
"""Freeze a balanced 200-parent CPU data cohort from adjudicated private pairs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from fama_style.research.data_scaling import nested_cohorts


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n' for r in rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    rows=[json.loads(line) for path in args.inputs for line in path.read_text().splitlines() if line.strip()]
    canonical=[];parents={};skipped=[];targets={}
    # The two paragraph children share one original annotation parent. Retain
    # one fixed-hash child for this cohort; never inflate N by splitting a row.
    for r in sorted(rows, key=lambda r: digest('stage200-child|'+r['id'])):
        if r['split']!='training' or r['quality']['text_pair_status']!='automated_pilot_approved':
            raise ValueError('Only adjudicated training pairs may enter this pool')
        if r['parent_id'] in parents:
            skipped.append({'id':r['id'],'reason':'One example per annotation parent for cohort-size accounting'})
            continue
        parents[r['parent_id']]=r
        target=' '.join(unicodedata.normalize('NFKC',r['original_paragraph']).casefold().split())
        fingerprint=digest(target)
        if fingerprint in targets: raise ValueError('Duplicate original target')
        targets[fingerprint]=r['id']
        if digest(r['plain_english'])!=r['source_sha256'] or digest(r['original_paragraph'])!=r['target_sha256']:
            raise ValueError('Changed approved text')
        canonical.append(dict(r,paragraph_id=r['id'],parent_target_id=r['parent_id'],source=r['plain_english'],target=r['original_paragraph'],training_ready=True))
    cohort=nested_cohorts(canonical,sizes=(200,))[200]
    # Nonselected approved pairs are retained, not thrown away.
    selected={r['id'] for r in cohort}
    remaining=[r for r in canonical if r['id'] not in selected]
    group_counts=dict(Counter(r['style_group'] for r in cohort))
    assert sorted(group_counts.values())==[50]*4
    args.output.mkdir(parents=True,exist_ok=True)
    write_rows(args.output/'training.private.jsonl',cohort)
    write_rows(args.output/'remaining_approved.private.jsonl',remaining)
    write_rows(args.output/'parent_accounting.private.jsonl',skipped)
    summary={
        'protocol':'data_scaling_v1_stage200_salvage_v1', 'status':'data_frozen_no_training_submitted',
        'training_examples':len(cohort),'unique_annotation_parents':len({r['parent_id'] for r in cohort}),
        'group_counts':group_counts,'papers':len({r['paper_id'] for r in cohort}),
        'approved_unique_training_pool':len(canonical),'remaining_approved_training_pairs':len(remaining),
        'duplicate_parent_children_not_counted':len(skipped),
        'selection':'data_scaling_v1 fixed-hash paper interleaving; one fixed-hash child per annotation parent',
        'new_GPU_jobs':0,'reserved_rows_read':0,
        'qualification':'Automated semantic reassessment plus retained source provenance; not independent human gold.',
        'evaluation':'Development release stays separate; fixed evaluation panel and training settings are not frozen by this command.'
    }
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    manifest={'protocol':summary['protocol'],'code_sha256':digest(Path(__file__).read_text()),
              'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in args.inputs},
              'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir() if p.name!='manifest.json' and p.is_file()}}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
