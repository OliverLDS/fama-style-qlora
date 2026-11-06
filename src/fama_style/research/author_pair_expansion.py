"""Prepare broader paper-level annotation coverage without generating pairs."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil

from .human_prose_annotations import CATALOG, compact_catalog, expand_catalog, output_schema

PROTOCOL = 'author_pair_expansion_v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shingles(text):
    tokens = re.findall(r"[a-z]+", text.lower())
    return {tuple(tokens[i:i+5]) for i in range(max(0, len(tokens)-4))}


def near_duplicate(a, b):
    return bool(a and b) and len(a & b) / len(a | b) >= .80


def select(rows, excluded_ids, comparison_sets, limit=5):
    """Conservative inputs; a shortfall stays a shortfall, never a promoted hold."""
    eligible = [r for r in rows if r['paragraph_id'] not in excluded_ids
                and r['status'] == 'annotation_candidate' and not r.get('hold_reasons')
                and not r.get('duplicate_of') and 70 <= r['word_count'] <= 260
                and not (set(r.get('quality_flags', [])) - {'OCR_not_independently_verified'})]
    chosen, rejected = [], []
    while eligible and len(chosen) < limit:
        seen_pages = {r['pdf_page_start'] for r in chosen}
        eligible.sort(key=lambda r: (r['pdf_page_start'] in seen_pages,
                                     abs(r['word_count']-150), r['paragraph_id']))
        row = eligible.pop(0)
        signature = shingles(row['original_paragraph'])
        if any(near_duplicate(signature, other) for other in comparison_sets):
            rejected.append(row['paragraph_id'])
            continue
        chosen.append(row)
        comparison_sets.append(signature)
    return chosen, rejected


def validate_roles(roles, previous, sealed):
    for pid, role in roles.items():
        if pid in sealed:
            raise ValueError('reserved_paper_in_annotation_plan')
        if role not in ('training', 'development', 'deferred'):
            raise ValueError('unknown_role')
        if pid in previous and previous[pid] == 'training' and role != 'training':
            raise ValueError('training_exposure_reassigned')
        if pid in previous and previous[pid] == 'development' and role != 'development':
            raise ValueError('development_role_changed')


def prepare(root, plan_path, output):
    if output.exists():
        raise ValueError('fresh_output_required')
    root = root.resolve()
    plan_path = plan_path.resolve()
    plan = json.loads(plan_path.read_text())
    sources = {}
    def load_json(relative):
        path = root / relative
        sources[relative] = digest(path)
        return json.loads(path.read_text())
    human = load_json('research_runs/human_prose_annotation_v1/package_v2/SPLIT_MANIFEST.json')
    fama = load_json('data/fama_corpus_v1/split_manifest.json')['paper_assignments']
    commitments = load_json('research_runs/neutral_reconstruction_v1/package_v1/paper_role_commitments.private.json')
    sealed = {r['paper_id'] for r in human['papers'] if r['split'] == 'reserved_test'}
    sealed.update(pid for pid, row in fama.items() if row['split'] == 'test')
    previous = {r['paper_id']: r['split'] for r in human['papers'] if r['split'] != 'reserved_test'}
    previous.update({r['paper_id']: 'training' for r in commitments})
    previous.update({pid: {'train':'training', 'validation':'development'}[r['split']]
                     for pid,r in fama.items() if r['split'] != 'test'})
    roles = {r['paper_id']: r['role'] for r in plan['papers']}
    if len(roles) != len(plan['papers']):
        raise ValueError('duplicate_plan_paper')
    validate_roles(roles, previous, sealed)
    excluded = {r['source_paragraph_id'] for r in load_json('docs/author_curation_v1/post_annotation_holds_v1.json')['holds']}
    # All four files are explicit nonreserved releases, never mixed curated/test exports.
    comparisons = []
    prior_selection = root/'research_runs/neutral_reconstruction_v1/package_v1/private_key/selection.private.jsonl'
    sources[str(prior_selection.relative_to(root))] = digest(prior_selection)
    for line in prior_selection.read_text().splitlines():
        row = json.loads(line)
        if row['paper_id'] in sealed:
            raise ValueError('reserved_prior_selection')
        excluded.add(row['source_paragraph_id'])
        comparisons.append(shingles(row['original_paragraph']))
    for relative in ['data/fama_corpus_v1/train.jsonl', 'data/fama_corpus_v1/validation.jsonl',
                     'data/teacher_pair_pilot_v2/training.private.jsonl',
                     'data/teacher_pair_pilot_v2/development.private.jsonl']:
        path = root / relative
        sources[relative] = digest(path)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            pid = row.get('paper_key', row['paper_id'])
            if pid in sealed:
                raise ValueError('reserved_row_in_nonreserved_release')
            comparisons.append(shingles(row.get('original_paragraph', row.get('target', ''))))
    for relative in plan.get('prior_packages', []):
        if relative not in ('research_runs/author_pair_expansion_v1/existing_collection_package_v1',
                            'research_runs/author_pair_expansion_v1/existing_collection_package_v2',
                            'research_runs/author_pair_expansion_v1/release_v1/existing_collection'):
            raise ValueError('prior_package_not_allowlisted')
        package = root/relative
        index_path = package/'BATCH_INDEX.json'
        sources[str(index_path.relative_to(root))] = digest(index_path)
        for batch in json.loads(index_path.read_text()):
            if batch['paper_id'] in sealed or batch['role'] not in ('training','development'):
                raise ValueError('reserved_prior_package')
            path = (package/batch['input']).resolve()
            if package.resolve() not in path.parents:
                raise ValueError('prior_package_path_escape')
            sources[str(path.relative_to(root))] = digest(path)
            comparisons.extend(shingles(json.loads(line)['text']) for line in path.read_text().splitlines())
    grouped = defaultdict(list)
    for relative in plan['candidate_files']:
        if relative not in ('data/author_style_corpus_v1/paragraphs.private.jsonl',
                            'data/author_style_corpus_v2/paragraphs.private.jsonl',
                            'data/author_style_corpus_v2_1/paragraphs.private.jsonl'):
            raise ValueError('candidate_source_not_allowlisted')
        path = root / relative
        sources[relative] = digest(path)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['paper_id'] in sealed:
                raise ValueError('reserved_candidate')
            if roles.get(row['paper_id']) in ('training', 'development'):
                if hashlib.sha256(row['original_paragraph'].encode()).hexdigest() != row['original_sha256']:
                    raise ValueError('candidate_original_hash_mismatch')
                grouped[row['paper_id']].append(row)
    full = json.loads(CATALOG.read_text())
    compact = compact_catalog(full)
    if expand_catalog(compact) != full['canonical_items']:
        raise ValueError('catalogue_not_lossless')
    output.mkdir(parents=True)
    common = output / 'common'; common.mkdir()
    def write(path, value):
        path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)+'\n')
    write(common/'taxonomy_catalog.compact.json', compact)
    write(common/'annotation_output.schema.json', output_schema(full['taxonomy_version'], full['taxonomy_sha256']))
    shutil.copyfile(root/'docs/human_prose_annotation_v1/ANNOTATION_PROMPT.md', common/'ANNOTATION_PROMPT.md')
    index, keys, coverage = [], [], []
    for paper in sorted(plan['papers'], key=lambda r: (r['role'] != 'training', r['style_group'], r['paper_id'])):
        if paper['role'] == 'deferred':
            continue
        selected, duplicates = select(grouped[paper['paper_id']], excluded, comparisons)
        coverage.append({'paper_id':paper['paper_id'], 'style_group':paper['style_group'],
                         'role':paper['role'], 'selected':len(selected), 'requested':5,
                         'duplicate_rejections':duplicates, 'available_candidates':len(grouped[paper['paper_id']])})
        if not selected:
            continue
        number = len(index)+1
        batch = 'batch_%02d' % number
        directory = output/'batches'/batch; directory.mkdir(parents=True)
        records = []
        for row in selected:
            keys.append({'batch_id':batch, 'paper_id':row['paper_id'], 'style_group':paper['style_group'],
                         'role':paper['role'], 'paragraph_id':row['paragraph_id'],
                         'original_sha256':row['original_sha256'], 'pdf_sha256':row['paper_pdf_sha256'],
                         'page_location':row['page_location'], 'line_refs':row['line_refs'],
                         'normalization_events':row['normalization_events'], 'quality_flags':row['quality_flags']})
            records.append({'paragraph_id':row['paragraph_id'], 'text':row['original_paragraph']})
        (directory/'paragraphs.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records))
        filename = 'responses_%02d.jsonl' % number
        (directory/'TASK.md').write_text('Batch ID: '+batch+'\nRequired response filename: '+filename+
            '\nRequired coverage report: coverage_%02d.md\n' % number+
            '\nUse GPT-6; request High thinking effort. Record the actual visible setting and conversation arrangement.\n\n'+
            (common/'ANNOTATION_PROMPT.md').read_text())
        response_dir = output.parent/'responses'/output.name/paper['role']; response_dir.mkdir(parents=True, exist_ok=True)
        index.append({'batch_id':batch, 'paper_id':paper['paper_id'], 'role':paper['role'],
                      'style_group':paper['style_group'], 'paragraphs':len(records),
                      'input':str(directory.relative_to(output)/'paragraphs.jsonl'),
                      'prompt':str(directory.relative_to(output)/'TASK.md'),
                      'save_response':'../responses/'+output.name+'/'+paper['role']+'/'+filename})
    write(output/'BATCH_INDEX.json', index)
    write(output/'selection.private.json', keys)
    write(output/'coverage.json', coverage)
    write(output/'PAPER_ROLES.json', plan)
    write(output/'session_notes.json', {'requested_model':'GPT-6', 'requested_effort':'High',
         'actual_model':None, 'actual_effort':None, 'conversation_arrangement':None,
         'cross_chat_memory':'not controlled; do not assume independent annotation'})
    summary = {'protocol':PROTOCOL, 'batches':len(index), 'paragraphs':len(keys),
               'selected_by_group':dict(Counter(k['style_group'] for k in keys)),
               'selected_by_role':dict(Counter(k['role'] for k in keys)),
               'papers_with_selected_rows':len(index), 'training_pairs_created':0,
               'status':'awaiting_manual_GPT6_responses', 'model_calls':0, 'GPU_jobs':0,
               'reserved_text_bodies_read':0, 'existing_release_changes':0,
               'shortfall_papers':[r['paper_id'] for r in coverage if r['selected']<5]}
    write(output/'summary.json', summary)
    (output/'START_HERE.md').write_text('# Broader author pairing\n\nFor each row of BATCH_INDEX.json upload its paragraphs.jsonl and TASK.md, plus\ncommon/annotation_output.schema.json and common/taxonomy_catalog.compact.json.\nRequest GPT-6 with High thinking effort. Save each response at save_response,\nrelative to this package. Keep first responses; use a new version for corrections.\nRecord visible model/effort and chat arrangement in session_notes.json.\n\nThese are annotation inputs, not admitted training pairs. No GPU run is needed.\nOnce responses return, audit exact originals, evidence quotes, claim IDs,\nsemantic alignment, taxonomy IDs and paper roles before admitting pairs.\nUncertain or context-dependent rows remain quarantined. All existing releases stay intact.\n')
    sources[str(plan_path.relative_to(root))] = digest(plan_path)
    sources[str(CATALOG.resolve().relative_to(root))] = digest(CATALOG)
    sources['docs/human_prose_annotation_v1/ANNOTATION_PROMPT.md'] = digest(common/'ANNOTATION_PROMPT.md')
    write(output/'manifest.json', {'input_sha256':sources, 'code_sha256':digest(Path(__file__)),
                                  'output_sha256':{str(p.relative_to(output)):digest(p)
                                      for p in sorted(output.rglob('*')) if p.is_file()}})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.plan.resolve(), args.output), indent=2))


if __name__ == '__main__':
    main()
