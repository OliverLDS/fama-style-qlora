"""Audit historical reuse and prepare only missing GPT-6 inputs; CPU only."""
import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import itertools
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / 'research_runs/data_scaling_v1/gpt6_track_0500_v1'
MASTER = 'data/data_scaling_v1/stage_0500_expansion_v1/training.private.jsonl'
EXPECTED_MASTER = '1ab5b21983dff0371d0446fdb0f96a131db24b7605f8804a135568e57263a59a'
POOLS = {
    'early_fama': 'data/fama_corpus_v1/train.jsonl',
    'early_20_batches': 'research_runs/multi_author_milestone_v1/package_v1/datasets/train.provisional.private.jsonl',
    'reviewed_teacher': 'data/teacher_pair_pilot_v2/training.private.jsonl',
    'later_54_batches': 'research_runs/author_pair_expansion_v1/response_audit_v1/audits.private.jsonl',
    'reviewed_salvage': 'data/author_pairs_salvage_v1/training.private.jsonl',
}
PROMPT = '''Create neutral, meaning-preserving inputs for an academic prose rewriting dataset.
This batch is approved for generation: source curation and target selection are complete.
Do not ask for prerequisite confirmation. Process every supplied record.

For each original_paragraph, produce plain_english for an economically literate reader
who is not a specialist in the paper's methods. Simplify difficult vocabulary and
sentence structure; use shorter sentences where helpful. Produce a full paragraph,
not a summary. Avoid mere connective/synonym substitutions or making the text more
formal. Do not deliberately imitate stereotypical AI prose or optimize detector scores.

Retain substantive claims and their scope, actors and roles, conditions, exceptions,
uncertainty, frequency, negation, quantities, names, citations, technical distinctions,
comparative rankings, attribution, and causal direction. Keep the distinction between
hypotheses, expectations, and findings. Do not add explanations, mechanisms, examples,
or conclusions. Keep technical terms when substitution would change meaning.
Do not force a particular edit distance. If already clear, retain the wording and
mark already_clear. If meaning is uncertain, provide the safest possible version,
mark pending_review, and identify the specific uncertainty rather than guessing.

These are curated text inputs: do not perform OCR, change the original target, infer
missing context, extract taxonomies, or write a claim ledger. Do not echo the original
paragraph as an extra field. Check the rewrite for added/lost claims and changed
quantifiers, conditions, numbers, or causal relationships before returning it.

Return a downloadable UTF-8 JSONL file with the filename specified below, one object
per input, in input order, with exactly these five fields:
- id: copy the supplied ID exactly.
- target_sha256: copy the supplied string exactly; do not calculate a new hash.
- plain_english: nonempty complete neutral paragraph.
- rewrite_status: simplified, already_clear, or pending_review.
- review_notes: an array of strings; empty if no uncertainty remains.

Do not silently omit records. No Markdown fences inside the downloadable file.
If file creation is unavailable, return all complete JSONL records as text.
Model/effort provenance is recorded separately; do not invent or certify settings.
'''
SCHEMA = {
    '$schema': 'https://json-schema.org/draft/2020-12/schema',
    'type': 'object', 'additionalProperties': False,
    'required': ['id', 'target_sha256', 'plain_english', 'rewrite_status', 'review_notes'],
    'properties': {
        'id': {'type': 'string', 'pattern': '^P[0-9]{3}$'},
        'target_sha256': {'type': 'string', 'pattern': '^[a-f0-9]{64}$'},
        'plain_english': {'type': 'string', 'minLength': 1},
        'rewrite_status': {'enum': ['simplified', 'already_clear', 'pending_review']},
        'review_notes': {'type': 'array', 'items': {'type': 'string'}},
    },
}


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + '\n')


def jsonl(rows):
    return ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in rows)


def norm(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def human(row):
    return row.get('target', row.get('original_paragraph', row.get('annotation', {}).get('original_paragraph', '')))


def identity(row):
    return row.get('id', row.get('example_id', row.get('paragraph_id')))


def classify(row, early_by_hash):
    p = row['provenance']
    if p.get('new_neutral_input_author') == 'Codex':
        return 'missing_GPT6_wholly_Codex_input'
    if 'GPT-6' in p.get('user_reported_generation', '') or p.get('generator', '').startswith('GPT-6'):
        return 'reuse_recorded_GPT6'
    if p.get('parent_file') == POOLS['early_fama']:
        return 'reuse_user_confirmed_historical_GPT6'
    if row['target_sha256'] in early_by_hash:
        old = early_by_hash[row['target_sha256']]
        if old['source'] != row['source'] or old['annotation_origin'] != 'paired_GPT6_not_independent':
            raise ValueError('historical_teacher_source_binding')
        return 'reuse_prior_known_GPT6'
    if p.get('generator') != 'gemini-3-flash-preview':
        raise ValueError('unknown_missing_source_origin')
    return 'missing_GPT6_current_Gemini_input'


def fuzzy_flags(left, right):
    """Candidate screen, not a semantic guarantee or an exhaustive paraphrase test."""
    out = []
    for a in left:
        at = re.findall(r'\w+', norm(human(a))); aset = set(at)
        for b in right:
            bt = re.findall(r'\w+', norm(human(b))); bset = set(bt)
            if not at or not bt or min(len(at), len(bt)) / max(len(at), len(bt)) < .55:
                continue
            if len(aset & bset) / min(len(aset), len(bset)) < .65:
                continue
            ratio = SequenceMatcher(None, at, bt, autojunk=False).ratio()
            if ratio >= .76:
                out.append({'left_id': identity(a), 'right_id': identity(b), 'token_sequence_ratio': ratio})
    return out


def overlap_audit(pools):
    exact = []; lineage = []
    for (an, aa), (bn, bb) in itertools.combinations(pools.items(), 2):
        a = defaultdict(list); b = defaultdict(list)
        for r in aa:
            a[norm(human(r))].append(identity(r))
        for r in bb:
            b[norm(human(r))].append(identity(r))
        common = a.keys() & b.keys()
        exact.append({'left': an, 'right': bn, 'shared_normalized_targets': len(common)})
        for text in sorted(common):
            lineage.append({'relationship': 'same_normalized_human_target', 'left_pool': an,
                            'right_pool': bn, 'left_ids': a[text], 'right_ids': b[text],
                            'normalized_target_sha256': sha(text)})
    raw = {r['paragraph_id']: r for r in pools['later_54_batches']}
    for r in pools['reviewed_salvage']:
        parent = raw.get(r['parent_id'])
        if parent is None:
            raise ValueError('missing_54_batch_parent')
        lineage.append({'relationship': 'reviewed_or_repaired_version', 'left_pool': 'later_54_batches',
                        'right_pool': 'reviewed_salvage', 'parent_id': r['parent_id'],
                        'child_id': r['id'], 'target_unchanged_normalized': norm(human(parent)) == norm(human(r)),
                        'source_unchanged': parent['annotation'].get('plain_english') == r['plain_english']})
    return exact, lineage, fuzzy_flags(pools['early_fama'], pools['later_54_batches'])


def prepare(out=DEFAULT_OUT):
    if out.exists():
        raise ValueError('refuse_overwrite_GPT6_preparation')
    evidence = {}

    def bound(relative, manifest, key=None):
        mp = ROOT / manifest; path = ROOT / relative
        expected = read(mp)['files'][key or path.name]
        if digest(path) != expected:
            raise ValueError('changed_bound_parent: ' + relative)
        evidence[manifest] = digest(mp); evidence[relative] = expected
        return lines(path)

    master = bound(MASTER, str(Path(MASTER).parent / 'manifest.json'))
    if digest(ROOT / MASTER) != EXPECTED_MASTER or len(master) != 500:
        raise ValueError('frozen_500_required')
    pools = {
        'early_fama': bound(POOLS['early_fama'], 'data/fama_corpus_v1/build_report.json'),
        'early_20_batches': bound(POOLS['early_20_batches'], 'research_runs/multi_author_milestone_v1/package_v1/manifest.json', 'datasets/train.provisional.private.jsonl'),
        'reviewed_teacher': bound(POOLS['reviewed_teacher'], 'data/teacher_pair_pilot_v2/manifest.json'),
        'later_54_batches': bound(POOLS['later_54_batches'], 'research_runs/author_pair_expansion_v1/response_audit_v1/manifest.json'),
        'reviewed_salvage': bound(POOLS['reviewed_salvage'], 'data/author_pairs_salvage_v1/manifest.json'),
    }
    pools['later_54_batches'] = [r for r in pools['later_54_batches'] if r['role'] == 'training']
    if [len(pools[k]) for k in POOLS] != [247, 58, 46, 185, 171]:
        raise ValueError('historical_inventory_changed')
    early = {r['target_sha256']: r for r in pools['early_20_batches']}
    legacy = {r['id']: r for r in pools['early_fama']}
    raw = {r['paragraph_id']: r for r in pools['later_54_batches']}
    archive = ROOT / 'archive/fama_corpus_20261019'
    evidence[str((archive / 'manifest.json').relative_to(ROOT))] = digest(archive / 'manifest.json')
    pdfs = {r['archive_path']: r for r in read(archive / 'manifest.json')['files'] if r['kind'] == 'pdf'}
    devpath = 'data/author_pairs_salvage_v1/development.private.jsonl'
    # Only PDF identity is used from development metadata; no development text is analyzed.
    dev = bound(devpath, 'data/author_pairs_salvage_v1/manifest.json')
    devpdfs = {r['provenance']['pdf_sha256'] for r in dev}
    index = []; reused = []; missing = []; aliases = defaultdict(set)
    group_pdfs = defaultdict(set); missing_by_group = Counter(); reuse_by_group = Counter()
    seen = set()
    for i, row in enumerate(master, 1):
        if row['split'] != 'training' or not row['training_ready'] or sha(row['target']) != row['target_sha256'] or sha(row['source']) != row['source_sha256']:
            raise ValueError('invalid_approved_parent')
        nt = norm(row['target'])
        if nt in seen:
            raise ValueError('duplicate_normalized_master_target')
        seen.add(nt)
        p = row['provenance']; category = classify(row, early)
        pdf = p.get('pdf_sha256')
        if not pdf and p.get('parent_file') == POOLS['early_fama']:
            old = legacy[p['parent_id']]
            entry = pdfs[old['provenance']['pdf_archive']]
            relative = str((archive / entry['archive_path']).relative_to(ROOT))
            if relative not in evidence:
                if digest(ROOT / relative) != entry['sha256']:
                    raise ValueError('archived_pdf_changed')
                evidence[relative] = entry['sha256']
            pdf = entry['sha256']
        if not pdf and row['target_sha256'] in early:
            pdf = early[row['target_sha256']]['pdf_sha256']
        if not pdf or pdf in devpdfs:
            raise ValueError('missing_pdf_identity_or_development_pdf_overlap')
        aliases[pdf].add(row['paper_id']); group_pdfs[row['style_group']].add(pdf)
        entry = dict(id=f'P{i:03d}', mixed_index=i, mixed_id=row['id'], target_sha256=row['target_sha256'],
                     parent_target_id=row['parent_target_id'], raw_paper_id=row['paper_id'],
                     canonical_pdf_sha256=pdf, style_group=row['style_group'], category=category,
                     current_source_sha256=row['source_sha256'])
        if category.startswith('reuse_'):
            reuse_by_group[row['style_group']] += 1
            reused.append(dict(entry, source=row['source'], target=row['target'],
                               original_pair_provenance=p, quality=row['quality'],
                               teacher='GPT-6', historical_effort='not independently established',
                               model_confirmation_basis='user confirmation of original PDF-to-dataset workflow' if category == 'reuse_user_confirmed_historical_GPT6' else 'linked earlier generation provenance',
                               independent_human_verification=False))
        else:
            missing_by_group[row['style_group']] += 1
            if category == 'missing_GPT6_wholly_Codex_input':
                if raw[row['parent_id']]['annotation'].get('plain_english') is not None:
                    raise ValueError('unexpected_existing_GPT6_source')
            missing.append(dict(entry, original_paragraph=row['target']))
        index.append(entry)
    counts = Counter(r['category'] for r in index)
    expected = dict(reuse_recorded_GPT6=158, reuse_user_confirmed_historical_GPT6=111,
                    reuse_prior_known_GPT6=4, missing_GPT6_wholly_Codex_input=13,
                    missing_GPT6_current_Gemini_input=214)
    if counts != Counter(expected) or len(aliases) != 48:
        raise ValueError('reuse_or_paper_inventory_changed')
    exact, lineage, fuzzy = overlap_audit(pools)
    missing.sort(key=lambda r: (r['mixed_index'] > 200, r['mixed_index']))
    # One attachment per job; the first completes the 13 missing first-200 inputs.
    out.mkdir(parents=True); batches = out / 'manual_batches_v1'; batches.mkdir()
    (out / 'responses').mkdir()
    write(out / 'user_model_confirmation.json', dict(
        model='GPT-6', evidence='User confirmed always GPT-6, including batches before the 20/54-batch work.',
        original_workflow='Raw article PDFs uploaded to ChatGPT for OCR/extraction/rewriting, as recalled by user.',
        historical_thinking_effort=None, independent_verification=False,
        scope='Historical model provenance only; does not certify extraction or semantic correctness.',
        original_artifacts_modified=False))
    write(out / 'output.schema.json', SCHEMA)
    (out / 'ANNOTATION_PROMPT.md').write_text(PROMPT)
    (out / 'target_lineage.private.jsonl').write_text(jsonl(index))
    (out / 'reused_pairs.private.jsonl').write_text(jsonl(reused))
    (out / 'overlap_lineage.private.jsonl').write_text(jsonl(lineage))
    write(out / 'paper_identity_audit.json', dict(stored_paper_ids=53, canonical_pdf_count=48,
        canonical_papers_by_group={k: len(v) for k, v in group_pdfs.items()},
        alias_groups=[dict(pdf_sha256=k, paper_ids=sorted(v)) for k, v in sorted(aliases.items()) if len(v) > 1],
        training_development_pdf_overlap=0, frozen_parent_files_modified=False))
    jobs = []
    for start in range(0, len(missing), 30):
        chunk = missing[start:start + 30]; num = len(jobs) + 1
        inputs = [dict(id=r['id'], target_sha256=r['target_sha256'], original_paragraph=r['original_paragraph']) for r in chunk]
        filename = f'responses_{num:02d}.jsonl'
        content = PROMPT + f'\nOutput filename: {filename}\nExpected records: {len(inputs)}\n\nINPUTS (JSONL):\n```jsonl\n' + jsonl(inputs) + '```\n'
        (batches / f'batch_{num:02d}.md').write_text(content)
        (batches / f'inputs_{num:02d}.jsonl').write_text(jsonl(inputs))
        jobs.append(dict(batch=num, ids=[r['id'] for r in chunk], attachment=f'manual_batches_v1/batch_{num:02d}.md',
                         response='responses/' + filename, records=len(chunk),
                         requested_model='GPT-6', requested_thinking_effort='High',
                         actual_model=None, actual_effort=None, cross_chat_memory_uncontrolled=True))
    write(out / 'jobs.json', jobs)
    summary = dict(protocol='gpt6_track_0500_v1', status='prepared_not_released',
        matched_human_targets=500, reused_GPT6_pairs=len(reused), GPT6_inputs_to_generate=len(missing),
        reused_by_group=dict(reuse_by_group), missing_by_group=dict(missing_by_group),
        category_counts=dict(counts), batches=len(jobs), batch_sizes=[r['records'] for r in jobs],
        first_200_reused=187, first_200_missing=13, paper_ids=53, canonical_pdfs=48,
        exact_overlap_by_pool=exact, fuzzy_early_fama_later_54_flags=fuzzy,
        fuzzy_screen=dict(min_token_length_ratio=.55, min_shared_smaller_vocabulary=.65, min_token_sequence_ratio=.76),
        later_54_reviewed_parent_count=len({r['parent_id'] for r in pools['reviewed_salvage']}),
        later_54_reviewed_rows=len(pools['reviewed_salvage']), development_pdf_overlap=0,
        limits=['Training pools only; reserved test text not accessed.',
                'Fuzzy candidate screen is not proof of absence of all semantic duplication.',
                'Historical/current prompts and workflows differ; matched targets do not imply prompt-controlled teacher comparison.',
                'Automated prior acceptance is not independent semantic verification.',
                'No new pairs released until responses have been checked; all 227 selected targets are intended for completion.'],
        provider_calls=0, training_calls=0)
    write(out / 'summary.json', summary)
    (out / 'START_HERE.md').write_text('''# GPT-6 matched 500-pair completion

273 accepted GPT-6 pairs are reused. Generate only the 227 missing inputs in eight jobs
(30 records in each of jobs 1–7; 17 in job 8). All selected targets belong to the same
frozen mixed-500 training cohort. These are not eight competing samples.

1. Select GPT-6 and request High thinking effort.
2. Upload only `manual_batches_v1/batch_01.md` and say:
   “Complete the attached batch and return the named downloadable JSONL file.”
3. Save `responses_01.jsonl` in this folder's `responses/` directory.
4. Repeat for batch_02.md through batch_08.md, keeping the corresponding filenames.

Each attachment contains its instructions and input paragraphs. Do not upload the
machine-readable inputs separately, the source PDFs, other teachers' prose, schemas,
or historical datasets. No taxonomy annotation or manual CSV work is needed.
If download is unavailable, save the complete JSONL text under the same filename.
Separate chats are optional; no independence or isolation from chat memory is assumed.
Report deviations from requested model/effort when returning the responses.

Batch 1 includes all 13 gaps in the first-200 cohort; its remaining 17 items and the
other batches complete the first 500. No additional selection discards are planned:
all 227 targets are intended for completion, with uncertain outputs repaired or
regenerated as needed after automated checks. Do not rerun the 273 reused pairs.

The generation direction here is HUMAN paragraph → neutral GPT-6 prose. Subsequent
fine-tuning reverses it: neutral GPT-6 prose → HUMAN target. No training is started.

## Overlap and lineage

The earlier 58 training candidates include all 46 reviewed teacher-pilot targets.
The later 171 salvage rows descend from 170 of the 185 raw training parents from the
54-batch collection (one parent was split). Reviewed/repaired versions are not new
independent targets. There are 117 unchanged normalized target matches between those
raw and salvage pools. No exact target overlap was found between the early Fama
247 and later 185 training records; the documented fuzzy screen flagged none.
Per-target evidence is in target_lineage.private.jsonl and overlap_lineage.private.jsonl.

Paper counting correction: 53 stored IDs represent 48 distinct PDFs. Five Fama papers
have both short and long IDs. The frozen 500 human paragraphs are distinct and remain
unchanged. The canonical PDF audit found no overlap with development PDFs.

The user's GPT-6 confirmation is preserved in a separate provenance record; historical
files and unknown effort fields are unchanged. Small recorded repairs to GPT-6 text
retain their repair authorship. Thirteen wholly Codex-written inputs are not relabeled
GPT-6: they are included in the missing-input jobs. Historical prompts/workflows differ,
so these matched cohorts are not a fully prompt-controlled teacher experiment.

The manifest binds all prepared inputs and audit files. Response files are intentionally
outside that manifest until separately imported. No final GPT-6-only training release
exists yet. No API calls, GPU jobs, or detector scans are requested by this packet.
''')
    evidence[str(Path(__file__).relative_to(ROOT))] = digest(Path(__file__))
    write(out / 'manifest.json', dict(protocol='gpt6_track_0500_v1', source_files=evidence,
        files={str(p.relative_to(out)): digest(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    validate(out)
    print('Prepared: 273 GPT-6 pairs reused; 227 missing inputs; 8 single-attachment batches. No provider or training calls.')


def validate(out=DEFAULT_OUT):
    manifest = read(out / 'manifest.json')
    for name, expected in manifest['source_files'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('source_changed: ' + name)
    for name, expected in manifest['files'].items():
        if digest(out / name) != expected:
            raise ValueError('packet_changed: ' + name)
    index = lines(out / 'target_lineage.private.jsonl'); reused = lines(out / 'reused_pairs.private.jsonl')
    jobs = read(out / 'jobs.json'); wanted = {r['id'] for r in index if r['category'].startswith('missing_')}
    actual = [rid for job in jobs for rid in job['ids']]
    if len(actual) != len(set(actual)) or set(actual) != wanted or {r['id'] for r in reused} & wanted:
        raise ValueError('coverage_or_duplicate')
    if len(index) != 500 or len(reused) != 273 or len(wanted) != 227:
        raise ValueError('counts')
    if not {r['id'] for r in index[:200] if r['id'] in wanted} <= set(jobs[0]['ids']):
        raise ValueError('first_200_priority')
    for job in jobs:
        inputs = lines(out / 'manual_batches_v1' / f"inputs_{job['batch']:02d}.jsonl")
        if [r['id'] for r in inputs] != job['ids'] or jsonl(inputs) not in (out / job['attachment']).read_text():
            raise ValueError('attachment_inputs')
    return {'reused': len(reused), 'missing': len(wanted), 'batches': len(jobs)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'check'])
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    else:
        print(json.dumps(validate(), sort_keys=True))
