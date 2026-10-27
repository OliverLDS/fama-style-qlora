"""Six-paper manual comparison of ordinary shared-evidence rewriting and plans."""
import argparse
import copy
import json
from pathlib import Path
import shutil
from . import readable_packing as readable
from . import elementary_edits as edits
from . import strong_realizer_check as transport
from . import bound_grouping as grouping
from .concise_plan import file_sha, read_lines
from .experiment import fresh, write_json, write_lines
from fama_style.research.claims import sha, risk_flags
from fama_style.research.signal import overlap, distance, features

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = 'paper_plan_comparison_v1'
CONDITIONS = ('A_ordinary_shared_readable_evidence', 'B_readable_edits_shared_readable_evidence')
ANCHORS = readable.CASE_IDS
RESPONSE_SCHEMA = transport.RESPONSE_SCHEMA
TASK_PROMPT = '''Process the attached jobs_01.jsonl or jobs_02.jsonl as six rewriting tasks.
For each job, follow its prompt exactly and produce one first-attempt paragraph.
Do not borrow facts or wording from another job, retrieve the original author
text, browse, grade/select answers or add critiques or suggested revisions.

The inner prompt's paragraph-only instruction applies inside the output field.
Return a downloadable UTF-8 JSONL file with one object per job and exactly:
{"job_id":"COPY_FROM_INPUT","prompt_sha256":"COPY_FROM_INPUT","output":"YOUR_PARAGRAPH"}

Copy job_id and prompt_sha256 exactly. Use responses_01.jsonl for jobs_01.jsonl
and responses_02.jsonl for jobs_02.jsonl. No Markdown fences or model-version
guesses. If incomplete, deliver completed records and identify missing job IDs
separately; do not invent placeholders or silently repair/replace the inputs.
'''
START_HERE = '''# Six papers: ordinary rewriting versus readable edit guidance

Prepared, not executed. Twelve requested paragraphs; no training or Kaggle run.
Use GPT-6 with High thinking effort for BOTH batches, matching the model/visible
setting you reported for the preceding check. Hidden decoding is uncontrolled.

1. Open a NEW chat. Attach ONLY batches/jobs_01.jsonl and response.schema.json.
   Paste TASK_PROMPT.md and keep the first returned responses_01.jsonl.
2. Open a DIFFERENT NEW chat with the same model/visible setting. Attach ONLY
   batches/jobs_02.jsonl and response.schema.json. Paste the same TASK_PROMPT.md
   and keep the first returned responses_02.jsonl.
3. Do not put the two batches or their answers into the same chat. Do not pass
   answers from the first chat into the second. Six jobs share each batch chat;
   these are not twelve independent single-job conversations. Separate chats do
   not eliminate memory effects; no chat deletion or memory clearing is required.
4. Save the two returned files unchanged in:
   research_runs/paper_plan_comparison_v1/responses/
   Optionally record the actual model label, visible effort setting and chat
   arrangement in responses/notes.txt. Unknown details are acceptable.
5. Tell Codex the files are ready. No human scoring/review form is needed.

Upload no private_key.private.json, manifests, this file, reports, original
reference paragraphs, old outputs, PDFs or earlier transcripts. Never attach
the entire directory. No credential/API key/HF_TOKEN belongs in the files.
All generation and returned paragraph files must remain private.

Both batches use the same six TRAINING paragraphs in the same deterministic
order, with identical concrete style instruction and readable source evidence.
One adds the existing MERGE/SPLIT/KEEP program; the other has no edit program,
desired sentence count or target grouping. Both source evidence inventories
were derived from paired annotations, so this ordinary condition is a shared
oracle-evidence baseline, not a bare-text deployment result. Reference wording
never enters generator inputs. Case/condition roles are kept in the local key;
formats themselves are visible, so the comparison is not genuinely blinded.

Two cases are previously successful anchors; four extend the external-model
check to other papers. All have been inspected during corpus/annotation work;
there is no held-out generalization claim. No development or reserved test rows
are used. Selection is fixed before these responses and does not rank candidates
by style scores. Hidden system context/decoding, memory, possible source
familiarity and batch interactions remain limitations.

Keep first attempts, including invalid/partial files. Do not retry, rewrite,
select a better answer or silently repair JSON. Syntax/count matches alone do
not establish claim preservation, operation realization or distinctive style.
Codex will separately inspect style/organization, copying and claim drift,
including whether modest drift is justified by worthwhile style gains.
'''
COMMON_TAIL = '\nKeep each composite claim intact. Resolve pronouns without changing attribution.'


def select_six(candidates):
    pool = [r for r in candidates if not r['diagnostic_probe']]
    selected = [next(r for r in pool if r['case_id'] == cid) for cid in ANCHORS]
    if len({r['paper_key'] for r in selected}) != 2: raise ValueError('Anchor paper collision')
    for paper in sorted({r['paper_key'] for r in pool} - {r['paper_key'] for r in selected}):
        selected.append(min((r for r in pool if r['paper_key'] == paper), key=lambda r: (len(r['prompt']), r['case_id'])))
    if len(selected) != 6 or len({r['paper_key'] for r in selected}) != 6: raise ValueError('Expected six paper-distinct nonprobes')
    return sorted(selected, key=lambda r: sha(PROTOCOL + '|' + r['case_id']))


def load_inputs():
    root = ROOT / 'research_runs/operation_expansion_v1'
    package = root / 'package_v1'; key = edits.base.check_package(package)
    compiled = ROOT / 'research_runs/elementary_packing_edits_v1/compiled_v1'
    manifest = json.loads((compiled / 'manifest.json').read_text())
    if manifest['code_sha256'] != file_sha(Path(edits.__file__)): raise ValueError('Changed elementary compiler')
    for name, digest in manifest['files'].items():
        if file_sha(compiled / name) != digest: raise ValueError('Changed compiled artifact')
    audit = root / 'response_audit_v1'; audit_manifest = json.loads((audit / 'manifest.json').read_text())
    if file_sha(audit / 'manifest.json') != manifest['audit_manifest_sha256']: raise ValueError('Changed annotation audit')
    for name, digest in audit_manifest['files'].items():
        if file_sha(audit / name) != digest: raise ValueError('Changed audited annotation')
    candidates = read_lines(compiled / 'oracle_candidates.private.jsonl')
    selected = select_six(candidates); infos = {r['case_id']: r for r in key['records']}
    annotations = {r['case_id']: r['annotation'] for r in read_lines(audit / 'annotations.checked.private.jsonl')}
    rows = []
    for c in selected:
        info = infos[c['case_id']]; record = info['record']
        rebuilt = edits.compile_edits(record, annotations[c['case_id']])
        if rebuilt['generation_payload'] != c['generation_payload'] or c['pair_sha256'] != record['pair_sha256']:
            raise ValueError('Changed claim program or pair')
        if info['split'] != 'train' or info['selection_role'] == 'preservation_or_no_increment_probe' or c['source_id'] != info['source_id'] or c['paper_key'] != info['paper_key']:
            raise ValueError('Wrong split, probe or source identity')
        rows.append({'id': c['case_id'], 'source_id': c['source_id'], 'paper_key': c['paper_key'], 'split': 'train',
                     'use': 'prior_success_anchor' if c['case_id'] in ANCHORS else 'cross_paper_extension',
                     'source': record['source'], 'reference': record['reference'], 'payload': copy.deepcopy(c['generation_payload']),
                     'annotation_sha256': c['annotation_sha256'], 'semantic_equivalence_verified': False})
    prior_path = ROOT / 'research_runs/readable_packing_ablation_v1/input_pack_v3/readable_pack.json'
    prior = readable.load_pack(prior_path, file_sha(prior_path))
    return rows, prior['style_profile'], {'compiled_manifest_sha256': file_sha(compiled / 'manifest.json'),
             'annotation_package_manifest_sha256': file_sha(package / 'manifest.json'), 'style_profile_parent_pack_sha256': file_sha(prior_path),
             'train_sha256': key['train_sha256']}


def shared_prefix(row): return readable.readable_prefix(row['payload']) + COMMON_TAIL


def prompt_for(row, condition):
    if condition not in CONDITIONS: raise ValueError('Unknown condition')
    prompt = shared_prefix(row)
    if condition == CONDITIONS[1]: prompt += '\nSentence-packing edit plan:\n' + readable.readable_plan(row['payload'])
    return prompt + '\nReturn only the rewritten paragraph.'


def generation_records(rows, condition):
    return [{'job_id': 'J' + sha(PROTOCOL + '|' + r['id'] + '|' + condition)[:16],
             'prompt_sha256': sha(prompt_for(r, condition)), 'prompt': prompt_for(r, condition)} for r in rows]


def prepare(output):
    rows, profile, provenance = load_inputs(); fresh(output); (output / 'batches').mkdir(); mapping = []
    for i, condition in enumerate(CONDITIONS, 1):
        batch = f'jobs_{i:02d}.jsonl'; response = f'responses_{i:02d}.jsonl'
        records = generation_records(rows, condition); write_lines(output / 'batches' / batch, records)
        for row, job in zip(rows, records):
            mapping.append({'job_id': job['job_id'], 'source_id': row['id'], 'condition': condition, 'use': row['use'],
                            'prompt_sha256': job['prompt_sha256'], 'batch_file': batch, 'response_file': response})
    write_json(output / 'private_key.private.json', {'protocol': PROTOCOL, **provenance, 'rows': rows, 'jobs': mapping, 'style_profile': profile,
               'selection_rule': 'Force the two prior anchors; shortest compiled prompt per remaining paper, ties by case ID. Opaque hash order shared across arms.',
               'planned_model': 'GPT-6', 'planned_thinking_effort': 'High', 'actual_generation_metadata': 'pending'})
    (output / 'TASK_PROMPT.md').write_text(TASK_PROMPT, encoding='utf-8'); (output / 'START_HERE.md').write_text(START_HERE, encoding='utf-8')
    write_json(output / 'response.schema.json', RESPONSE_SCHEMA)
    write_json(output / 'manifest.json', {'protocol': PROTOCOL, 'status': 'prepared_not_executed', **provenance,
               'code_sha256': file_sha(Path(__file__)), 'requested_outputs': 12, 'papers': 6, 'anchor_cases': 2, 'extension_cases': 4,
               'validation_rows_read': 0, 'reserved_test_rows_read': 0, 'model_calls': 0, 'training_called': False,
               'files': {str(p.relative_to(output)): file_sha(p) for p in output.rglob('*') if p.is_file()}})
    print('Prepared two six-job batches: matched ordinary/shared-evidence versus readable edits. No model request or training.')


def check_package(folder):
    manifest = json.loads((folder / 'manifest.json').read_text())
    if manifest['protocol'] != PROTOCOL or manifest['code_sha256'] != file_sha(Path(__file__)): raise ValueError('Changed package/compiler')
    for name, digest in manifest['files'].items():
        if file_sha(folder / name) != digest: raise ValueError('Prepared file changed')
    key = json.loads((folder / 'private_key.private.json').read_text()); rows, profile, provenance = load_inputs()
    if key['rows'] != rows or key['style_profile'] != profile or any(key[k] != v or manifest[k] != v for k, v in provenance.items()): raise ValueError('Parent or selection changed')
    if len(key['jobs']) != 12 or len({j['job_id'] for j in key['jobs']}) != 12: raise ValueError('Wrong mapping coverage')
    batches = sorted((folder / 'batches').glob('*.jsonl'))
    if [p.name for p in batches] != ['jobs_01.jsonl', 'jobs_02.jsonl']: raise ValueError('Wrong batches')
    expected_mapping = []
    for i, (batch, condition) in enumerate(zip(batches, CONDITIONS), 1):
        expected = generation_records(rows, condition)
        if read_lines(batch) != expected: raise ValueError('Changed generation fields, prompts or order')
        for row, job in zip(rows, expected):
            expected_mapping.append({'job_id': job['job_id'], 'source_id': row['id'], 'condition': condition, 'use': row['use'],
                                     'prompt_sha256': job['prompt_sha256'], 'batch_file': batch.name, 'response_file': f'responses_{i:02d}.jsonl'})
    if key['jobs'] != expected_mapping: raise ValueError('Changed job mapping')
    return key


def audit_responses(package, responses, output):
    key = check_package(package)
    valid, coverage = transport.parse_responses(key, responses)
    fresh(output); (output / 'raw').mkdir()
    for name in coverage['response_file_hashes']: shutil.copyfile(responses / name, output / 'raw' / name)
    if (responses / 'notes.txt').exists(): shutil.copyfile(responses / 'notes.txt', output / 'raw/notes.txt')
    mapping = {j['job_id']: j for j in key['jobs']}; rows = {r['id']: r for r in key['rows']}; metrics = []
    for r in valid:
        j = mapping[r['job_id']]; row = rows[j['source_id']]; text = r['output']; parsed = grouping.output_units(text)
        desired = sum(len(op['output_slots']) for op in row['payload']['edits'])
        metrics.append({**r, 'source_id': row['id'], 'condition': j['condition'], 'use': j['use'], 'source_sha256': sha(row['source']), 'output_sha256': sha(text),
                        'source_similarity': overlap(row['source'], text), 'reference_similarity_evaluation_only': overlap(row['reference'], text),
                        'style_distance_gain_proxy': distance(row['source'], key['style_profile']) - distance(text, key['style_profile']),
                        'features': features(text), 'surface_flags': risk_flags(row['source'], text),
                        'output_sentence_count': len(parsed['units']) if not parsed['ambiguous'] else None,
                        'desired_sentence_count_evaluation_only': desired, 'count_matches': len(parsed['units']) == desired if not parsed['ambiguous'] else None,
                        'relation_surface_checks': transport.relation_check(row['source'], text), 'model_identity': 'not_verified',
                        'termination_reason': 'not_reported', 'semantic_assessment': 'not_executed',
                        'operation_realization': 'not_assessed', 'acceptable_claim_drift': None, 'author_style_accuracy': None})
    summary = {'protocol': PROTOCOL, 'status': 'import_complete_quality_assessment_pending' if coverage['complete'] else 'partial_or_invalid_import_quality_assessment_pending',
               'expected_outputs': 12, 'accepted_outputs': len(valid), 'conditions': {}, 'model_identity': 'not_verified',
               'matched_decoding': False, 'system_context_matches_Kaggle': False, 'model_calls': 0, 'training_called': False,
               'validation_rows_read': 0, 'reserved_test_rows_read': 0, 'semantic_accuracy': None, 'author_style_accuracy': None}
    for c in CONDITIONS:
        group = [r for r in metrics if r['condition'] == c]
        summary['conditions'][c] = {'expected': 6, 'accepted': len(group), 'count_matches': sum(r['count_matches'] is True for r in group),
                                    'near_copies': sum(r['source_similarity'] >= .9 for r in group)}
    summary['roles'] = {use: {'expected_outputs': sum(j['use'] == use for j in key['jobs']), 'accepted_outputs': sum(r['use'] == use for r in metrics)} for use in ('prior_success_anchor', 'cross_paper_extension')}
    write_json(output / 'coverage.json', coverage); write_lines(output / 'metrics.private.jsonl', metrics); write_json(output / 'summary.json', summary)
    write_json(output / 'manifest.json', {'protocol': PROTOCOL, 'code_sha256': file_sha(Path(__file__)), 'package_manifest_sha256': file_sha(package / 'manifest.json'),
               'response_file_hashes': coverage['response_file_hashes'], 'files': {str(p.relative_to(output)): file_sha(p) for p in output.rglob('*') if p.is_file()}})
    print('CPU import saved; semantic, operation and style assessment remain pending. No model call or training.')



def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true'); mode.add_argument('--check', action='store_true'); mode.add_argument('--audit-responses', action='store_true')
    p.add_argument('--package', type=Path); p.add_argument('--responses-dir', type=Path); p.add_argument('--output-dir', type=Path)
    a = p.parse_args(argv)
    if a.prepare:
        if a.output_dir is None: p.error('--output-dir required')
        prepare(a.output_dir)
    elif a.check:
        if a.package is None: p.error('--package required')
        check_package(a.package); print('Six paper-distinct cases, 12 prompts and parent provenance verified on CPU.')
    else:
        if a.package is None or a.responses_dir is None or a.output_dir is None: p.error('--package, --responses-dir and --output-dir required')
        audit_responses(a.package, a.responses_dir, a.output_dir)


if __name__ == '__main__': main()
