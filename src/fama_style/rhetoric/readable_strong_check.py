"""Four frozen single-job requests for a manual external-realizer check; CPU only."""
import argparse
import json
from pathlib import Path
import shutil
from . import readable_packing as parent
from . import strong_realizer_check as transport
from . import bound_grouping as grouping
from .concise_plan import file_sha, read_lines
from .experiment import fresh, write_json, write_lines
from fama_style.research.claims import sha, risk_flags
from fama_style.research.signal import overlap, distance, features

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = 'readable_strong_realizer_v1'
RESPONSE_SCHEMA = transport.RESPONSE_SCHEMA
TASK_PROMPT = '''Process the attached jobs_XX.jsonl as one independent rewriting task.

Follow the prompt field exactly. Produce one first-attempt candidate. Do not
retrieve the original author text, browse, import facts from other tasks, grade
your answer, select among alternatives, or add critiques or suggested revisions.

Delivery contract: the inner prompt's request to return only the paragraph
applies inside the output field. Return one JSON object with exactly:
{"job_id":"COPY_FROM_INPUT","prompt_sha256":"COPY_FROM_INPUT","output":"YOUR_PARAGRAPH"}

Copy job_id and prompt_sha256 exactly. Return a downloadable UTF-8 JSONL file
named responses_XX.jsonl, replacing XX with the attached input file's number.
Do not include Markdown fences or invent a model identity, version, or completion
claim. If unable to complete the task, state that separately; do not fabricate
a placeholder paragraph or silently repair/replace the task.
'''
START_HERE = '''# External realizer: four fixed prompts — prepared, not executed

Use the same chosen instruction-following model and visible settings for all
four tasks. A product/model label does not establish that it is stronger; this
experiment checks its actual responses. No API key or HF_TOKEN is needed.

For each XX from 01 through 04:
1. Open a separate new chat with the same chosen model/settings.
2. Attach ONLY batches/jobs_XX.jsonl and response.schema.json.
3. Copy TASK_PROMPT.md into that chat and send it.
4. Keep the first returned responses_XX.jsonl unchanged. Save it in:
   research_runs/readable_strong_realizer_v1/responses/

After all four files are saved, tell Codex they are ready. No quality scoring
or human-review form is required. Codex will check coverage and then inspect
organization, style, copying and claim drift separately.

Optionally fill responses/notes.txt with the visible model label, known visible
settings and whether separate chats were used. Unknown is acceptable. Do not
ask the model to invent its hidden version, temperature, seed or token budget.
Do not put credentials in any file.

Do NOT upload this file, private_key.private.json, manifest.json, reports,
reference paragraphs, old outputs, the whole package directory or earlier chat
transcripts. Only the numbered input file, response schema and shared task
prompt are generation materials. Keep all source/response files private.

The four inner prompts are exactly those from the completed readable-packing
Kaggle comparison: two offset/JSON controls and two readable source-binding/plan
conditions on the hierarchy merge and market-efficiency split. There are no
original reference paragraphs in the generation files. Plans/claim inventories
still contain oracle information from paired annotations. Both cases are
previously inspected TRAINING examples, selected after observing prior outputs.

Single-job chats reduce direct shared answer context, but do not prove
independence or eliminate chat memory. Chat deletion or memory clearing is not
required. The delivery wrapper, system prompts, hidden settings and possible
source familiarity differ from Kaggle. Do not claim a matched model-size study,
verified semantic fidelity, author-style accuracy or held-out generalization.

Do not retry, edit an answer, select the best candidate or silently fix invalid
JSON. Partial/invalid files will be retained and reported. Syntax completeness
does not establish normal generation termination. No external requests, Kaggle
runs, annotation collection, model downloads or training occur in preparation.
'''


def parent_inputs():
    base = ROOT / 'research_runs/readable_packing_ablation_v1'
    path = base / 'input_pack_v3/readable_pack.json'
    pack = parent.load_pack(path, file_sha(path))
    run = base / 'kaggle_version_1/readable_packing_ablation_v1'
    review = base / 'version_1_review/verification.json'
    verified = json.loads(review.read_text())
    checks = ['all_artifact_hashes_verified', 'pack_and_frozen_runtime_match', 'jobs_match',
              'source_output_evidence_bound', 'review_examples_exactly_anchored', 'historical_prompts_and_outputs_identical', 'completion_log_verified']
    if not all(verified.get(k) is True for k in checks): raise ValueError('Parent review incomplete')
    if file_sha(run / 'manifest.json') != verified['raw_manifest_sha256']: raise ValueError('Reviewed manifest changed')
    m = json.loads((run / 'manifest.json').read_text())
    if m['pack_sha256'] != file_sha(path) or m['runtime_code_files'] != pack['runtime_code_files']: raise ValueError('Parent identity changed')
    for name, digest in m['files'].items():
        if file_sha(run / name) != digest: raise ValueError('Parent artifact changed')
    jobs = read_lines(run / 'jobs.private.jsonl')
    if jobs != parent.jobs_for(pack): raise ValueError('Parent prompts changed')
    return pack, jobs, {'parent_pack_sha256': file_sha(path), 'parent_manifest_sha256': file_sha(run / 'manifest.json'), 'parent_review_sha256': file_sha(review)}


def requests_for(jobs):
    result = []
    for i, j in enumerate(jobs, 1):
        result.append({'job_id': 'J' + sha(PROTOCOL + '|' + j['id'] + '|' + j['condition'])[:16],
                       'prompt_sha256': j['prompt_sha256'], 'prompt': j['prompt']})
    return result


def prepare(output):
    pack, jobs, provenance = parent_inputs(); requests = requests_for(jobs)
    if len(requests) != 4: raise ValueError('Expected four requests')
    fresh(output); (output / 'batches').mkdir(); mapping = []
    for i, (j, r) in enumerate(zip(jobs, requests), 1):
        batch = f'jobs_{i:02d}.jsonl'; response = f'responses_{i:02d}.jsonl'
        write_lines(output / 'batches' / batch, [r])
        mapping.append({'job_id': r['job_id'], 'source_id': j['id'], 'condition': j['condition'],
                        'prompt_sha256': j['prompt_sha256'], 'batch_file': batch, 'response_file': response})
    write_json(output / 'private_key.private.json', {'protocol': PROTOCOL, **provenance, 'jobs': mapping,
               'rows': pack['rows'], 'style_profile': pack['style_profile'], 'model_identity': 'not_supplied_not_verified'})
    (output / 'TASK_PROMPT.md').write_text(TASK_PROMPT, encoding='utf-8')
    (output / 'START_HERE.md').write_text(START_HERE, encoding='utf-8')
    write_json(output / 'response.schema.json', RESPONSE_SCHEMA)
    write_json(output / 'manifest.json', {'protocol': PROTOCOL, 'status': 'prepared_not_executed', **provenance,
               'code_sha256': file_sha(Path(__file__)), 'requested_outputs': 4, 'single_job_files': 4, 'model_calls': 0,
               'training_called': False, 'validation_rows_read': 0, 'reserved_test_rows_read': 0,
               'files': {str(p.relative_to(output)): file_sha(p) for p in output.rglob('*') if p.is_file()}})
    print('Prepared four exact frozen prompts in four single-job files. No external request or GPU run made.')


def check_package(folder):
    m = json.loads((folder / 'manifest.json').read_text())
    if m['protocol'] != PROTOCOL or m['code_sha256'] != file_sha(Path(__file__)): raise ValueError('Package/compiler changed')
    for name, digest in m['files'].items():
        if file_sha(folder / name) != digest: raise ValueError('Prepared file changed')
    key = json.loads((folder / 'private_key.private.json').read_text())
    pack, jobs, provenance = parent_inputs()
    if any(key[k] != v or m[k] != v for k, v in provenance.items()): raise ValueError('Parent provenance changed')
    if key['rows'] != pack['rows'] or key['style_profile'] != pack['style_profile']: raise ValueError('Private evaluation key changed')
    expected = requests_for(jobs); actual = []; mappings = key['jobs']
    if len(mappings) != 4 or len({j['job_id'] for j in mappings}) != 4: raise ValueError('Wrong mapping coverage')
    batches = sorted((folder / 'batches').glob('*.jsonl'))
    if len(batches) != 4: raise ValueError('Wrong batch coverage')
    for i, (batch, mapping, job) in enumerate(zip(batches, mappings, jobs), 1):
        items = read_lines(batch)
        if len(items) != 1 or set(items[0]) != {'job_id', 'prompt_sha256', 'prompt'}: raise ValueError('Wrong generation fields/count')
        r = items[0]
        if batch.name != f'jobs_{i:02d}.jsonl' or mapping != {'job_id': r['job_id'], 'source_id': job['id'], 'condition': job['condition'],
             'prompt_sha256': job['prompt_sha256'], 'batch_file': batch.name, 'response_file': f'responses_{i:02d}.jsonl'}:
            raise ValueError('Changed job mapping')
        if sha(r['prompt']) != r['prompt_sha256']: raise ValueError('Changed prompt')
        actual.append(r)
    if actual != expected: raise ValueError('Generation prompts differ from completed experiment')
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
        metrics.append({**r, 'source_id': row['id'], 'condition': j['condition'], 'source_sha256': sha(row['source']), 'output_sha256': sha(text),
                        'source_similarity': overlap(row['source'], text), 'reference_similarity_evaluation_only': overlap(row['reference'], text),
                        'style_distance_gain_proxy': distance(row['source'], key['style_profile']) - distance(text, key['style_profile']),
                        'features': features(text), 'surface_flags': risk_flags(row['source'], text),
                        'output_sentence_count': len(parsed['units']) if not parsed['ambiguous'] else None,
                        'desired_sentence_count': desired, 'count_matches': len(parsed['units']) == desired if not parsed['ambiguous'] else None,
                        'relation_surface_checks': transport.relation_check(row['source'], text), 'model_identity': 'not_verified',
                        'termination_reason': 'not_reported', 'semantic_assessment': 'not_executed',
                        'operation_realization': 'not_assessed', 'acceptable_claim_drift': None, 'author_style_accuracy': None})
    summary = {'protocol': PROTOCOL, 'status': 'import_complete_quality_assessment_pending' if coverage['complete'] else 'partial_or_invalid_import_quality_assessment_pending',
               'expected_outputs': 4, 'accepted_outputs': len(valid), 'conditions': {}, 'model_identity': 'not_verified',
               'matched_decoding': False, 'system_context_matches_Kaggle': False, 'model_calls': 0, 'training_called': False,
               'validation_rows_read': 0, 'reserved_test_rows_read': 0, 'semantic_accuracy': None, 'author_style_accuracy': None}
    for c in parent.CONDITIONS:
        group = [r for r in metrics if r['condition'] == c]
        summary['conditions'][c] = {'expected': 2, 'accepted': len(group), 'count_matches': sum(r['count_matches'] is True for r in group),
                                    'near_copies': sum(r['source_similarity'] >= .9 for r in group)}
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
        check_package(a.package); print('Four frozen prompts and reviewed provenance verified on CPU.')
    else:
        if a.package is None or a.responses_dir is None or a.output_dir is None: p.error('--package, --responses-dir and --output-dir required')
        audit_responses(a.package, a.responses_dir, a.output_dir)


if __name__ == '__main__': main()
