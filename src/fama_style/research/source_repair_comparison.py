"""Prepare a private 2x2 source-repair/packing diagnostic; no model execution."""
import argparse
import copy
import json
from pathlib import Path

from . import repaired_source_bindings as bindings
from fama_style.rhetoric import paper_plan_comparison as parent
from fama_style.rhetoric import operation_expansion as base
from fama_style.rhetoric.concise_plan import file_sha, read_lines
from fama_style.rhetoric.experiment import fresh, write_json, write_lines
from fama_style.prompting.economics import make_prompt

ROOT = bindings.ROOT
PROTOCOL = 'source_repair_comparison_v1'
CONDITIONS = (('original', 'ordinary'), ('original', 'planned'),
              ('repaired', 'ordinary'), ('repaired', 'planned'))
TASK = '''Process the attached jobs_XX.jsonl using response.schema.json.
Follow each job's prompt independently. Return one first-attempt paragraph per job.
Do not browse, retrieve original papers, use another job's facts, select among
candidates, or add explanations or critiques. Do not rewrite earlier responses.
Return a downloadable UTF-8 JSONL file named responses_XX.jsonl, matching the
attached batch number, with exactly these fields per line:
{"job_id":"COPY_FROM_JOB","prompt_sha256":"COPY_FROM_JOB","output":"YOUR_PARAGRAPH"}
Copy both identifiers exactly. The paragraph-only instruction applies inside
output. No Markdown fences or model-version guesses. If incomplete, preserve
completed rows and report missing job IDs separately; do not invent placeholders.
'''
START = '''# Original versus repaired inputs: 16-output diagnostic

Prepared only. No training, GPU job or model call has been made.

Use GPT-6 with High thinking effort for ALL FOUR batches, matching the earlier
external-model setting. Keep the model and visible settings unchanged throughout.
If that model/setting is unavailable, record the actual choice before starting
and use it consistently. ChatGPT does not expose temperature, seed or all hidden
decoding/system settings: these are uncontrolled, not claimed identical.

For each XX = 01, 02, 03, 04:
1. Prefer a separate new conversation with the same visible model/settings.
2. Attach ONLY batches/jobs_XX.jsonl and response.schema.json.
3. Paste TASK_PROMPT.md. Keep the first returned responses_XX.jsonl unchanged.
4. Save it under research_runs/source_repair_comparison_v1/responses/.

Each batch has four jobs, one per paper and one per experimental condition.
No batch contains two variants of the same paragraph. Four conversations are
preferred to reduce direct exposure to paired inputs; memory can still cross
conversations. No deleting chats, clearing memory, or proving independence is
required. If using one conversation or another arrangement, record that fact.

Optionally put the actual model label, visible effort, dates, chat arrangement,
changes in settings and interruptions in responses/notes.txt. Do not include
credentials. Hidden settings are unknown. Save partial/invalid first attempts;
do not retry or silently repair files to improve the result.

Do not attach private_key.private.json, source cards/specifications, manifests,
reports, reference paragraphs, PDFs, old answers, this instruction file or an
entire project directory. These are private training-paper diagnostics; generated
paragraphs and source/reference text must remain outside the public repository.

After saving all four responses, tell Codex they are ready. No human scoring is
required. This is not a genuinely blinded or independent evaluation: formats
reveal plan presence, examples have been inspected, the model may know the source
papers, and shared context/memory is possible. Source evidence and grouping are
reference-informed annotations, not a deployable source-only mapper. Four cases
support mechanism diagnosis, not population-level or held-out author-style claims.
'''


EVALUATION = """# Prespecified interpretation: source repair x sentence-packing plan

Analyze all 16 new first attempts; historical outputs are context, not controls.
Report missing/invalid outputs without silently replacing them. Freeze raw files
and their hashes before assessing generated text. Do not claim matched hidden
decoding or independent model judgment from ChatGPT-visible settings.

For each paper, compare repaired minus original within ordinary and within
planned rewriting, then planned minus ordinary within each input variant. Show
case-level outcomes; four inspected cases do not warrant significance or held-out
generalization claims. Both arms have corrected annotation policy; this tests
source-plus-grounded-evidence repair, not bare-source wording in isolation.

Separate axes:
1. Supplied-input fidelity: added/lost claims, actors, quantities, attribution,
   uncertainty, conditions, causal direction and scope. Ground verdicts in exact
   source/output evidence. Same claim IDs express correspondence across revisions,
   not proof that original and repaired propositions are semantically identical.
2. Reference-claim recovery: use the private reference and nine-edit repair audit
   only after generation. Improvements in reference resemblance are not necessarily
   improvements in source fidelity. Record appropriate new information separately
   from unsupported invention. Do not reward recognition/correction substitution
   in ME1997 as a style improvement. NL2003's comparator stays unspecified.
3. Copying: report unchanged outputs, normalized token overlap and edit distance
   against BOTH source versions, plus evaluation-only reference overlap. Repairing
   an input may itself make it closer to the reference; do not attribute that
   baseline shift to the generator. A copying threshold is a diagnostic, not a
   semantic or style grade.
4. Organization/style: assess grouping realization separately from changes in
   theme, consequence expression and label continuity. Count matching alone is
   not operation adherence or distinctive author style. No new richer-style
   instruction is introduced in this experiment.

Assess drift severity and worthwhile style gain separately; acceptability can
include modest drift, with the actual changed claim and benefit disclosed.
Do not collapse all axes into a single lexical score or require verbatim reference
reconstruction. Surface checks are fallible; no semantic accuracy is reported
until output-level evidence assessment has actually been performed.

Stratify the two bounded cases (FJ1983/F1980) from the qualified diagnostics
(NL2003/ME1997). Useful evidence would be more source-faithful recovery of restored
economics without simply copying the repaired input. A negative or mixed outcome
is informative. No training, mapper fitting, additional prose corpus or reserved
test use is authorized by preparation or import of these responses.
"""


def original_spec(repaired_spec, annotation, source, target):
    """Freshly resolve old source quotes under the reviewed annotation policy.

    Broad whole-claim protections and epistemic categories match the repaired
    arm. No repaired text or missing economics is inserted into the original.
    Fine facets are audit-only and omitted here, not passed to either generator.
    """
    spec = copy.deepcopy(repaired_spec)
    spec['expected_source_sha256'] = base.sha(source)
    spec['expected_target_sha256'] = base.sha(target)
    old = {c['claim_id']: c for c in annotation['claims']}
    for claim in spec['claims']:
        evidence = old[claim['claim_id']]['source_evidence']
        if len(evidence) != 1: raise ValueError('requires_review_of_multispan_claim')
        claim['source_quote'] = evidence[0]['quote']
    claims = {c['claim_id']: c['source_quote'] for c in spec['claims']}
    for relation in spec['relations']:
        for side in ('from', 'to'):
            relation[side+'_quote'] = claims[relation[side+'_claim_refs'][0]]
        # The future-wage clause supports this edge in BOTH original/repaired input.
        if spec['paper_key'] == 'F1980' and relation['relation_id'] == 'r3':
            relation['from_quote'] = repaired_spec['relations'][-1]['from_quote']
    spec['source_facets'] = []
    # Source-defined stages differ in wording; never add the repaired definition.
    if spec['paper_key'] == 'FJ1983':
        constraint = spec['intra_claim_constraints'][0]
        constraint['parts'][0]['quote'] = 'approval'
        constraint['rule'] = 'preserve_approval_before_monitoring'
    spec['changes_reviewed'].append({'action': 'original_source_rebound_under_same_policy',
        'reason': 'Use original exact claim quotes, corrected relation endpoints and source-owned conditions; never insert repaired facts.'})
    return spec


def load_cards():
    root = ROOT/'research_runs/repaired_source_bindings_v1'
    release = root/'release_v1'
    manifest = json.loads((release/'manifest.json').read_text())
    if manifest['code_sha256'] != file_sha(Path(bindings.__file__)): raise ValueError('binding_code_changed')
    bindings.verify_files(release, manifest['files'])
    for path, digest in manifest['inputs'].items():
        if file_sha(ROOT/path) != digest: raise ValueError('binding_parent_changed')
    package = ROOT/'research_runs/paper_plan_comparison_v1/package_v1'
    parents = {r['id']: r for r in parent.check_package(package)['rows']}
    op_root = ROOT/'research_runs/operation_expansion_v1'
    infos = {r['case_id']: r for r in base.check_package(op_root/'package_v1')['records']}
    audit = op_root/'response_audit_v1'
    bindings.verify_files(audit, json.loads((audit/'manifest.json').read_text())['files'])
    annotations = {r['case_id']: r['annotation'] for r in read_lines(audit/'annotations.checked.private.jsonl')}
    spec_path = root/'specification_v1/source_bindings.private.json'
    specs = {s['parent_case_id']: s for s in json.loads(spec_path.read_text())['records']}
    repairs = {r['id']: r for r in read_lines(ROOT/'data/meaning_alignment_repair_v1/repaired_pairs.private.jsonl')}
    cards = []
    for repaired in read_lines(release/'paired_cards.private.jsonl'):
        cid = repaired['parent_case_id']; prior = parents[cid]
        row = copy.deepcopy(repairs[prior['source_id']]); row['plain_english'] = prior['source']
        spec = original_spec(specs[cid], annotations[cid], prior['source'], prior['reference'])
        original = bindings.rebind(row, spec, annotations[cid],
              {**prior['payload'], 'target_record_units': infos[cid]['record']['target_units']})
        if original['plan'] != repaired['plan']: raise ValueError('plan_changed_across_input_arms')
        for card, variant in ((original, 'original'), (repaired, 'repaired')):
            card = copy.deepcopy(card); card['input_variant'] = variant
            card['id'] = cid+'_'+variant
            cards.append(card)
    return cards, {'binding_manifest_sha256': file_sha(release/'manifest.json'),
                   'binding_specification_sha256': file_sha(spec_path), 'parent_package_manifest_sha256': file_sha(package/'manifest.json'),
                   'frozen_train_sha256': bindings.TRAIN_SHA}


def prompt_for(card, mode):
    return make_prompt(card['record']['source'], mode, card['evidence'], card['plan'] if mode == 'planned' else None)


def build_jobs(cards):
    papers = sorted({c['paper_key'] for c in cards}, key=lambda p: base.sha(PROTOCOL+'|'+p))
    if len(papers) != 4 or len(cards) != 8: raise ValueError('four_paired_papers_required')
    lookup = {(c['paper_key'], c['input_variant']): c for c in cards}
    batches = []; mapping = []
    for b in range(4):
        jobs = []
        for index, paper in enumerate(papers):
            variant, mode = CONDITIONS[(index+b) % 4]; card = lookup[paper, variant]
            prompt = prompt_for(card, mode)
            # Reference is evaluation-only; never serialize paired audit metadata.
            if card['record']['reference'] in prompt: raise ValueError('full_reference_leakage')
            job = {'job_id': 'J'+base.sha(PROTOCOL+'|'+card['id']+'|'+mode)[:16],
                   'prompt_sha256': base.sha(prompt), 'prompt': prompt}
            jobs.append(job)
            mapping.append({'job_id': job['job_id'], 'source_id': card['id'], 'paper_key': paper,
               'input_variant': variant, 'mode': mode, 'condition': variant+'_'+mode,
               'prompt_sha256': job['prompt_sha256'], 'batch_file': f'jobs_{b+1:02d}.jsonl',
               'response_file': f'responses_{b+1:02d}.jsonl'})
        batches.append(sorted(jobs, key=lambda j: base.sha(j['job_id'])))
    if len({j['job_id'] for j in mapping}) != 16: raise ValueError('duplicate_or_missing_job')
    return batches, mapping


def prepare(output):
    cards, provenance = load_cards(); batches, mapping = build_jobs(cards)
    fresh(output); (output/'batches').mkdir()
    for i, jobs in enumerate(batches, 1): write_lines(output/'batches'/f'jobs_{i:02d}.jsonl', jobs)
    repair_log = ROOT/'data/meaning_alignment_repair_v1/repair_log.private.json'
    write_json(output/'private_key.private.json', {'protocol': PROTOCOL, **provenance, 'rows': cards, 'jobs': mapping,
        'repair_changes_evaluation_only': json.loads(repair_log.read_text()), 'repair_log_sha256': file_sha(repair_log),
        'planned_model': 'GPT-6', 'planned_thinking_effort': 'High', 'actual_generation_metadata': 'pending',
        'hidden_decoding_controlled': False, 'grouping_source': 'unchanged reference-informed oracle plan',
        'annotation_policy': 'Original and repaired inputs both freshly rebound with corrected relations; no legacy annotation-error control.',
        'batch_design': 'Four-paper Latin rotation: each batch has one of each condition and one input per paper; hash-ordered jobs.'})
    (output/'TASK_PROMPT.md').write_text(TASK, encoding='utf-8')
    (output/'START_HERE.md').write_text(START, encoding='utf-8')
    write_json(output/'response.schema.json', parent.RESPONSE_SCHEMA)
    (output/'EVALUATION_PLAN.md').write_text(EVALUATION, encoding='utf-8')
    summary = {'protocol': PROTOCOL, 'status': 'prepared_not_executed', 'papers': 4, 'requested_outputs': 16,
        'conditions': [a+'_'+b for a, b in CONDITIONS], 'batches': 4, 'jobs_per_batch': 4,
        'binding_policy_matched': True, 'grouping_plan_fixed_across_input_arms': True,
        'decoding_settings_identical_verified': False, 'model_calls': 0, 'training_called': False,
        'gpu_jobs_started': 0, 'validation_rows_read': 0, 'reserved_test_rows_read': 0,
        'semantic_accuracy': None, 'author_style_accuracy': None,
        'qualified_cases': ['NL2003: unspecified comparison/context', 'ME1997: recognition versus correction'],
        'evaluation_axes': ['supplied-input fidelity', 'reference-claim recovery', 'copying relative to both inputs', 'organization and nonpacking style'],
        'limits': ['Four inspected training cases; exploratory mechanism diagnostic.',
            'Reference alignment is not source fidelity; input repairs can legitimately change claims.',
            'Compare all four new conditions; historical outputs are not concurrent controls.',
            'Oracle evidence/grouping, hidden settings, memory and batch context limit causal interpretation.',
            'No distinctive author-style or source-only deployment claim.']}
    write_json(output/'summary.json', summary)
    write_json(output/'manifest.json', {'protocol': PROTOCOL, **provenance, 'code_sha256': file_sha(Path(__file__)),
        'prompt_builder_sha256': file_sha(ROOT/'src/fama_style/prompting/economics.py'),
        'files': {str(p.relative_to(output)): file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    print('Prepared four balanced batches / 16 first-attempt outputs; no generation or training.')


def check_package(package):
    manifest = json.loads((package/'manifest.json').read_text())
    if manifest['protocol'] != PROTOCOL or manifest['code_sha256'] != file_sha(Path(__file__)): raise ValueError('comparison_code_changed')
    if manifest['prompt_builder_sha256'] != file_sha(ROOT/'src/fama_style/prompting/economics.py'): raise ValueError('prompt_builder_changed')
    bindings.verify_files(package, manifest['files'])
    cards, provenance = load_cards(); batches, mapping = build_jobs(cards)
    key = json.loads((package/'private_key.private.json').read_text())
    if key['rows'] != cards or key['jobs'] != mapping or any(key[k] != v or manifest[k] != v for k,v in provenance.items()): raise ValueError('comparison_parent_changed')
    if sorted(p.name for p in (package/'batches').iterdir()) != [f'jobs_{i:02d}.jsonl' for i in range(1,5)]: raise ValueError('unexpected_batch_files')
    for i, jobs in enumerate(batches, 1):
        if read_lines(package/'batches'/f'jobs_{i:02d}.jsonl') != jobs: raise ValueError('comparison_jobs_changed')
    return key


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path); parser.add_argument('--check-package', type=Path)
    args = parser.parse_args(argv)
    if bool(args.output_dir) == bool(args.check_package): parser.error('Choose output-dir or check-package')
    if args.check_package: check_package(args.check_package); print('Package and parent hashes verified.')
    else: prepare(args.output_dir)


if __name__ == '__main__': main()
