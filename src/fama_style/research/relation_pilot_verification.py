"""Prepare source-only fidelity/readability checks of preferred local merges.

Preparation only: no model calls, training, semantic certification or label edits.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from .author_control_replication import verify_files
from .relation_pilot_review import file_manifest, file_sha, read_rows
from fama_style.rhetoric.experiment import fresh, write_json, write_lines

PROTOCOL = 'relation_pilot_verification_v1'
PROMPT = '''Check a proposed local sentence merge against the supplied source paragraph.
Use only these supplied texts. Do not consult remembered papers, other answers,
prior labels or outside material. Work as a verifier, not as the candidate's
advocate. Same-model and possible chat retrieval limit independence; this is
an automated second assessment, not independent semantic certification.

Each item gives an opaque item_id, source_paragraph, left_sentence,
right_sentence and candidate. The two sentences are adjacent in the source.
Assess ONLY replacement of that pair by the candidate, in its paragraph context.
The candidate is fixed: do not repair it or produce a better rewrite. Do not
require reproduction of an author's wording or infer the author. Do not assume
the candidate is preferred, faithful, or better simply because it was supplied.

First extract fresh source_units V1, V2, ... covering all substantive content of
the TWO source sentences, including questions, definitions, intentions and
hypotheses. Separate distinct propositions. For each, retain actors/roles,
conditions, quantities, negation, modality and attribution. Record exact quotes
from source_paragraph and candidate with one-based occurrence numbers. Evidence
may span sentences; a pronoun alone does not repeat a complete proposition.
Use preserved, omitted, changed or uncertain for each unit's candidate alignment.
For an omitted unit, candidate_evidence may be empty. Do not resolve ambiguity
by guessing. Facts outside the pair need not be repeated by its replacement.

Use surrounding source text only to check referents or restrictions already
needed by the pair. Record context_evidence for any such use. Clarifying an
unambiguous antecedent is not automatically a new claim; importing a substantive
neighboring claim into the candidate must be identified as added content relative
to the pair, even when that claim exists elsewhere in the paragraph. Never use
absent document context or presumed economic knowledge to fill a gap.

Explicitly check additions, omissions, actor identity, quantities/quantifiers,
negation, modality, conditions, attribution, speech acts and causal/inferential
direction. Especially check whether a new because/since/therefore connection is
supported, whether a condition now governs an extra clause, and whether a
question, conjecture or intended result became an assertion. Synonym changes,
removed repeated actor names, grammatical restructuring and different protected
word counts are not errors by themselves. Explain substantive effects, not
literal edit distance. List added content/relations separately with exact candidate
evidence and source_support as absent or uncertain. Do not invent a source quote
for an absent claim. Use source units to identify omissions rather than hiding
them in a general verdict.

Give fidelity_verdict independently of readability:
- faithful: all substantive content and force are retained, with no unsupported
  addition or relation; wording and organization may differ;
- local_shift: a limited, explainable interpretive change while the main economic
  relationships remain intact; state precisely what changed;
- material_shift: changed economic/logical content, such as reversed causality,
  altered actor/quantity, lost substantive claim or materially changed restriction;
- unresolved: ambiguity, missing context or insufficient evidence prevents judgment.
Do not force faithfulness or a rejection rate. A local_shift is not automatically
acceptable for training: downstream selection is separate. Do not penalize an
unresolved referent shared unchanged by source and candidate as an invented
candidate error; distinguish pre-existing uncertainty from a new change.

Separately judge readability_in_context as merged_better, separated_better,
comparable or unresolved. Consider clarity, economic organization, scope,
attribution, referent resolution and sentence density. Concision is not always
an improvement, and a semicolon does not establish readability. Judge organization
even when fidelity has a problem; report that problem in the fidelity fields.
Provide exact evidence and rationale. Assess candidate_single_sentence as yes,
no or uncertain, treating abbreviations appropriately. Decisions apply to this
pair only; they do not license chaining several merges.

Follow verification_output.schema.json exactly. Copy source_paragraph,
left_sentence, right_sentence and candidate exactly. Cover every item once and
return a downloadable UTF-8 JSONL file without Markdown fences. Separately
provide a brief coverage/limitations report. Preserve the first complete response.
Visible model and effective thinking setting should be null if unknown; a requested
High setting does not establish effective effort. earlier_judgments_supplied is
false and independently_verified is false. Do not claim hidden-context isolation.
'''


def output_schema():
    string = {'type': 'string'}
    arr = lambda s: {'type': 'array', 'items': s}
    obj = lambda p: {'type': 'object', 'properties': p, 'required': list(p), 'additionalProperties': False}
    enum = lambda values: {'type': 'string', 'enum': list(values)}
    ev = arr(obj({'quote': {'type': 'string', 'minLength': 1}, 'occurrence': {'type': 'integer', 'minimum': 1}}))
    unit = obj({'unit_id': {'type': 'string', 'pattern': '^V[1-9][0-9]*$'},
        'proposition': {'type': 'string', 'minLength': 1},
        **{k: arr(string) for k in ('actors_and_roles', 'conditions', 'quantities', 'negation', 'modality', 'attribution')},
        'source_evidence': ev, 'candidate_evidence': ev,
        'alignment': enum(('preserved', 'omitted', 'changed', 'uncertain')), 'notes': string})
    return obj({'annotation_version': {'const': PROTOCOL}, 'item_id': string,
        **{k: {'type': 'string', 'minLength': 1} for k in ('source_paragraph', 'left_sentence', 'right_sentence', 'candidate')},
        'source_units': {'type': 'array', 'items': unit, 'minItems': 1},
        'added_content': arr(obj({'description': string, 'candidate_evidence': ev,
            'source_support': enum(('absent', 'uncertain')), 'explanation': string})),
        'context_evidence': ev,
        'fidelity_verdict': enum(('faithful', 'local_shift', 'material_shift', 'unresolved')),
        'fidelity_rationale': string,
        'issue_categories': arr(enum(('added_relation', 'added_claim', 'omitted_claim', 'changed_actor',
            'changed_quantity_or_quantifier', 'changed_negation', 'changed_modality', 'changed_condition',
            'changed_attribution', 'changed_speech_act', 'unclear_reference', 'other'))),
        'readability_in_context': enum(('merged_better', 'separated_better', 'comparable', 'unresolved')),
        'readability_rationale': string,
        'readability_evidence': arr(obj({'text_origin': enum(('source', 'candidate')),
            'quote': {'type': 'string', 'minLength': 1}, 'occurrence': {'type': 'integer', 'minimum': 1}})),
        'candidate_single_sentence': enum(('yes', 'no', 'uncertain')),
        'needs_external_context': {'type': 'boolean'}, 'review_notes': arr(string),
        'provenance': obj({'visible_model_label': {'type': ['string', 'null']},
            'thinking_setting': {'type': ['string', 'null']},
            'earlier_judgments_supplied': {'const': False}, 'independently_verified': {'const': False}})})


def payload(row, boundary):
    ss = {s['sentence_id']: s['exact_text'] for s in row['source_sentences']}
    identity = row['paragraph_id'] + ':' + boundary['boundary_id']
    return {'item_id': 'RV_'+hashlib.sha256(identity.encode()).hexdigest()[:16],
        'source_paragraph': row['source_paragraph'],
        'left_sentence': ss[boundary['left_sentence_id']], 'right_sentence': ss[boundary['right_sentence_id']],
        'candidate': boundary['merge_candidate']}


def allocate(jobs, size=4):
    """Spread repeated paragraphs and papers across balanced fixed-size batches."""
    groups = defaultdict(list)
    for job in jobs: groups[job['paragraph_id']].append(job)
    batches = [[] for _ in range((len(jobs)+size-1)//size)]
    for pid in sorted(groups, key=lambda p: (-len(groups[p]), p)):
        for job in sorted(groups[pid], key=lambda j: j['payload']['item_id']):
            choices = [i for i, batch in enumerate(batches) if len(batch)<size and all(j['paragraph_id']!=pid for j in batch)]
            if not choices: raise ValueError('cannot_avoid_repeated_paragraph_in_batch')
            index = min(choices, key=lambda i: (sum(j['paper_id']==job['paper_id'] for j in batches[i]), len(batches[i]), i))
            batches[index].append(job)
    return batches


def build(audited, selection_path, output, expected_merges=52):
    verify_files(audited)
    summary = json.loads((audited/'summary.json').read_text())
    if summary['invalid_records'] or not summary['records'] == summary['mechanically_valid_records']:
        raise ValueError('invalid_stage2_records')
    details = read_rows(audited/'row_audit.private.jsonl')
    if any(not d['mechanically_valid'] or d['errors'] for d in details): raise ValueError('invalid_stage2_rows')
    selection = {r['paragraph_id']: r for r in json.loads(selection_path.read_text())}
    diagnostics = {(d['paragraph_id'], d['boundary_id']): d for d in read_rows(audited/'candidate_diagnostics.private.jsonl')}
    jobs = []
    for row in read_rows(audited/'annotations.private.jsonl'):
        for b in row['boundaries']:
            if b['presentation_preference'] != 'MERGE': continue
            if b['permissible'] not in ('MERGE', 'BOTH') or not b['merge_candidate']: raise ValueError('invalid_preferred_candidate')
            pid = row['paragraph_id']; diag = diagnostics[pid, b['boundary_id']]
            jobs.append({'paragraph_id': pid, 'paper_id': selection[pid]['paper_id'],
                'boundary_id': b['boundary_id'], 'risk_flags': diag['risk_flags'], 'payload': payload(row, b)})
    if len(jobs) != expected_merges or len({j['payload']['item_id'] for j in jobs}) != len(jobs):
        raise ValueError('preferred_candidate_count_or_duplicate')
    batches = allocate(jobs)
    fresh(output); (output/'common').mkdir(); (output/'responses').mkdir()
    (output/'common/PROMPT.md').write_text(PROMPT)
    write_json(output/'common/verification_output.schema.json', output_schema())
    mapping = []
    for bi, batch in enumerate(batches, 1):
        folder = output/('batches/batch_%02d' % bi); folder.mkdir(parents=True)
        write_lines(folder/'items.jsonl', [j['payload'] for j in batch])
        (folder/'TASK.md').write_text(PROMPT+'\nProcess all %d attached items. Save responses_%02d.jsonl and coverage_%02d.md.\n' % (len(batch), bi, bi))
        mapping.extend({**{k: j[k] for k in ('paragraph_id', 'paper_id', 'boundary_id', 'risk_flags')},
                        'item_id': j['payload']['item_id'], 'batch': bi} for j in batch)
    write_json(output/'PRIVATE_MAPPING.json', mapping)
    report = {'protocol': PROTOCOL, 'status': 'prepared_not_executed', 'items': len(jobs),
        'paragraphs': len({j['paragraph_id'] for j in jobs}), 'papers': len({j['paper_id'] for j in jobs}),
        'batches': len(batches), 'batch_sizes': [len(b) for b in batches],
        'items_with_surface_flags': sum(bool(j['risk_flags']) for j in jobs),
        'selection': 'all_stage2_merge_preferred_candidates_no_surface_filter',
        'same_paragraph_repeated_within_batch': False,
        'earlier_judgments_exported': False, 'reference_paragraphs_exported': False,
        'stage2_claims_exported': False, 'responses_received': 0,
        'fidelity_and_readability_assessed_separately': True, 'automatic_training_acceptance': False,
        'independent_evaluation_claimed': False, 'reserved_content_used': False,
        'new_model_calls': 0, 'training_called': False, 'gpu_jobs_started': 0}
    write_json(output/'summary.json', report)
    (output/'START_HERE.md').write_text('''# Verify fixed local candidates: fidelity and readability

52 items from 27 source paragraphs across four papers, in 13 batches of four.
For each batches/batch_01 through batch_13, paste TASK.md and attach ONLY that
batch's items.jsonl and common/verification_output.schema.json. Use GPT-6 and
request High thinking effort. If effective effort is unclear or changes, report
it as uncertain; do not label every answer High simply because you selected it.

Save first complete unedited responses_01.jsonl through responses_13.jsonl in
this packet's responses/ directory, along with coverage_XX.md. Notify Codex when
all thirteen are ready. Record known retries or settings changes in your reply
or SESSION_NOTES.md; unknown details may remain unknown. No credentials needed.

Do not upload this file, private mapping, original author paragraphs, earlier
responses, labels, rationales, warnings, source metadata or other batch answers.
Candidate and two-sentence source layout are necessarily visible. Earlier
selection/preferences are withheld from the verifier, so this masks prior
judgments, not the merge operation. Every selected candidate is checked regardless
of surface flags. No paragraph repeats within a batch; paragraphs recur across
batches when several local candidates are checked.

The full neutral source paragraph supplies context. Only content of the indicated
pair must be preserved; do not import other paragraph claims into its replacement.
Extract fresh V-units, assess the fixed candidate without repairing it, and judge
fidelity separately from readability. Local shifts must be described, not silently
accepted for training. Abstention is valid. Checks are local, not permission to
chain merges. No classifier fit or Kaggle job is needed.

Separate chats can reduce direct carryover but cannot establish independence from
pretraining, same-model self-review or automatic chat retrieval. This is an
exploratory second assessment, not an independent semantic certificate.
''')
    (output/'SESSION_NOTES.md').write_text('Visible model label:\nRequested thinking setting:\nKnown effective setting (or uncertain):\nChat/context arrangement (if known):\nRetries or changed outputs (if known):\n\nNo credentials.\n')
    # Mutable user session form stays outside the frozen file list.
    file_manifest(output, protocol=PROTOCOL, code_sha256=file_sha(Path(__file__)),
        stage2_audit_manifest_sha256=file_sha(audited/'manifest.json'), selection_sha256=file_sha(selection_path))
    manifest = json.loads((output/'manifest.json').read_text()); manifest['files'].pop('SESSION_NOTES.md')
    write_json(output/'manifest.json', manifest)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('audited', 'selection', 'output'): p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); print(json.dumps(build(a.audited, a.selection, a.output), indent=2))


if __name__ == '__main__': main()
