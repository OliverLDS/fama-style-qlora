Check a proposed local sentence merge against the supplied source paragraph.
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
