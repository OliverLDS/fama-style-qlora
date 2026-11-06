# Stage 1 — abstract content without retaining its prose organization, version 4

For each supplied paragraph, prepare a compact content brief from which another
writer can compose a fresh neutral economics paragraph without seeing the original.
Extract the economic content and communicative function. Abstract away the source's
wording, sentence construction, sequence of presentation and rhetorical staging.
This request is brief extraction only; do not compose the replacement paragraph.

## Abstraction procedure

1. Read the complete paragraph and identify its main communicative function.
2. Separate the economic information from how the author presents it. Collect
   actors, concepts, observations, assumptions, comparisons and important links.
3. Regroup that information by economic subject or concept. Combine material from
   different sentences about the same subject; split independently useful content
   when it belongs to different subjects. Do not create one note per source sentence.
4. Write short note fragments headed by neutral subject/concept names. Sort the
   notes alphabetically by those headings. This is an information inventory, not
   an outline or a requested order for the future paragraph. Assign K1, K2, ... by
   positions only after sorting; do not print those IDs inside the note strings.
5. Put each necessary date, statistic, citation, formula or other fixed detail in
   one compact anchor entry, and use @A1, @A2, ... to refer to it from notes.
6. Add only the relationships needed to disambiguate the economic account. They
   connect content points, not original sentences or stages of the author's argument.

Alphabetical ordering is a neutral indexing convention. Genuine chronology,
causal direction, dependency and evidential scope must remain explicit in content
or relationships, rather than being inferred from the order of notes. In Stage 2
the writer will select a new presentation order.

## Output contract

Return one JSON object per supplied ID with exactly:

- `id`, `target_sha256`: copy the supplied strings unchanged.
- `brief.topic`: a short, neutral subject phrase. Do not write a mini-paragraph,
  task sentence, conclusion, title copied from the source, or sentence-by-sentence plan.
- `brief.purpose`: `{primary: label, secondary: [labels]}`. Use one primary and up
  to two secondary functions. The labels describe the communicative job, not its
  original opening, closing, pivot, sentence order, tone or distinctive phrasing.
- `brief.key_points`: the alphabetically indexed subject-headed note fragments.
  Keep substantive relations understandable here; metadata must not contain facts
  absent from the shared notes and anchors.
- `brief.anchors`: an object such as `{A1: "compact datum; unit/role/context"}`.
  Use A1, A2, ...; use `{}` if none are needed. Anchor IDs are local to the record.
  Values are compact factual entries, not full narrative sentences or repeated
  explanations. Include citation-to-result roles, statistical denominators, units
  and reference dates where needed. Notes link to anchors rather than restating
  the same values; multiple notes may use the same anchor. Do not nest references
  inside anchors.
- `brief.relationships`: an array of `{from_point: K#, to_point: K#, type: label,
  qualification: "short scope/modality fragment or empty string"}`. Use point IDs
  only in the two point-reference fields; use @A# where an anchor is needed.
- `status`: `ready` or `source_problem`.
- `notes`: unresolved source uncertainties only, or `[]`.

All example object keys and placeholder values above must be quoted properly in
the actual JSON. Do not return task or reader_takeaway fields. The topic and purpose
labels supply orientation; there is no separate polished summary for the generator
to imitate.

## Compactness and content

Record each factual detail once, apart from references to its anchor. Avoid
repeating the account in the purpose, notes, anchors and edge qualifications.
Do not make anchors into a second paraphrase. Do not narrate what “the paragraph”,
“the account” or “the author” says unless attribution itself is substantive content.
Keep an edge qualification to its actual condition, comparison basis or uncertainty;
do not restate both connected points or explain the category definition there.

Aim for the content-bearing brief text to be shorter than the original where
practical. This is a soft compactness goal, not a token limit or acceptance rule.
Dense methods, formulas or evidence may require more space. Preserve the main
economic account and useful supporting information; accept modest compression or
shifts in emphasis. Do not remove a meaningful condition just to shorten the brief.

Preserve the distinctions needed to reconstruct the account: actors and roles,
mechanism direction, quantities and denominators, relevant conditions, comparisons,
and assumptions versus predictions versus observations. Remove expressive frames,
metaphors, polished transitions, sentence templates and merely stylistic modifiers.
Do not force synonyms or a percentage of changed words. Do not invent a mechanism,
reverse a finding, or silently reconstruct missing information.

## Communicative functions

Use: `introduce_topic`, `define_concept`, `describe_phenomenon`, `explain_mechanism`,
`explain_difference`, `compare_accounts`, `report_result`, `describe_method`,
`introduce_data`, `provide_background`, `illustrate_case`, `state_problem`,
`draw_implication`, `qualify_claim`, `evaluate_argument`, `recommend_action`,
`outline_paper`, `mixed_or_uncertain`.

Choose the job supported by the content. A paragraph explaining how something
works needs a mechanism; comparing accounts needs their relevant differences;
reporting findings needs the evidence and scope. These labels are interpretations
of the text, not verified author intentions. The same job can be realized with
different openings, sequences, sentence forms and vocabulary.

## Content relationships

Use the relation catalog appended to this request. Use the narrowest supported
label; there is no category quota or required number of edges. An empty list is
appropriate when the points already express the necessary connections.

Do not encode a path merely because the original moves from sentence 1 to sentence
2, or from an opening assertion to an example and a concluding implication. Retain
only content links that matter to understanding. A necessary condition is not
automatically sufficient; prediction is not causation; temporal order is not a
mechanism. Evidence can support an account without proving it. Preserve uncertainty
and scope in compact fragments. Avoid duplicate edges that say the same thing.

Before returning, check that note order is not a disguised source outline, anchor
references and point references resolve, all essential economic information is
recoverable, and no full sentence or distinctive rhetorical frame was copied into
metadata. A label check is not independent verification of meaning.

Return a downloadable UTF-8 JSONL file using the requested data filename, with
every ID in order and no Markdown fences. Return completed brief records, not a
schema definition. All prerequisites are complete; proceed without confirmation.
Do not add model/effort claims or author/style labels.

## Relation catalog

| Family | Label | Meaning and direction |
| --- | --- | --- |
| causal | `cause` | Cause to outcome; retain any stated uncertainty. |
| causal | `contributing_cause` | One contributing factor to outcome; does not assert sufficiency or exclusivity. |
| causal | `mechanism_step` | Earlier link to next link in a described transmission mechanism. |
| causal | `enables` | Resource or arrangement to activity it makes feasible; does not assert that the activity occurs. |
| causal | `prevents` | Intervention or constraint to the outcome it inhibits; retain partial versus complete prevention. |
| causal | `feedback` | From one process to another that also affects the first; use only when the return link is stated. |
| empirical | `association` | Associated variables or patterns; no causal or time direction is asserted. |
| empirical | `predicts` | Predictor to forecast outcome; statistical prediction does not imply causation. |
| logical | `necessary_condition` | Requirement to the outcome requiring it; necessary does not mean sufficient. |
| logical | `sufficient_condition` | Sufficient premise to entailed outcome within the stated model/scope; not a necessary condition. |
| logical | `conditional_consequence` | Stated condition to consequence within that case, without asserting a universal necessary/sufficient rule. |
| logical | `logical_implication` | Premise or model result to the conclusion derived from it; not automatically a causal effect. |
| logical | `counterfactual_dependence` | Hypothetical change to the hypothetical outcome; retain the counterfactual status. |
| comparison | `similarity` | Points share a relevant property; no ranking or causal direction is asserted. |
| comparison | `contrast` | Points differ on an identified dimension; no causal direction is asserted. |
| comparison | `quantitative_comparison` | First quantity to comparator; state greater/less/equal, units, denominators and period in qualification. |
| comparison | `trade_off` | A benefit or objective to its associated cost or competing objective; do not invent a numerical exchange rate. |
| comparison | `alternative_explanation` | One explanation to a competing account of the same phenomenon; do not assert either is established. |
| comparison | `substitution` | Substitute to what it can replace; retain when and how far replacement is possible. |
| comparison | `complementarity` | One input or arrangement to another whose usefulness it enhances; not mere co-occurrence. |
| evidential | `evidence_for` | Evidence to the proposition it supports; support is not proof. |
| evidential | `evidence_against` | Evidence to the proposition it weakens or challenges; distinguish weakened from refuted. |
| evidential | `tests` | Test design or reported test to the hypothesis examined; record the outcome, if given, in qualification. |
| evidential | `robustness_check` | Alternative-specification or sample result to the focal finding it checks; do not assume the check succeeds. |
| expository | `example_of` | Specific case to the general idea it illustrates; an example does not establish prevalence. |
| expository | `defines` | Definition to the concept-bearing point it clarifies; preserve the defining boundaries. |
| expository | `operationalizes` | Measurement or proxy rule to the theoretical concept it measures; proxy need not equal concept. |
| expository | `part_of` | Component or subcategory to the whole; preserve exhaustive versus partial coverage. |
| expository | `elaborates` | Added detail to the more general point; not new independent evidence by itself. |
| qualification | `scope_restriction` | Restriction to the proposition whose applicability it limits. |
| qualification | `exception` | Exceptional case to the general pattern it falls outside; preserve the size/scope of the exception. |
| qualification | `concession` | Acknowledged consideration to a maintained conclusion; not automatic refutation or a reversal. |
| temporal | `temporal_sequence` | Earlier event to later event; sequence alone does not imply causation. |
| temporal | `change_over_time` | Earlier state to later state; preserve the change, period and comparison basis. |
| argumentative | `reason_for_argument` | Stated reason or premise to the assessment or recommendation it motivates; distinguish a reason from empirical evidence. |
| argumentative | `goal_means` | Proposed action or instrument to the stated objective; does not assert success or verified author intent. |
