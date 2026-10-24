# Fixed-unit feature check

Read `paragraphs.jsonl`, `feature_checklist.json`, and `output.schema.json`.
If an attachment is unavailable, stop and name it. Do not design replacements.

This task measures the consistency of a small checklist, not completeness of an
author-style taxonomy. Assess the actual text. Do not rewrite it, guess its author,
consult earlier annotations, or create new categories, units, claims or relations.

## Fixed scope

Each paragraph includes frozen operational units and an `expected_cells` list.
Return EVERY cell exactly once with its supplied cell/unit/feature/canonical IDs.
Do not select additional units, omit negative findings, or merge cells. The unit
boundaries are an operational snapshot, not verified grammatical ground truth.
Judge each unit's suitability separately. If a boundary obstructs a decision,
mark the relevant observation unresolved/not applicable and explain the problem;
do not silently change the boundary.

Always read the full paragraph. For a cell, judge only its specified unit. Positive
evidence must fall within that unit's character range. For sentence-pair cells,
do not infer a logical relationship merely because the sentences are adjacent.
A rhetorical move may span different boundaries; flag that mismatch explicitly.

For predicates, the fixed subsentence span can contain more than one construction.
Active, copular and passive are separate yes/no questions about supported
predications within the span, not a forced exclusive classification of the whole
sentence. Do not classify the main clause from an embedded clause's form.

## Required state for every cell

- `present`: the category is actually supported within this unit. Give exact
  positive evidence and preserve its scope.
- `absent`: the unit is suitable and has been examined, but the feature is absent.
  An empty evidence list is allowed; explain the inspected scope. Do not treat
  omission, uncertainty or an unread unit as absence.
- `unresolved`: the text, unit boundaries or required bindings do not settle it.
  Give the ambiguity and evidence where available. Do not guess to improve agreement.
- `not_applicable`: the feature's scope is genuinely inapplicable to this unit,
  such as a nonpredicative fragment for a voice question. Explain why. It is not
  a substitute for absent or uncertain.

State, applicability and evidence are not interchangeable. Do not force all cells
to positive, negative or unresolved just to finish the form.

## Semantic safeguards

The read-only semantic context is a prior interpretation, not a verified graph.
Use only its existing claim/relation IDs for bindings. Do not borrow an unrelated
relation, invent an ID, or repair the graph in this output. If a required binding
is missing/wrong, flag it. Grammar observable in the text can still be assessed
while its semantic binding is uncertain. A positive evidential-support progression
requires the category's actual source-backed relation conditions; otherwise abstain.

Keep actors, quantities, dates, conditions, negation, modality, comparison and causal
direction unchanged. Interpret a word's function in context: calendar names are
not modals, and grammatical coordination is not automatically discourse addition.
The comparison feature retains its supplied explicit-comparator definition. Record
implicit-comparison or exemplification coverage concerns as notes; do not change
the catalogue to fix them during this check.

For evidence, copy exact quotes with one-based occurrence in the FULL paragraph.
The program will resolve offsets; do not calculate new boundaries yourself.
Copy all version/hash metadata from the input, not another round or file.

## Delivery

Return downloadable `fixed_unit_annotations.jsonl`, one object per supplied
paragraph, following the schema, plus `coverage_report.md`. Include one unit
assessment per supplied unit and one observation per expected cell. Report
supplied/completed/missing IDs and cell totals; retain uncertainty and limitations.
Record the model version only if known; otherwise use null. Do not claim independent
verification or require human review. Do not compare with other conversations.
