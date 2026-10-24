# Task: discover rhetorical structures in supplied paragraphs

Annotate every record in the attached `paragraphs_*.jsonl` file. Use
`annotation_output.schema.json` and the synthetic example as the output contract.
This is taxonomy discovery, not rewriting or author identification.

Apply the same specification to every paragraph. Do not assume its author,
intended register, or membership in a source/target group. Do not classify a
paragraph as an imitation of a particular author. Do not use other paragraphs
to invent information absent from the paragraph being annotated.

## Separate semantics from expression

First record the semantic content, keeping actors/entities, predicates, quantities,
quantifiers, comparisons, attribution, uncertainty, negation and conditions.
Give each atomic claim an ID. Split independently changeable propositions when
possible, but retain the scope of qualifications across connected propositions.
Record explicit relationships between claims, their direction, and their textual
basis. Distinguish causal statements from associations, conditions and inferences.
Mark implicit/uncertain relations; do not turn them into established facts.
Keep hypotheses, expectations, definitions and findings distinct.

Then describe THREE levels of presentation:

1. **Paragraph structure:** main-claim position, ordered rhetorical moves,
   argument progression, qualification placement and connections between moves.
2. **Sentence structure:** actual sentence/clause organization, proposition
   grouping, subordination/coordination, subject selection, active/passive voice
   and the placement/scope of conditions and qualifications.
3. **Language features:** nominalization, terminology, lexical register,
   concision/density, connective and function-word choices, and the *expression*
   of hedging. Hedging strength and technical meaning belong to the semantic
   record and must not be treated as freely adjustable style.

Use interpretable PROVISIONAL labels with a short definition for each. The JSON
field names are fixed; the discovered label names are open. Do not squeeze all
paragraphs into a preselected academic-style stereotype or invent a taxonomy
item merely to populate a field. Record absent, ambiguous or unresolvable
features as uncertainty. Do not equate longer sentences or formal synonyms with
better writing or a distinctive author style.

## Evidence and binding rules

- Link rhetorical moves, sentence features and language features to the claims
  they express. Claim IDs are local to each paragraph.
- Every substantive annotation needs exact evidence copied from that paragraph.
  Use `quote` plus a ONE-BASED `occurrence` when the exact quote repeats. Do not
  fabricate numeric character offsets. Preserve punctuation and Unicode.
- Sentence entries contain the exact source sentence and cover the paragraph in
  reading order. Flag uncertain segmentation, abbreviations or embedded lists.
- A rhetorical progression must cite semantic relation IDs where it asserts a
  logical relation. For purely presentational ordering, an empty relation list
  is allowed; explain that it makes no claim of causal or logical dependence.
- Never infer causation from sentence adjacency, a semicolon, or an author's
  reputation. Do not claim to know the author's intentions.
- Definitions of provisional labels should be reusable and content-independent;
  quotes and claim descriptions stay in evidence/semantic fields.
- Do not invent identity links across separate paragraphs or counterpart texts.

## Self-check and delivery

Check that every supplied ID appears once, every local reference resolves, every
quote occurs verbatim, all substantive claims and qualification scopes are
represented, and no rhetorical label introduces unsupported semantics. Record
uncertainty using `needs_review` and specific `review_notes`; this requests later
automated checking, not mandatory human review. Your self-check is not independent
verification.

Set `annotation_context` to `single_paragraph`. Record a model version only if
actually known; otherwise use null. Do not revise the input text.

Return a downloadable UTF-8 JSONL file, one schema-valid annotation per supplied
paragraph, with no Markdown fences. Use `annotations_A.jsonl` or
`annotations_B.jsonl` to match the input filename; for `paragraphs_A_01.jsonl`,
use `annotations_A_01.jsonl`, and likewise for other chunks. Separately report supplied,
completed, partial and missing IDs, uncertainty counts, and limitations. If output
limits prevent completion, deliver the completed rows and explicitly list the
remaining IDs; never claim complete coverage or fabricate empty annotations.
