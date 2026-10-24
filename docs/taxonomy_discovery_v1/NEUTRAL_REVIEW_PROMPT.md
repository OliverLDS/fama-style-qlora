# Task: design a faithful neutral realization, using the supplied rewrite as reference

Use `neutral_reference_pairs.jsonl`, `annotation_output.schema.json`,
`neutral_review_output.schema.json` and the synthetic example.
Chunked input filenames such as `neutral_reference_pairs_01.jsonl` are equivalent.
Use ANNOTATION_PROMPT.md as the annotation specification for the final proposal;
this neutral-review prompt controls which text may be revised in this task.

Each pair contains an original paragraph and a previously generated plain-English
reference. The reference is a starting point, NOT ground truth and NOT a required
final wording or rhetorical organization. The original supplies the claims to
preserve. Do not alter the archived original or pretend it has been independently
verified against a PDF.

## Quality assessment

Check the reference against the original for lost/added claims; changed actors,
quantities, comparisons, attribution or causal direction; changed conditions,
negation, uncertainty and scope; and unnecessary formality or superficial synonym
replacement. Distinguish semantic problems from presentation choices.

Choose `retain`, `revise`, or `unresolved` per pair:

- **retain:** the reference is faithful, clear and reasonably neutral. Do not
  force changes merely to create style contrast.
- **revise:** create a new neutral realization, using the original to restore
  any missing claims and correct semantic drift. Also revise organization when
  the reference is an inadequate neutral expression. Record exactly why.
- **unresolved:** uncertain original transcription/meaning prevents a faithful
  result. Leave the proposal and its annotation null and explain the uncertainty.
  Do not guess, silently reconstruct text, or request mandatory human review.

Aim for an economically literate reader who is not a specialist in this paper's
methods. Simplify expression while preserving every claim and qualification.
Do not add teaching material, examples, mechanisms, citations or conclusions.
Do not manufacture a universally short/simple structure or erase technical
distinctions. A neutral paragraph can still have subordinate clauses, technical
terms and careful qualifications.

## Proposed neutral taxonomy

For retained or revised text, annotate the EXACT `proposed_neutral_text` using
the same three-level specification in ANNOTATION_PROMPT.md. You may change the
proposed neutral taxonomy when the supplied reference is a poor rewrite, but
every proposed structure must be grounded in the actual new neutral text.
Do not claim that a redesigned taxonomy describes the unchanged reference.

Create a complete annotation under `proposed_neutral_annotation`; set its
`paragraph_id` to the supplied `proposal_id` and `annotation_context` to
`paired_neutral_design`. Annotate only the final neutral realization's structure,
not the original author's structure. Keep evidence quotes anchored to the final
neutral text. Record unsupported or ambiguous features explicitly.

Also record issues with evidence from both original and reference where available,
and a short list of structural revisions. If a claim was omitted from the
reference, its reference-side evidence can be empty; identify the original
evidence instead. Do not invent a source-side counterpart for an omitted claim.

These are PROPOSED repairs and provisional annotations, not accepted replacements
for the repository's dataset. The annotation has seen the paired original;
do not describe it as a blind independent annotation. A later independent
reannotation can assess whether that context influenced its structural labels.

## Delivery

Return `neutral_proposals.jsonl` (or `neutral_proposals_01.jsonl` etc. to match a
chunked input), one object per pair, following
`neutral_review_output.schema.json`. Preserve pair IDs, reference IDs and proposal
IDs. Do not include revised original paragraphs or silently change existing rows.
Separately report counts for retain/revise/unresolved, completed and missing IDs,
semantic versus presentation issues, and limitations. If incomplete, deliver
finished rows and list the remaining IDs. Do not claim independent verification.
