# Paired annotation output v1

One UTF-8 JSON object per line, with the same `paragraph_id` and exactly the same
`original_paragraph` as the supplied text. Downloadable response and coverage
filenames are specified in each batch's TASK.md. The machine-readable schema is
`common/annotation_output.schema.json` inside the private package.

- `plain_english`, `rewrite_status`, `rewrite_notes`: generated source, accepting
  `already_clear`; null source is allowed only when reliable rewriting is blocked.
- `source_quality`: extraction, paragraph-boundary and missing-context flags.
- `semantic_ledger`: local E/C/R IDs; original propositions, operators, quantities,
  actors and supported relations; original and rewrite quote evidence; alignment
  status and any unsupported claims introduced by the rewrite.
- `source_schema`, `target_schema`: identical structures. Ordered sentence entries
  cover each text; three observation arrays use the existing paragraph/sentence/
  language canonical IDs with exact evidence, claim/sentence/relation references,
  ambiguity and coverage gaps. Both schemas may be null for unusable material.
- `transformations`: finite descriptive operations from rewrite to original,
  bound to ledger claims and local source/target sentences. No instruction to
  increase formality or complexity, nor permission to change semantic relations.
- `taxonomy_version`, `taxonomy_sha256`: copy constants from the attached catalogue.
- `needs_review`, `review_notes`, `provenance`: visible model/settings/date only;
  paired context is true and independent verification is false.

Every quote must occur verbatim in the corresponding supplied/generated text.
`occurrence` is one-based when the same quote occurs repeatedly. Entity/claim/
relation IDs must be unique; cross-references must resolve. Sentence IDs are local
to each side; equal sentence numbers do not imply alignment. Match claims through
explicit ledger bindings. A positive self-check is not proof of equivalence.

Do not include Markdown fences, invented canonical IDs, detector scores, author
intent guesses, credentials or extra unrequested fields. Missing or uncertain
material must be reported explicitly; no padded placeholder claims.
