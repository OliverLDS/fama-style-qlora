# Task: propose a standardized rhetorical vocabulary from discovery annotations

Use the returned paragraph annotations and, when present, `neutral_proposals.jsonl`.
All discovery texts come from training papers. Do not request development or
reserved-test examples. This task standardizes annotation CATEGORIES; it does not
merge semantic claims, rewrite paragraphs, train a mapper, or establish author intent.

Use observed-original annotations plus retained/revised proposed-neutral
annotations as the main discovery evidence. Keep context provenance visible:
`single_paragraph` and `paired_neutral_design` are different annotation processes.
Original reference annotations, if also supplied, are diagnostics; do not count
them as independent extra paragraphs or treat rejected references as good neutral
training examples. Exclude unresolved proposals from positive mapping examples.

1. Inventory every provisional label by LEVEL, DIMENSION, NAME and DEFINITION.
   Identical names with different meanings are different entries. Report paragraph
   support and provenance. Raw annotation count is not independent support.
2. Propose canonical items for paragraph, sentence and language levels. For each,
   supply a stable ID, definition, inclusion/exclusion rules, allowed values when
   relevant, applicability, evidence requirements, and links to related items.
3. Merge only genuinely equivalent concepts. Separate items that combine different
   dimensions. Preserve distinctions in scope, qualification placement, attribution,
   causal direction, and condition versus consequence. Never merge those semantic
   distinctions merely because two labels sound similar.
4. Preserve rare but coherent concepts as provisional when warranted. Do not force
   a miscellaneous bucket or erase uncertainty to reduce vocabulary size. Mark
   proposed distinctions not adequately supported by the supplied examples.
5. Provide a COMPLETE mapping from each input label occurrence to a canonical ID,
   split rule, deferred item, or explicit unresolved status. Identify occurrences by
   paragraph/proposal ID and JSON path. Retain the original label and definition.
   Do not silently omit, relabel, or modify a paragraph's claims or evidence.
6. Classify each canonical item as descriptive, a candidate presentation control,
   or semantic-sensitive/protected. Candidate controls need preconditions, source
   claim bindings and prohibited effects. They must not authorize unsupported
   causality, certainty, citations, technical claims or entity-role changes.
7. Explain what the resulting vocabulary cannot yet express and which distinctions
   need fresh TRAINING examples. Do not claim it is validated, exhaustive, or frozen.

Return downloadable `taxonomy_v0_1_proposal.json` with these top-level fields:
`taxonomy_version` ("rhetorical_discovery_v0.1_proposal"), `status`
("proposed_not_frozen"), `source_annotation_files`, `canonical_items`,
`occurrence_mapping`, `merge_and_split_decisions`, `unresolved_items`,
`consistency_checks`, and `limitations`.

Each canonical item must contain `canonical_id`, `level`, `dimension`, `name`,
`definition`, `inclusion_rules`, `exclusion_rules`, `allowed_values`,
`applicability`, `evidence_requirements`, `control_status`, `preconditions`,
`prohibited_effects`, `supporting_annotation_refs`, and `related_item_ids`.
Use empty arrays where appropriate instead of inventing support.

Each mapping entry must contain `annotation_id`, `json_path`, `original_label`,
`original_definition`, `canonical_ids`, `status`, and `reason`.
Separately provide a concise report of vocabulary size, merges/splits, unmapped
occurrences, provenance balance and limitations. Do not reproduce full copyrighted
paragraphs in this consolidation report. Preserve discovery annotations unchanged.
