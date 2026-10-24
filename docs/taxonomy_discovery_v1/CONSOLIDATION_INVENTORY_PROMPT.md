# Consolidate provisional rhetorical categories into a shared vocabulary

Use the attached `label_inventory.jsonl` and `discovery_summary.json`.
This is taxonomy discovery from 28 TRAINING pairs. No development-validation or
reserved-test examples are available or needed. Do not generate new paragraphs,
revise claims, train models or infer verified author intentions.

## What these files mean

Each inventory row identifies a distinct LEVEL + DIMENSION + LABEL + DEFINITION
combination. The same label with different definitions can appear in several
rows. `inventory_id` is the stable key, not the label name alone. A row aggregates
repeated occurrences and includes representative evidence and provenance counts.
It does NOT represent an independently sampled paragraph.

The two contexts are original observed text and proposed neutral text. Neutral
proposals were annotated with access to the original/reference pair; they are not
blind independent annotations. They retained or revised the supplied plain-English
references. Their semantic faithfulness has not been independently verified.
Original uncertainty flags are retained and must not be converted into certainty.

Evidence examples are excerpts, not full context. `occurrence: null` for clause
text means the original format supplied no occurrence index; do not invent one.
Do not resolve a context-dependent ambiguity from a short quote alone.

## Your task

1. Propose finite, interpretable canonical categories for paragraph organization,
   sentence/clause organization and language features. Define inclusion and
   exclusion rules, applicability and evidence requirements.
2. Merge equivalent labels and definitions; split categories that conflate
   distinct dimensions. Do not merge categories across levels merely because
   their names match. Related concepts can remain distinct with related-item links.
3. Preserve distinctions involving condition/consequence, scope, qualification,
   attribution, modality, negation and causal direction. Semantic claims are
   protected content, not vocabulary items to combine or simplify away.
4. Classify each canonical item as descriptive, a candidate presentation control,
   or semantic-sensitive/protected. Candidate controls require source-claim
   bindings, preconditions and prohibited effects. They cannot license new causal
   claims, citations, certainty, mechanisms or technical content.
5. Keep rare coherent distinctions provisional when evidence is sparse. Do not
   discard them solely for low frequency, or invent a catch-all category to hide
   an ambiguity. Identify where additional TRAINING examples are needed.
6. Map EVERY inventory ID exactly once. Identical definitions can share a canonical
   item. If one inventory row spans several meanings, mark it
   `split_requires_context` with candidate canonical IDs and a disambiguation rule.
   If unsupported, use `unresolved` and explain why. Do not pretend all occurrences
   can be assigned to one item when their context has not been examined.

## Output contract

Return downloadable `taxonomy_v0_1_proposal.json` with these fields:

- `taxonomy_version`: `rhetorical_discovery_v0.1_proposal`
- `status`: `proposed_not_frozen`
- `source_files`: the two supplied filenames
- `canonical_items`: an array; each item contains `canonical_id`, `level`,
  `dimension`, `name`, `definition`, `inclusion_rules`, `exclusion_rules`,
  `allowed_values`, `applicability`, `evidence_requirements`, `control_status`,
  `preconditions`, `prohibited_effects`, `supporting_inventory_ids`,
  `related_item_ids`, and `provisional_reason` (null when not needed).
  Use `paragraph`, `sentence`, or `language` for level; use
  `descriptive`, `candidate_presentation_control`, or `semantic_sensitive_protected`
  for control_status. Use arrays for rule/value/reference fields. Use stable
  canonical IDs, for example P001, S001 and L001. Do not invent supporting IDs.
- `inventory_mapping`: one entry for EACH input inventory ID, containing
  `inventory_id`, `canonical_ids`, `status`, `reason`, and `disambiguation_rule`.
  Status is `mapped`, `split_requires_context`, or `unresolved`. The rule can be
  null for unambiguous mappings. Unresolved mappings may have an empty ID array.
- `merge_and_split_decisions`: explanations with affected inventory IDs.
- `unresolved_items`: remaining ambiguities and information needed.
- `consistency_checks`: input/mapped/split/unresolved counts, duplicate/missing
  inventory IDs, unknown canonical references, and limitations of support counts.
- `limitations`: what this vocabulary cannot yet express or establish.

Mapping by inventory ID is intentional: local code retains the full occurrence
index and can expand unambiguous mappings without asking you to repeat thousands
of rows. Do not fabricate annotation-level mappings. Context-dependent splits
remain unresolved until their occurrences are examined.

Separately return `consolidation_report.md` summarizing category counts,
merges/splits, provenance imbalance and limitations. Do not reproduce full
copyrighted paragraphs. This is a proposed standard to check on fresh training
examples and through independent reannotation, not a frozen or validated taxonomy.
