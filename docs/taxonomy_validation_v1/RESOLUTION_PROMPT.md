# Resolve existing ambiguous label occurrences using full context

Use the attached `cases.jsonl`, `contexts.jsonl`, `candidate_categories.json` and
`resolution_output.schema.json`. Read all four files before starting. If a file
is unavailable, stop and report its filename rather than designing a substitute.

Each case is ONE existing ambiguous annotation occurrence. Its context provides
the full paragraph, annotated claims, qualification scopes, relations, sentence
and clause units, and uncertainty. These annotations are interpretations, not
verified semantic facts. The candidate categories are from a proposed vocabulary.

## Decide only for the supplied occurrence

- Read the whole paragraph, its governed spans and neighboring clauses. Inspect
  the case's exact JSON path and target node. An isolated connective, participle
  or quantifier is not sufficient evidence for a broad rhetorical function.
- Use category definitions, inclusion/exclusion rules and level/dimension.
  Do not prefer a presentation control over a protected or descriptive category.
- Preserve distinctions among addition, sequence, contrast, inference, cause,
  reason, result and condition; contrast versus caveat; overt subject versus
  fronted frame; active versus copular versus passive/resultative predication.
- Preserve entities and roles, quantities, conditions, uncertainty, negation,
  quantifier scope, attribution, and direction/explicitness of semantic relations.
  Do not infer causation from punctuation or adjacency.
- Classify what is actually expressed. Do not rewrite the paragraph, repair its
  claims, invent relation IDs, or treat a category as permission to execute a
  stylistic transformation. Flag errors in the supplied annotation separately.

Use one status:

1. `resolved_single`: exactly one supplied candidate fits the occurrence.
2. `resolved_components`: the broad annotation demonstrably bundles two or more
   distinct supported components. Give separate evidence and bindings for each.
   This is not permission to assign all alternatives indiscriminately. Explain
   whether components occupy different spans or independently coexisting axes.
3. `unresolved`: the candidates cannot be distinguished reliably. Keep assignments
   empty and retain plausible candidate IDs with the missing information.
4. `taxonomy_gap`: none of the candidates adequately describes the occurrence.
   Keep assignments empty and describe the missing distinction; do not invent
   canonical IDs or edit the taxonomy in this task.

Each assignment needs exact evidence from this context's `text`, one-based quote
occurrence, and applicable existing claim/relation refs. Extend a repeated quote
to make its anchor unambiguous when possible. If a required binding is uncertain,
retain that uncertainty; do not manufacture a claim to justify a category.
The original clause evidence sometimes lacked an occurrence index. Do not copy
null into a newly claimed resolved anchor or invent its position.

## Output

Return downloadable `resolutions.jsonl`, one schema-valid object per supplied
`case_id`, plus `resolution_report.md` with supplied/completed/unresolved/missing
IDs and limitations. Copy version/hash fields from `candidate_categories.json`;
do not calculate them or substitute another taxonomy version.

State a substantive rationale for each decision and record annotation/binding
issues separately. Your decisions are proposals for automated checking, not
verified semantic judgments. Do not request mandatory human review. If output
limits prevent completion, return finished rows and list missing IDs explicitly.
Do not generalize a decision to other occurrences or to an entire inventory label.
