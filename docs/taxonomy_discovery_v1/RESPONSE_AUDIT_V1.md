# Discovery responses: first audit

All 28 original annotations and 28 neutral proposals were received. Each chunk
matches its supplied IDs. The combined neutral-proposal file is identical to its
four chunks and was counted only once.

## Executed checks

- Both JSON Schema contracts pass.
- All 2,444 annotation evidence anchors and 56 issue-evidence anchors resolve to
  their proper input text or proposed neutral text.
- Local entity, claim, semantic-relation, move and clause references resolve;
  IDs are unique in their namespaces.
- Sentence coverage and clause-text checks pass. All claims are linked to
  sentence entries.
- Proposal IDs match the source pair bindings. Retained references are unchanged.
- All response files remain unchanged; their hashes are recorded.

ChatGPT proposed retaining 13 references and revising 15. No proposed repair has
been applied to the dataset. Original annotations contain 165 claim units; neutral
proposals contain 132. These are independently segmented units, not evidence that
all claims are aligned or that the proposals are semantically faithful.

The original annotations flag 27 paragraphs for uncertainty; neutral annotations
flag none. This is self-reported annotation uncertainty, not independently measured
accuracy or grounds for requiring human review. Preserve this provenance during
consolidation. No new model/NLI call or held-out-data evaluation was performed.

## Prepared consolidation input

There are **1,660 label occurrences**, reduced without semantic merging to **154
distinct level/dimension/name/definition combinations**. Full occurrence paths
remain in a private index; consolidation maps each inventory ID once. This avoids
asking ChatGPT to repeat thousands of occurrence mappings. Ambiguous splits must
remain unresolved until their contexts are examined.

Private audit and immutable response hashes:
`research_runs/taxonomy_discovery_v1/response_audit_v1/`.

Upload ONLY these three files from its `consolidation_upload/` subfolder:

1. `CONSOLIDATION_PROMPT.md` — paste as the instruction.
2. `label_inventory.jsonl` — upload as evidence.
3. `discovery_summary.json` — upload as provenance and audit context.

Use a new ChatGPT conversation. This prompt is adapted to the inventory format;
do not use the earlier prompt asking for full occurrence-by-occurrence mappings.
Save `taxonomy_v0_1_proposal.json` and `consolidation_report.md` under
`research_runs/taxonomy_discovery_v1/responses/task_3_consolidation/`.

The outputs remain a proposed vocabulary. Check them and independently reannotate
accepted neutral realizations before treating them as standardized mapping data.
