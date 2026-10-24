# Optional B and the first consolidation response

All 28 optional B annotations match the expected existing-reference paragraph IDs
and pass the output schema. Evidence, sentence coverage, claim bindings and local
references pass after an unambiguous namespace normalization: clause IDs were
sentence-local in the response, so a derived copy assigns paragraph-wide IDs and
updates their dependencies. There are 139 changed ID/reference fields across
28 records. No text, claims, labels, evidence or original responses were edited.
Every changed field is recorded in the private normalization log.

## Consolidation response is a design scaffold

The received `taxonomy_v0_1_proposal.json` explicitly identifies itself as
`schema_guided_design_scaffold_NOT_discovery_consolidation`. It reports that the
completed annotation files and neutral proposals were unavailable. Its 38
categories have no observed support, and both occurrence and inventory mappings
are empty. Preserve it as a design artifact; do not adopt it as the empirical
standard or infer that the earlier 154 label definitions have been consolidated.

The file appears to describe access to raw paragraph inputs and format guidance.
The response itself establishes that it did not have the annotation evidence
needed for the requested consolidation; this does not establish why attachments
were missing or unavailable.

## Replacement input package

`research_runs/taxonomy_discovery_v1/response_audit_v2/consolidation_upload/`
contains the corrected instruction and actual observed-label evidence:

- `CONSOLIDATION_PROMPT.md`
- `label_inventory.jsonl`
- `discovery_summary.json`

There are **200 distinct level/dimension/name/definition combinations**, accounting
for **2,508 label occurrences** from 84 annotations of the same 28 training pairs:
28 original, 28 proposed-neutral, and 28 existing-reference diagnostic annotations.
B is explicitly diagnostic; inadequate references do not become positive neutral
examples. Diagnostic-only labels are identified. Inventory IDs are version-local
and must not be joined to IDs from the earlier package.

Thirteen retained proposals have text identical to their existing references,
but their complete annotations differ even after excluding paragraph IDs and
provenance. Different annotation context, claim segmentation, uncertainty notes
or labels may explain this; it is not evidence of stylistic transformation in
unchanged text. Independent reannotation and a common vocabulary remain necessary.

Use a new ChatGPT conversation. Paste the new prompt and upload only the inventory
and discovery summary. The prompt requires confirming the inventory row count,
mapping every inventory ID, and stopping rather than inventing a replacement if
an attachment is unavailable. Request `taxonomy_v0_1_proposal_v2.json` and
`consolidation_report_v2.md`; save them alongside the previous response without
overwriting it.

All original response hashes are preserved in
`response_audit_v2/manifest.json`; normalized B, the normalization log, full
occurrence index and audit summary remain private and Git-ignored. No data repairs
were accepted for training, no model calls were made, and no held-out examples
were used. These checks do not establish semantic faithfulness or label validity.
