# Context resolution and fresh training-only vocabulary check

This package prepares the next two checks. It does not resolve labels, accept
repairs, change the taxonomy or train a model. Everything remains provisional.

## Stage 1: resolve representative control-related ambiguities

Open `01_context_resolution/batch_01/` and repeat for the other numbered batches.
In a NEW ChatGPT conversation for each batch:

1. Paste `RESOLUTION_PROMPT.md`.
2. Upload `cases.jsonl`, `contexts.jsonl`, `candidate_categories.json` and
   `resolution_output.schema.json` from that SAME batch folder.
3. Save `resolutions.jsonl` as `resolutions_01.jsonl` etc., with the corresponding
   `resolution_report_01.md`, under
   `research_runs/taxonomy_discovery_v1/responses/task_4_context_resolution/`.

There are up to three representative occurrences for each of the 13 ambiguous
inventory groups touching candidate presentation controls: 39 cases in six
batches. Their complete paragraphs and semantic annotations are included.
Sampling favors different papers/pairs and differing evidence wording; it is not
a prevalence estimate and does not guarantee every candidate meaning is covered.
No unsampled occurrence receives a label from this exercise. All 720 original
ambiguous occurrences remain unassigned until actual decisions are checked.

Bring these results back first. They can establish whether the candidate
definitions need clarification. If the taxonomy changes, the fresh-validation
catalogue and schema must be rebuilt with the new hash before Stage 2.

## Stage 2: fresh annotation under the same candidate vocabulary

`02_fresh_validation/batch_A/` and `batch_B/` contain seven paragraphs each,
one original and one existing reference per newly selected training pair. These
seven pairs were excluded from the 28-pair discovery set. They are fresh to
taxonomy discovery, not a new model-training or held-out test split.

In a NEW conversation per batch:

1. Paste `VALIDATION_PROMPT.md`.
2. Upload that batch's `paragraphs.jsonl`, `taxonomy_categories.json` and
   `canonical_annotation_output.schema.json`.
3. Save the output as `canonical_annotations_A.jsonl` or `_B.jsonl` and its report
   under `research_runs/taxonomy_discovery_v1/responses/task_5_fresh_validation/`.

Keep the two batches and previous annotations in separate conversations. Do not
tell the annotator which paragraphs are originals or references. The filenames
and IDs mask group membership; textual clues inside paragraphs are not removed.
Existing references are annotated as written, not accepted as ideal neutral text.

Finally process `batch_C/` with the same prompt in another fresh conversation.
Its two paragraphs are masked repeats from A/B, with new IDs. Save
`canonical_annotations_C.jsonl` and its report. These support a small consistency
check; they are not two additional independent examples or proof of reliability.
Do not show the model previous responses or identify the repeats.

The catalogue includes all proposed categories, including rare/protected ones;
it omits old annotation examples and support IDs. New annotations must use
canonical IDs or explicitly report ambiguity/coverage gaps. We will assess
coverage and consistency, not force the full vocabulary to fit these examples.

## Files to keep local

Do not upload this guide, `PRIVATE_SELECTION_AND_LINKS.json`, sampling/audit
metadata or the manifest to annotation chats. Use only files inside the matching
batch folder. The private key preserves pair/repeat links and selection reasons.
Original data, responses, clause normalization logs and previous results remain
unchanged. No credentials, model calls, GPU jobs or reserved-test data are needed.
