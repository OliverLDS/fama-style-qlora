# Small fixed-unit consistency check

This repeats ONLY the same two paragraphs used in the Stage 2 repeat diagnostic.
No new paragraphs, model weights, GPU run or dataset repairs are involved.

## Run two independent conversations

For `round_1/`:

1. Start a new ChatGPT conversation and paste `ANNOTATION_PROMPT.md`.
2. Upload `paragraphs.jsonl`, `feature_checklist.json`, and `output.schema.json`
   from that SAME round folder.
3. Save the JSONL as `fixed_unit_annotations_01.jsonl` and the report as
   `coverage_report_01.md` under
   `research_runs/taxonomy_discovery_v1/responses/task_6_fixed_unit_check/`.

Repeat in another NEW conversation using `round_2/`, saving `_02` files. Do not
show prior outputs, identify repeats, or upload this guide/private linking key.
The two rounds use identical text, unit geometry and checklist, with different
anonymous paragraph/cell IDs. Keep the available model/settings the same if possible;
do not claim they are pinned when the ChatGPT interface does not expose them.

## What is held fixed

The first Stage 2 A/B annotations supply the operational sentence/subsentence
spans; masked-repeat C annotations do not select or reconcile boundaries. These
are not guaranteed true clauses. Unit-suitability judgments make that limitation
explicit. Character offsets and exact source text are checked programmatically.

The checklist samples 13 observable categories from the unchanged 132-category
proposal: paragraph illustration/contrast, local progression, grammatical linkage,
voice, sequencing, comparison and nominalization. Every applicable unit has a
mandatory cell, including explicit absent/unresolved/not-applicable states.
The input lists the exact cells; there is no need to construct the matrix manually.

This intentionally narrows measurement scope. It is not a full style annotation,
an issue-level analysis, a test of all categories, or a semantic-fidelity certificate.
Agreement dominated by unresolved/not-applicable states will not count as useful
informative agreement. No two-text check establishes broad reliability.

Bring both results back for unit/cell completeness, anchor and binding checks,
and a direct comparison of the same cells. Existing taxonomies and responses remain
unchanged. This package contains private paragraph text and remains Git-ignored.
