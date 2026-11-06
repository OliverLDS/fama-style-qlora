# Stage 2 — realize an unordered content brief, version 4

Compose one fresh neutral economics paragraph per brief. No original paragraph
is supplied. The topic identifies the subject; key points are a content inventory
in alphabetical order, not a requested sentence sequence. K1, K2, ... refer to note
positions. Resolve @A# through the local anchor dictionary before writing.

Select an appropriate new opening and a coherent presentation order from the
content. Combine, split and interleave notes as useful. Do not turn each point
into one sentence or follow alphabetical indexing as an outline. Express anchor
facts naturally once where useful; do not print IDs, subject labels or annotations.

Where purpose labels and relationships are supplied, use them to understand the
communicative job and the substantive dependencies. They do not prescribe original
sentence order, connectives or wording. Without those fields, infer an appropriate
job and organization from the same topic, notes and anchors.

Use ordinary clear prose for an economically literate reader. Cover the main
account and useful supporting evidence. Keep relevant quantities, units, citation
roles, causal directions and conditions. Modest shifts in emphasis are acceptable;
do not invent a mechanism or reverse the explanation. Do not try to recover an
original author's wording, imitate a named author, or optimize for a detector.
There is no required percentage of changed words or word-count target.

Return every ID in order as UTF-8 JSONL, exactly five fields per object:
`id`, `target_sha256`, `plain_english`, `rewrite_status`, `review_notes`.
Copy ID/hash unchanged. `plain_english` must be one complete paragraph without a
heading. `rewrite_status` is `rewritten`, `limited_rewrite_room`, or `source_problem`.
`review_notes` is an array, empty if none. Give your best new formulation if there
is limited room for rewriting. Put genuine unresolved source difficulties in the
notes rather than writing an annotation critique into the paragraph.

Do not output schemas, key-point lists, IDs inside the prose, taxonomy annotations,
model-setting claims or Markdown fences. Proceed without prerequisite confirmation.
