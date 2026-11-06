# Stage 2 — fresh prose from content and purpose briefs, version 3

For every supplied brief, compose one new neutral economics paragraph.
You receive no original paragraph. Do not infer or reconstruct an author's wording.

Use the task and content to decide what the paragraph needs to do. If explicit
purpose and reader-takeaway fields are provided, use them as additional guidance:
define, explain, compare, report, motivate, qualify, or perform the other specified
job. Present an intelligible account, not a sequence of unrelated facts. Use the
point relationships, when provided, to preserve important comparisons, causal direction, chronology,
conditions and evidence. They constrain content, not sentence order or wording.

Choose fresh subjects and natural sentence constructions. Combine, separate and
order points as useful; do not mechanically turn one note into one sentence.
Use clear ordinary prose for an economically literate reader. Cover the main
account and supporting evidence without padding, headings or a discussion of the
annotation process. Keep technical terms, citation roles and numerical denominators
where needed. Minor shifts of emphasis are acceptable; do not invent a mechanism
or reverse the explanation. Do not imitate a named author or optimize for a detector.

If room for reconstruction is limited, still give your best fresh formulation and
mark `limited_rewrite_room`; do not force synonyms for a change percentage.
If an unresolved source problem prevents an account, mark `source_problem` and
explain it in `review_notes`, rather than inserting an extraction critique into
the paragraph. Do not treat normal ambiguity as a reason to abandon usable content.

Return every supplied ID in order in the requested UTF-8 JSONL file. Each object
has exactly: `id`, `target_sha256`, `plain_english`, `rewrite_status`, `review_notes`.
Copy ID/hash unchanged. `plain_english` is one complete paragraph.
`rewrite_status` is `rewritten`, `limited_rewrite_room`, or `source_problem`.
`review_notes` is an array of short notes, empty if none. No Markdown fences,
original wording, taxonomy output or model-setting claims. Proceed without asking
for prerequisite confirmation.
