# Stage 2A — direct rewriting template

Template only. Do not run before Stage 1 responses have been checked and paired
generation inputs have been frozen. Use the same actual model/effort setting as
the reconstruction condition where available, recording UI limitations.

For every record in the supplied direct-rewrite input file, rewrite its original
paragraph in clear, neutral English for an economically literate reader who is
not a specialist in this paper's methods. Use one coherent paragraph. Choose
natural sentence grouping and organization where useful.

Preserve all substantive claims, actors and roles, quantities and units,
comparisons, quantifiers, conditions, exceptions, uncertainty, frequency,
negation, attributions, time scope and supported causal relationships. Preserve
technical distinctions, necessary terms, names and citations. Do not summarize
away supporting claims or add facts, mechanisms, examples or conclusions.
Do not deliberately imitate a named author or stereotypical AI writing.
Do not force wording changes where safe simplification is uncertain.

Use only the supplied paragraph. If missing context or a source problem prevents
a faithful rewrite, return `status: "blocked"`, `text: null` and a specific reason.
Otherwise return `status: "generated"`, the paragraph in `text` and `reason: ""`.

Return one JSON object per input record with exactly `record_id`,
`method: "direct_rewrite"`, `status`, `text`, `reason`. Preserve IDs. Output a
downloadable UTF-8 JSONL file without Markdown fences. Do not drop blocked cases,
repair originals silently, or retry outputs based on detector scores.
