# Stage 2B — neutral reconstruction template

Template only. Codex will create the actual content-brief-only payload after
checking Stage 1. Do not supply the original paragraphs, PDFs, private selection
key or direct-rewrite outputs to this generation task.

For every supplied content brief, write one ordinary explanatory economics
paragraph for an economically literate reader who is not a specialist in this
paper's methods. Use clear, neutral English. Choose a natural organization and
sentence grouping yourself. Opaque claim IDs and array order prescribe no order.

Use only the brief. Preserve all substantive claims, actors and roles,
quantities and units, comparisons, quantifiers, conditions, exceptions,
uncertainty, frequency, negation, attributions, time scope and supported causal
relationships. Preserve technical distinctions, necessary terms, names and
citations. Do not summarize away supporting claims or add facts, mechanisms,
examples or conclusions. Do not infer a causal link from claim ordering.
Do not deliberately imitate a named author or stereotypical AI writing.

If a missing fact, unresolved context gap or inconsistency prevents faithful
generation, return `status: "blocked"`, `text: null` and a specific reason.
Otherwise return `status: "generated"`, the paragraph in `text` and `reason: ""`.
Do not recover missing material through browsing or earlier conversations.

Return one JSON object per input record with exactly `record_id`,
`method: "neutral_reconstruction"`, `status`, `text`, `reason`. Preserve IDs.
Output a downloadable UTF-8 JSONL file without Markdown fences. Do not drop
blocked cases or retry outputs based on detector scores.

Use a separate conversation from extraction/direct rewriting as practical task
separation. Account-level memory/retrieval is uncontrolled: a separate chat is
not evidence of independent generation or guaranteed absence of reference access.
