# Paired neutral-rewrite and rhetorical annotation task v1

Use only the attached `paragraphs.jsonl`, `annotation_output.schema.json` and
`taxonomy_catalog.compact.json`. Read all three attachments. If any is missing,
report it and stop. Return the requested downloadable UTF-8 JSONL file. The batch
header supplies the exact response filename and batch ID.

We are preparing research data for meaning-preserving academic prose rewriting.
Each supplied text is an original passage from an economics/finance paper
published before 2020. The paired training direction would be:
`plain_english -> original_paragraph`. These are academic prose examples from
different authors, not an ideal or universal human style. Do not infer an author,
consult the Internet, imitate a famous author, or use detector scores. Do not
optimize text to hide AI authorship.

For EACH input paragraph:

1. Copy `paragraph_id` and the supplied `text` EXACTLY into `original_paragraph`.
   Do not correct spelling, punctuation, names, numbers, typography or OCR. Do not
   reconstruct missing words or merge/split unrelated paragraphs. If the input is
   incomplete, corrupted, contains a heading/footnote or depends on unavailable
   context, identify that in `source_quality`. Flag it rather than guessing.

2. Extract a compact semantic ledger from that original: entities/actors and
   roles, substantive propositions, conditions, quantities and their scope,
   negation, modality, quantifiers, comparisons and attribution. Distinguish
   findings, hypotheses, expectations, definitions and assertions. Do not turn a
   conditional antecedent into an observed fact. Record causal direction only
   where supported by the text; association and explanation are not automatically
   causation. Use local IDs E1, E2, C1, C2 and R1, R2. Every substantive claim
   needs an exact original evidence quote and one-based occurrence.

3. Produce `plain_english`: a clear, neutral version for an economically literate
   reader unfamiliar with this particular paper. Simplify difficult constructions
   where useful. Preserve all substantive claims and scope, actor roles,
   conditions, uncertainty, frequency, names, citations, quantities and technical
   distinctions. Preserve both the direction and strength of causal claims.
   Do not summarize, add mechanisms/examples, update historical terminology, or
   make the source more formal. Synonym substitution alone is not a useful
   simplification. Do not target edit distance or force wording changes. If the
   original is already clear, keeping it unchanged with `already_clear` is valid.
   If reliable rewriting is blocked, use null with `pending_review` and explain.

4. Check the rewrite against the ledger. For each C-ID supply exact evidence
   from the rewrite, or an empty array if absent, and an alignment status. List
   any newly introduced unsupported claims in `unmatched_source_claims`. Explicitly
   flag changes in many/some, most/more, helps/requires, possible/established,
   participates/decides, associated/causes, and other logical force. Repair your
   proposed rewrite when a safe repair is evident; otherwise retain uncertainty.
   A self-check is not independent verification.

5. Annotate BOTH texts AS WRITTEN with the SAME three-level specification:
   - paragraph structure: moves, their order, relationships and qualifications;
   - sentence structure: clause organization, sentence grouping, subjects and voice;
   - language features: terminology, register, nominalization, connectives, hedging
     and other relevant catalogued features.
   Use only canonical IDs from the attached compact view of the existing
   provisional taxonomy. Its IDs and definitions were retained from the project;
   the vocabulary is still proposed, not empirically validated for these authors.
   Choose by definition and the correct level/dimension, not by a label's name.
   Use `unresolved` or `not_covered` with null canonical_id when needed. Do not
   invent IDs, call every formal feature desirable, or force all 132 labels to
   appear. Annotate only relevant supported observations. Use `evidence` quotes
   exact to the corresponding text and bind observations to C-IDs, sentence IDs,
   and existing R-IDs when applicable. Repeated words are not separate claims.
   Cover each text in ordered sentence entries; `exact_text` must reproduce the
   whole sentence, not a summary. Sentence IDs are local to each text. A changed
   sentence count alone does not establish changed organization or better style.
   Do not change content to make it fit the taxonomy. Protected semantic-sensitive
   categories describe meaning that must be preserved, not freely edited controls.

6. Describe the source-to-original structural differences in `transformations`.
   This direction matters: neutral AI rewrite -> original author paragraph. Use
   only the finite operation names allowed by the output schema. Bind each
   operation to existing claims and source/target sentence IDs. Multiple operations
   may be empty or KEEP when there is no supported difference. An operation does
   not authorize unsupported causal relations, added conclusions or omitted claims.
   Flag uncertainty rather than claiming that the operation is executable or that
   the pair is meaning-equivalent by definition.

The ledgers and paired annotations are produced jointly in one context. Do not
claim independent source/target annotation, verified author intent, verified
transcription or semantic proof. Reader intentions need not be inferred. A later
experiment must derive its constraints from the new source alone and must exclude
reference wording and reference evidence from generator prompts. Target structures
can serve only as explicitly marked oracle diagnostics.

Return exactly one JSON object per input paragraph, in input order, matching the
attached schema. Be concise but do not omit claims or qualifications. Do not write
Markdown fences or commentary inside JSONL. Do not add extra keys. Use empty arrays
for absent operators and null for genuinely unavailable metadata. Never fabricate
an exact model version or runtime setting. In `provenance`, record only the visible
model label and thinking setting you actually know; otherwise null.

Deliver a downloadable response file and a separate short `coverage_XX.md` report:
completed/missing IDs, source-quality flags, rewrite-status counts, unresolved
alignments, uncovered taxonomy dimensions and limitations. If output capacity is
insufficient, deliver completed rows and list missing IDs. Do not truncate JSON or
claim complete coverage for unfinished work. Do not require human review; retain
flags for the next automated audit.
