# Stage 1 — source check and economic content brief

Use GPT-6 with High thinking effort if available. Record the actual visible
setting; do not claim knowledge of hidden sampling parameters.

You are preparing a paired academic rewriting study. Process every record in
the attached `inputs_XX.jsonl`, following `stage1_output.schema.json` exactly.
Do not generate rewritten paragraphs, rhetorical taxonomies or style plans yet.

For each supplied paragraph:

1. Assess visible source problems: incomplete meaning or boundaries, apparent
   transcription artifacts, undefined antecedents, and technical ambiguity.
   You have extracted text, not the PDF. Set `transcription_verification` to
   `not_performed`; do not certify that the text matches the original page.
   Do not silently repair the paragraph or invent missing context. Use `uncertain`
   or `unusable` when the supplied text cannot support a reliable content brief.
2. Extract every substantive claim and its qualifications into a detailed,
   neutral economic content brief. Preserve supporting claims, not just the main
   conclusion. Distinguish claims, hypotheses, expectations, findings and author
   attributions. Preserve actors and their roles; numbers, units and time periods;
   quantifiers, comparisons and scope; negation and what it negates; modality,
   uncertainty and frequency; conditions and exceptions; and causal direction.
3. List only supported relationships between claims. Distinguish causation from
   association, explanation, contrast, comparison and conditions. Do not supply
   a causal connection merely because two claims occur next to one another.
4. Check the brief against the supplied paragraph for omissions, additions,
   altered logical force and unresolved uncertainty. This is a self-check, not
   independent semantic verification.

Use short neutral propositions rather than copying full reference sentences.
Retain necessary names, numbers, citations and technical terms exactly; do not
force synonyms where this could change meaning. An `essential_terms` entry is
a short technical anchor of at most eight words, not a sentence or paragraph.
Do not use that field to smuggle reference wording into the later generator.

Use `A1`, `A2`, etc. for actors and `C1`, `C2`, etc. for claims within each record.
IDs identify entities; their numbering does not prescribe presentation order.
Relationships reference claim IDs through the dedicated fields. Use natural
actor names in semantic prose, not embedded claim-ID references. The transport
step will replace and reorder claim IDs before generation.

Use exactly the same claim fields for every claim. Empty lists mean that an
attribute is absent, not permission to omit a present qualifier. `quantities`
entries retain the value, unit and economic scope as text. `causal_direction`
entries describe only explicitly supported direction; use an empty list if none.
Some statements have no explicit actor; an empty `actor_ids` list is allowed.
`predicate` is a neutral description of the economic relation, not a stylistic
instruction. `proposition` states the complete atomic claim with its scope.

The brief must not include the original paragraph, original sentences, author
or paper metadata, paragraph outlines, sentence groupings, rhetorical labels,
recommended connectives or proposed reference-style wording. Essential names
and citations already present in the economic content must remain; those can
still reveal a source. Do not infer a named author's style from the text.

Set `brief.needs_review` to true for unresolved content uncertainty or missing
context, and list the issue in `context_gaps`. The absence of a PDF check alone
is recorded separately and does not automatically mean that the content brief
is incomplete. A `usable` source rating is provisional, not training approval.
For an unusable source, retain its response row, report the issues and set
`brief` to null. Do not replace it with another paragraph.

Return a downloadable UTF-8 JSONL file, one object per input record, preserving
`record_id` and copying `original_sha256` exactly. Do not compute a new hash or
echo the original text. Do not put Markdown fences or commentary in the JSONL.
Keep every record, including uncertain/unusable ones. Do not silently combine,
split or omit inputs. If completion is interrupted, separately report the IDs
completed and remaining; do not claim complete coverage.

Save the first response as `briefs_XX.jsonl`, using the input batch number.
Do not revise previous responses in place or regenerate until they look better.
