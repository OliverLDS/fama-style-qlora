# Two source-generation methods

Agreed addition, 30 October 2026. This design is prepared, not executed.
Keep all existing direct-rewrite datasets unchanged. Do not add a deliberately
"AI-like" condition. The training direction stays synthetic source → original
human target; provenance establishes how a text was generated, while stylistic
contrast is measured separately.

## Proposed small comparison

Before large annotation batches, select 12 nonreserved training-side paragraphs
across several papers. Compare the existing direct-rewrite method with neutral
reconstruction from a content brief. Freeze the paragraph IDs and first outputs;
do not resample until a detector flags a source. New acquisitions require source
extraction/quality checks before their paragraphs can enter the comparison.
This document does not select examples, issue annotation jobs or report outcomes.

**Direct rewrite:** GPT-6 sees the human paragraph and generates a neutral
meaning-preserving rewrite, as in the current workflow.

**Neutral reconstruction:** one stage extracts a complete content brief from the
human paragraph. A generation stage receives that brief, without the paragraph
or its author identity, and writes ordinary explanatory economics prose. This
permits different organization without deliberately imitating an AI stereotype.
It is not fresh independent composition: the content still derives from the
human paragraph, and ChatGPT UI memory/retrieval is uncontrolled.

## Content-brief extraction prompt

> Extract a wording-free economic content brief from the supplied paragraph.
> Do not rewrite the paragraph yet. Give every substantive claim a stable ID.
> For each claim record actors, proposition, attribution, quantities and units,
> comparisons and quantifiers, negation, modality, conditions, exceptions and
> temporal scope. List only supported relationships between claim IDs; distinguish
> causal direction from association, justification, contrast and mere sequence.
> Preserve cited names, years and essential technical terms. Include definitions
> or antecedents only if the supplied material supports them; otherwise flag the
> missing context. Do not guess facts from outside knowledge.
>
> Use short neutral records rather than copying full sentences. Keep essential
> terminology, names and numbers exactly where necessary. Claim IDs identify
> propositions; their numbering must not prescribe presentation order. Do not
> include the original paragraph, author-style labels, rhetorical taxonomy,
> sentence groupings, paragraph outline, reference phrases or suggested wording.
> Report any uncertain extraction or missing information explicitly. Do not claim
> independent semantic verification.

The handoff format has `paragraph_id`, `brief_version`, `claims`,
`supported_relationships`, `essential_terms`, `citations`, `context_gaps` and
`needs_review`. Each claim carries the fields listed in the extraction prompt.
Relationships reference existing claim IDs and record type and direction.
Full source/target text remains in the private audit, outside the generator
handoff. Claim numbering is an identifier, not an instruction to shuffle facts.

## Reconstruction prompt

> Write one ordinary explanatory economics paragraph using only the supplied
> content brief. Choose a natural organization and sentence grouping yourself;
> claim IDs do not specify an order. Use clear neutral English for an economically
> literate reader. Preserve all claims, quantities, attributions, comparisons,
> qualifications, uncertainty, negation, conditions and supported relationships.
> Do not add mechanisms, evidence, examples or conclusions. Keep necessary
> technical terms and citations. Do not imitate a named author or deliberately
> produce stereotypical AI writing. Return only the paragraph. If a context gap
> prevents a faithful paragraph, return a structured unresolved-context status
> instead of filling the gap.

Use separate extraction and generation handoffs; only the brief is intentionally
supplied to the generator. Record actual model label, requested and observed
thinking effort, conversation arrangement and memory/retrieval limitations.
Separate chats do not certify independence. Generation has no access to reference
wording through the supplied payload, but UI-level retrieval cannot be certified.

## Verification and decision

Verify both the brief and reconstructed text against the original in a separate
automated assessment. Check for unsupported additions, omissions and changes to
economic relationships or qualifications. Preserve original annotations and
disagreements. Quarantine material errors and unresolved brief omissions; local
expression shifts are recorded separately under the project's drift policy.
Do not require a human review panel.

Compare lexical copying and longest shared spans, sentence grouping and claim
order, readability, and the existing three-level structural features extracted
under the same specification. More edits are not inherently better; more detector
flags do not establish a better training source. Retain both methods when useful,
with source-method provenance, rather than replacing earlier labels.

Keep all variants of one original paragraph, and all versions of one paper, in
the same partition. A held-out paper stays held out regardless of input method.
The tiny comparison diagnoses whether content briefs increase useful stylistic
contrast while retaining economic information; it cannot establish broad training
effectiveness. Detector calibration stays a sparse secondary study.
