# Targeted meaning-preservation QA recovery v1

Use GPT-6 with High thinking effort if available. Read the attached input JSONL
and closed output schema. Save `responses_01.jsonl` for `inputs_01.jsonl`, and so on.
Return one object per case, preserving case_id and target_sha256. No Markdown
fences inside JSONL. Record actual visible model/effort where known, otherwise null.

We need faithful plain-English input -> original-author target pairs for direct
SFT. Review the existing pair before proposing another rewrite. This task does
not annotate the 132-category taxonomy or verify whether the author's scientific
conclusions are true. Previous flags are hypotheses to assess, not ground truth.

Assess two questions separately:

1. Is the supplied target transcribed and delimited well enough for this task?
2. Does the plain-English input preserve its substantive meaning and qualifiers?

Preserve actors, supporting claims, quantities, comparisons, conditions, negation,
uncertainty, attribution and causal direction. Check both directions for added and
lost claims. Do not strengthen a finding, turn association into causation, supply
a missing mechanism, or import another paragraph's claims into this one.

Do NOT reject a pair solely because:

- a technical term lacks a textbook definition, if the term and its role are
  preserved without an invented explanation;
- a citation, footnote or table is not supplied, if both texts faithfully preserve
  the paragraph's explicit assertions and do not require its contents to recover
  a missing actor, condition, quantity or relation;
- you cannot reproduce an empirical estimate or independently verify a study;
- surrounding prose contains a roadmap or the author promises later explanation.

DO retain a hold when an unresolved referent or missing context prevents reliable
meaning alignment, symbols or quantities are corrupted, paragraph boundaries are
uncertain, or the rewrite adds or omits content. Explain the exact missing item.
Distinguish these real blockers from scientific fact checking that is outside
this task. An undefined actor is different from a technical term retained intact.

Read-only context can help assess a referent. It is not extra training target
text. Do not combine unrelated paragraphs, insert context-only claims into the
rewrite, or guess damaged symbols. Raw context is not independently verified.
The supplied source_check may describe a limited prior PDF check; do not claim
you performed that check yourself. Source repairs already supplied as target_text
are versioned separately; never alter target_text or recompute its hash here.

If the existing plain English is faithful, retain it exactly. If a minimal safe
repair is justified, return repaired plain English and explain every substantive
change. If the existing input is null but the supplied corrected target now
supports a reliable rewrite, propose a neutral input, use repaired_input_proposed
and explicitly note that the input was previously absent. Do not rewrite just
for variety, impose an AI-like tone, or optimize for
a detector. If the target itself needs correction, return needs_target_source_repair;
do not silently fix it. Return null when a reliable input cannot be supplied.

For each alignment issue provide short exact evidence from the corresponding
texts. `input_evidence` may be null for an omitted input claim. If proposing a
repair, issue evidence describes the existing input, while plain_english contains
the proposed repair. Clearly distinguish uncertain interpretation from a confirmed
material change. Empty issue arrays mean no issue identified, not proof of truth.

Use faithful_as_supplied only when the input is unchanged and meaning appears
preserved. Use repaired_input_proposed only for a justified input repair. A
direct_pair_candidate may be true only in these two cases, with no unresolved
material/minor issue or missing information and needs_additional_check false.
These are provisional automated judgments, not independent semantic proof or
training authorization. Structural-label problems do not by themselves invalidate
a faithful text pair; taxonomy_assessment must remain not_performed here.

Provide a separate short coverage/limitations report. Retain uncertain/unusable
cases and list missing IDs if incomplete. Do not regenerate earlier responses in
place, require a human reviewer, or claim full coverage for unfinished work.
