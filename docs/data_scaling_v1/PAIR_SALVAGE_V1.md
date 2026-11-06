# Completed pair salvage and first 200-pair cohort

Executed 3 November 2026. The previous **41/249 provisional count was not a
validity estimate**: the gate combined direct text-pair suitability with optional
taxonomy, context and review flags. That interpretation and the proposed extra
manual QA uploads are superseded by this completed reassessment.

## What was recovered

All 249 collected units were read as input/target pairs and given explicit,
versioned automated decisions:

| Parent decision | Units |
| --- | ---: |
| Retained without text changes | 158 |
| Repaired | 71 |
| Excluded from this narrative-writing release | 20 |

The **229 retained parents produce 230 examples**, because one incorrectly
combined two-paragraph selection was separated at its PDF-confirmed boundary.
The resulting split is **171 training examples and 59 development examples**.
These are usable under an automated pilot admission standard, not independently
verified human gold labels. The retention ratio is not a GPT-6 accuracy estimate.

| Group | Training examples | Development examples |
| --- | ---: | ---: |
| Fama/coauthors | 14 | 5 |
| Shiller | 58 | 19 |
| Holmström/coauthors | 50 | 20 |
| Baker–Wurgler/coauthors | 49 | 15 |

Private release: `data/author_pairs_salvage_v1/`. It contains
`training.private.jsonl`, `development.private.jsonl`, `excluded.private.jsonl`,
`adjudications.private.jsonl`, `repair_audit.private.jsonl`, hashes and a summary.
Nothing was deleted from the collected responses or earlier datasets.

## Repairs and admission decisions

Targeted PDF-image checks accompany 90 parent decisions. Repairs include
same-baseline OCR fragments in the wrong order; fused words; misplaced ordinal
suffixes; damaged signs, operators and subscripts; punctuation; headings mixed
into prose; and a quotation combined with the following author-analysis paragraph.
Neutral inputs were adjusted where needed to preserve comparative force or causal
attachment. Exact before/after text, reasons, hashes, PDF crop hashes and original
line references remain in the private audit.

**Thirteen missing neutral inputs were supplied by Codex from PDF-checked
targets.** They are labeled separately from GPT-6 output. The user's raw nulls
and generation metadata were preserved. This release does not relabel Codex
repairs as GPT-6 High-effort generations.

Context pointers, retained technical terms, citations and reported empirical
claims do not require external truth checking merely to preserve what a paragraph
says. Missing optional taxonomy fields or invalid transformation bindings do not
alone invalidate a faithful direct text pair. Original flags remain available;
they are not erased. Old structural annotations are not certified by this audit,
and changed text requires rebinding before structural training.

The 20 exclusions comprise 15 table/figure captions or notes, one paragraph
explicitly attributed to another paper's discussant, one quotation-dominated
unit and three incomplete/misbounded selections. These are principally source
selection problems; they are not counted as 20 demonstrated GPT-6 rewriting
failures. Their records and annotations remain preserved.

Some draft grammar and internal numerical/specification inconsistencies are
visible in the PDFs themselves. These are explicitly flagged and are not silently
changed into invented author wording. Fidelity assessment is separate from
copyediting the author's draft or validating its research findings.

## First balanced training cohort

Thirty-six older Fama pairs and four older Baker–Wurgler pairs were reassessed
using the same text-fidelity distinction. Thirteen of those inputs were repaired;
27 were retained. One additional legacy selection repeated a finding already
retained from the same paper and was excluded as a near duplicate. The other
older pairs are untouched and have not been automatically requalified.

The combined pool has **210 approved unique training parents** after restricting
cohort accounting to one child per original annotation parent. The first cohort
contains **200 pairs across 48 papers**, exactly 50 per group: 160 from the
recovered batch collection and 40 from reassessed legacy releases. Ten approved
unique parents remain outside this first cohort, as does the extra paragraph
child. They have not been discarded.

Frozen file:

`data/data_scaling_v1/stage_0200_salvage_v1/training.private.jsonl`

Its SHA-256 is:

`fc50963ab2fcd0c37e434774f515cddcddf192732fd8256aceec05bfbd738f9a`

Selection uses the declared `data_scaling_v1` fixed-hash, paper-interleaved rule,
with one deterministic child per parent. No model outputs or detector scores were
used to choose the cohort. Development examples remain separate. Reserved test
records were not opened or repurposed.

## Verification and reproducibility

The assembler verified all 113 frozen parent response/audit artifact hashes and
all 54 selected source PDF hashes. It rejects unresolved decisions, missing prose,
private-use characters in approved pairs, duplicate targets and paper-role
conflicts. The cohort builder checks text hashes, admission status and training
roles before selecting 50 per group.

Independent CPU rebuilds reproduce the release data/audits and the 200-pair
cohort byte-for-byte. There are no shared paper IDs, exact targets or target
5-word-shingle containment pairs above 50% between/within the 200 training and 59
development examples. This lexical screen is not a proof against all possible
semantic overlap. None of the 200 inputs is exactly its target. All 708 existing
CPU tests pass; these checks do not establish semantic correctness by themselves.

Public assembly code:

- `scripts/data/build_pair_salvage.py`
- `scripts/data/freeze_salvaged_stage_200.py`

Private adjudication inputs and source evidence are required to rebuild the data;
copyrighted paragraphs, PDFs and private annotations are not distributed.
Public aggregate summaries: [salvage](salvage_summary.json) and
[stage 200](stage_0200_summary.json).

## Limits and next action

Codex performed the semantic adjudication and repairs. There is no independent
human review or independent semantic verifier behind the word "approved."
Targeted crops do not amount to full visual transcription verification of every
paragraph. Prior exposure and same-reviewer limitations still apply.

The concrete next action is to prepare the fixed development panel, matching
prompt format, token-length checks and capped training configuration for this
200-pair stage. No further manual annotation uploads are needed for that step.
No GPU jobs, detector scans or additional external model calls were submitted.
Larger cohorts and improved writing/detector outcomes are not claimed.
