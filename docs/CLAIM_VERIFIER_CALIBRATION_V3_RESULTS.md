# Verifier calibration v3: Kaggle version 1 results

Reviewed `oliverlds/verifier_calibration_v3`, version 1. All 168 synthetic cases
completed. **The predeclared synthetic readiness gate passed.** No Gemma
generation, training, private-corpus access or reserved-test use occurred.

## Fresh held-out comparison

Both methods below were assessed on the same 48 fresh v3 cases. The v2 column
is not its earlier, different held-out set.

| Measurement | v2 verifier on v3 cases | v3 verifier |
|---|---:|---:|
| Changed outputs detected | 13/16 (81.3%) | 16/16 (100%) |
| Changed outputs falsely accepted | 3/16 (18.8%) | 0/16 |
| Faithful outputs falsely flagged | 2/32 (6.3%) | 2/32 (6.3%) |
| Nontrivial faithful paraphrases accepted | 14/16 (87.5%) | 14/16 (87.5%) |
| Abstentions | 0/48 | 0/48 |

All four frozen criteria pass: detection at least 80%, faithful false alarms at
most 10%, nontrivial paraphrase acceptance at least 50%, and no changed output
accepted on this set. No thresholds, labels or fixtures were altered in this
review.

The original global/sentence NLI baseline and the v2 claim verifier have identical
decisions on this set. The new guards override three of their acceptances, all
changed examples, with no additional false alarms or abstentions:

- An inclusive 8.25 percent lower bound changed to a strict lower bound.
- An inclusive rating bound changed to a strict bound while a separate charge
  bound stayed unchanged; the guard aligns the correct variable.
- Association between delivery speed and customer satisfaction changed into
  a causal assertion.

The remaining changed cases were already flagged by the v2 baseline. Thus the
fresh comparison supplies incremental evidence for the two targeted guard
families; it does not establish benefits from the extra claim-alignment machinery
independently of those guards.

## Remaining false alarms

Two faithful paraphrases are flagged by both v2 and v3:

- An upper bound expressed as **at most 2 percent** versus **two percent or lower**
  inside the same order-acceptance condition.
- **Most submissions** versus **more than half of submissions**.

These are NLI false alarms on the authored labels; the new guards add neither.
Passing the gate is not perfect paraphrase recognition and does not make every
risk signal a confirmed claim error.

All 120 previously exposed cases were development-only. In that split, v3 detects
40/40 changed cases, flags 2/80 faithful cases and accepts 38/40 nontrivial faithful
paraphrases. Those results agree with the earlier cached development replay and
are not fresh validation.

## Verification and execution

The frozen protocol, fixture and source hashes match, as do all downloaded
manifest file hashes. Exported runtime code matches local code. Replaying the
saved probability requests reproduces every one of the 168 verdicts exactly,
including scoped guard evidence. Split/category metrics, both baselines, paired
overrides, abstention counts and the readiness decision also recompute exactly.
The run is complete and logs contain neither a Python traceback nor CUDA
out-of-memory error.

The assessment loop took 12.93 seconds on one Tesla T4, excluding model
download/loading, dependency installation and queue time. Runtime used the same
pinned DeBERTa revision and FP32 settings, 462 cached prediction pairs and about
0.71 GiB peak allocated GPU memory. No text was truncated.

Downloads and verification evidence remain under
`research_runs/claim_verifier_calibration_v3/kaggle_version_1/` and
`version_1_review/`. Public aggregate metrics are in
[claim_verifier_calibration_v3_results.json](claim_verifier_calibration_v3_results.json).
Earlier experiments and frozen preparation documents remain unchanged.

## Decision

Proceed to preparing the planned **21-example development inference comparison**:
the same source-only claim guidance in both arms, with optional soft rhetorical
guidance added in one arm. Let sentence count vary; use the same pinned base
Gemma, examples and decoding controls. Evaluate semantic screening, copying,
style and cost separately, retaining unknowns and false-alarm caveats.

This passes the operational synthetic gate for an exploratory follow-up, not a
claim of reliability on economics paragraphs. The held-out cases are a small
authored challenge designed around supported grammar, not blinded or externally
annotated domain evidence. Long paragraphs, new paraphrases, nested scopes and
context limits can still cause errors or abstention. Verifier consistency remains
a proxy, not semantic proof.

The existing frozen comparison package records the earlier failed checker and
remains untouched. A follow-up should create a new package recording this v3
calibration evidence rather than changing its old gate field in place. The 45
reserved test examples must remain outside development and model selection.

Do not start further QLoRA or mapper training yet. No new generation, GPU job or
automatic execution of the 42 comparison prompts was initiated during this review.
