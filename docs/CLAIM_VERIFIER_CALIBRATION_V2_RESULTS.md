# Claim-verifier calibration v2: Kaggle version 1

Reviewed `oliverlds/verifier_calibration`, version 1. All 120 synthetic cases
completed using the pinned public NLI model on one Tesla T4. There was no Gemma
generation, training, private-dataset access or reserved-test use.

## Held-out results

| Measurement | Original global/sentence NLI | Claim-aware verifier |
|---|---:|---:|
| Changed outputs detected | 14/16 (87.5%) | 14/16 (87.5%) |
| Faithful outputs falsely flagged | 1/32 (3.1%) | 1/32 (3.1%) |
| Nontrivial faithful paraphrases accepted | 15/16 (93.8%) | 15/16 (93.8%) |
| Changed outputs falsely accepted | 2/16 (12.5%) | 2/16 (12.5%) |
| Abstentions | 0/48 | 0/48 |

The frozen gate passes detection, false alarms and paraphrase acceptance, but
fails its zero-changed-output-acceptance requirement. The gate remains failed;
neither thresholds nor labels were changed after seeing results.

Across all 120 cases, the new verifier produces **exactly the same decisions**
as the original global/sentence NLI baseline. Local alignment and role guards
provide evidence traces and explicit safeguards, but show no incremental
decision improvement on this set. These data favor retaining the simpler
baseline unless additional controls demonstrate a benefit.

On the 72 previously exposed development cases, both methods detect 24/24
changed outputs, flag 1/48 faithful outputs and accept 23/24 nontrivial faithful
paraphrases. This supports using a semantic model over the earlier CPU-only
surface rules, but development results are not fresh validation. The old v1
held-out scenarios are development-only here. Comparing v1's 5/12 rule detections
directly with v2's 14/16 held-out detections would mix methods and different cases.

## Two false acceptances

All examples below are authored synthetic text, not paper excerpts.

**Strict endpoint:**

- Source: admission is permitted only when the score is **at least 10**.
- Changed output: admission is permitted only when the score is **higher than 10**.

The change removes equality at the threshold. Minimum bidirectional entailment
is approximately 0.974, despite the changed condition. The bound-direction guard
handles upper/lower reversals, not strict versus inclusive endpoints.

**Association strengthened to causation:**

- Source: advertising spending **is associated with** sales growth.
- Changed output: advertising spending **causes** sales growth.

Minimum bidirectional entailment is approximately 0.805, just above the frozen
0.8 threshold. Association alone does not supply the causal claim. Raising the
threshold after observing this case would be post-hoc tuning and would not fix
the high-confidence endpoint failure.

The faithful paraphrase falsely flagged is **most shipments are delayed** versus
**more than half of shipments are delayed**. Minimum entailment is approximately
0.606, driven by one direction; contradiction remains low. Thus low entailment
does not necessarily identify a meaning error.

## Verification and runtime

All protocol, fixture, exported source and manifest hashes match the frozen
local files. Every verdict was recomputed exactly from the saved NLI probabilities,
without weights or new model calls. Split/category metrics, baseline comparisons
and the readiness decision also reproduce exactly. Run status confirms 120/120
complete; logs contain neither a Python traceback nor CUDA out-of-memory error.

The assessment loop reports 8.44 seconds, excluding model download/loading,
notebook installation and Kaggle queue time. There are 282 cached prediction
pairs; peak allocated memory is approximately 0.71 GiB. Runtime uses FP32,
a 512-token limit and model revision
`6f5cf0a2b59cabb106aca4c287eed12e357e90eb`. Context is not truncated.

Private downloads and verification records are preserved under
`research_runs/claim_verifier_calibration_v2/kaggle_version_1/` and
`version_1_review/`. Aggregate machine-readable results are in
[claim_verifier_calibration_v2_results.json](claim_verifier_calibration_v2_results.json).
The preparation protocol and all previous experiments remain unchanged.

## Decision

This is substantial progress from surface-only warnings toward useful semantic
screening, but it does not certify claim preservation. The sample is small and
authored with knowledge of the approach; it is not blinded, independently
annotated or representative of long economics paragraphs.

Keep the 21-example style comparison and further fine-tuning blocked under the
current protocol. The next focused change should address inclusive/strict
endpoints and association-to-causation strengthening, with scoped evidence and
explicit abstention outside supported cases. Use these exposed cases as
regressions, freeze the revision, and evaluate on fresh controls rather than
claiming success by repairing this held-out set. Retain the NLI baseline in
comparisons, since the added alignment machinery has not earned a quality claim.

No new GPU job was submitted during this review.
