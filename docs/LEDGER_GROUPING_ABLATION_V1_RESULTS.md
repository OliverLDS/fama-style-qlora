# Ledger/grouping ablation: Kaggle version 1

Reviewed `oliverlds/ledger_grouping_ablation`, version 1. All eight first-attempt
generations completed, the runtime verifier canary passed, and **all four
cue-rich control outputs exactly reproduced their earlier output hashes**.
The controlled comparison therefore reproduces the known failure rather than
relying on a different baseline output.

## Results

Each condition contains the same two already-observed development failures.
Grouping on includes the whole existing soft-plan block, not just a single
instruction to merge. The master rewrite instruction and decoding are unchanged.

| Condition | Cue-artifact warnings | Single-sentence outputs | Mean paragraph-level words per sentence | Mean style-gain proxy |
|---|---:|---:|---:|---:|
| A: sentence evidence, grouping off | 0/2 | 1/2 | 64.3 | -0.309 |
| B: sentence evidence, grouping on | 0/2 | 2/2 | 143.0 | -3.963 |
| C: cue-rich ledger, grouping off | 0/2 | 2/2 | 156.5 | -3.146 |
| D: cue-rich ledger, grouping on | 1/2 | 2/2 | 150.0 | -4.457 |

All eight formats are valid paragraphs. There are no exact or near copies under
the declared 0.90 threshold. Every condition has one verifier risk signal and
one abstention, with zero verifier-consistent outputs. A warning is not a
confirmed error; an abstention is not a success or a failure judgment.

Style gain is change in distance to the training-reference feature profile,
not a validated authorship or prose-quality score. Higher is better. A is the
cleanest diagnostic setting here, but its negative mean and unresolved semantic
screening do not establish faithful Fama/coauthor-style transfer.

## What the controlled changes show

On the longer case, the cue-rich grouped control reproduces the annotation-word
leakage seen previously. Removing only the isolated `protected_evidence` fields
eliminates those artifact-screen hits while keeping the grouping plan fixed.
This supports a prompt-format contribution to that observed artifact. It is one
reproduced case, not a general error-rate estimate or proof that all added claims
have disappeared.

Removing cues alone does not fix sentence inflation. The simple-ledger grouped
output remains a single 202-word sentence on the longer case. Without grouping,
the simple-ledger version has six sentences and approximately 35.7 words per
sentence, compared with source/reference means of 22 and 23 for that case.
The cue-rich setting without grouping instead has one 223-word sentence. Thus
both the ledger presentation and the plan block can contribute to poor realization
on this example; their combination is not necessary for long sentences.

The shorter case remains a single sentence under all four conditions, with
84–93 word/number tokens versus source/reference sentence-length means of about
17.3 and 23. The outputs differ, but no condition resolves its verifier risk
signal. At least one version also appends speculative explanatory content absent
from the source. Changing these two prompt factors is therefore insufficient to
solve all residual drift or sentence-level organization.

The simple/no-grouping setting has a positive style proxy on the longer case and
a negative value on the shorter case. Its aggregate improvement should not be
interpreted as a uniformly successful method. The result concerns a narrow fixed
plan; it does not evaluate a learned taxonomy mapping or the full structural
style-transfer approach.

## Semantic-screening limits

The longer case is NLI-consistent under all four conditions before the scoped
guards, but the v3 verifier abstains on unresolved scope in each version. The
shorter case is NLI-flagged in each version. No semantic screening improvement is
established by this ablation. The earlier synthetic calibration pass did not
establish accuracy on these longer economics paragraphs.

The artifact screen was defined from an already-seen failure. Its absence is a
useful formatting observation, not evidence that every claim was preserved.
Neither the two-case selection nor the marker definitions are independent of
the earlier results. No examples were dropped after these generations.

## Verification and runtime

All input, recursive manifest, exported source and model/control hashes match.
Saved probability requests reproduce all eight output verifier decisions, both
source/reference checks and all 168 runtime-canary verdicts exactly. Metrics and
contrasts recompute exactly. All generations are nonempty, below the output
limit and marked as real inference. Logs contain neither a Python traceback nor
CUDA out-of-memory message. Rich-control replication is verified for both cases
and both plan settings.

Runner time is 253.9 seconds (4.23 minutes), including approximately 130.0 seconds
of generation, 19.2 seconds for the canary stage and 14.0 seconds for the final
verifier stage; remaining time is model loading/overhead. Queue and dependency
installation are excluded. Gemma and NLI peak allocated memory are approximately
4.79 and 0.78 GiB, respectively, on one T4 with sequential model loading.

Private raw text, per-case scores, probability caches, logs and verification
records remain under `research_runs/ledger_grouping_ablation_v1/`. Public aggregate
metrics are in [ledger_grouping_ablation_v1_results.json](ledger_grouping_ablation_v1_results.json).
All earlier runs, source pairs, code snapshots and gate thresholds remain intact.
There was no training, reserved-test access or new GPU submission in this review.

## Decision

Use **sentence evidence without the grouping plan** as the working prompt
baseline for the next diagnostic. Retire the cue-rich ledger and optional
sentence-grouping block from that baseline; preserve them as experimental controls
and historical results. Do not describe the simplified prompt as a validated
claim-preserving solution, and do not change existing training data or adapters.

The next focused step is source-sentence-preserving editing against this cleaner
baseline: test whether limited edits can improve expression without merging
separate claims into one long sentence or appending explanations. Track copying
explicitly, since smaller edits can increase it. This would be a realization
diagnostic, not a new training run. No follow-up code or job was started here.
