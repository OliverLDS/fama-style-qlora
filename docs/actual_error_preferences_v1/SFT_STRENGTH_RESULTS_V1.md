# Stronger SFT: learning responds, but development fidelity still fails the gate

Follow-up: the user has agreed to the
[matched DPO-only implementation](DPO_STRENGTH_V1.md). Its prepared status does
not change the SFT results below or imply a DPO result.

User-reported notebook: `oliverlds/sft-strength-4b`, version 1.
The fixed `1e-5` continuation completed successfully. It changes behavior much
more than `1e-6` and repairs some targeted claims, but introduces a new development
error. **The checkpoint does not qualify for detector scans or broader training.**

## Verified execution

All 22 updates, 22 finite/nonzero gradient checks, 25 processor/mask audits and
27 generations completed. The original eight control outputs and seventeen
initial target likelihoods reproduced before training. Update grouping, sibling
normalization and the tenfold linear learning-rate schedule match the frozen plan.
Both saved checkpoints have the expected 136 F32 LoRA tensors and 2,228,224
parameters; their downloaded file hashes match the run record. The final tensor
fingerprint matches the report, and final evaluation used the reloaded, frozen
epoch-2 adapter. All three emitted runtime/helper files match their frozen hashes.

Worker runtime was **587.5 seconds (9.8 minutes)** on one Tesla T4, with peak
allocated GPU memory **5.10 GiB**, within the 30-minute notebook cap. Nineteen
final outputs completed: eleven seen training cases and eight development cases,
in addition to eight startup control generations. None is empty, token-limited
or an exact input copy. Credentials were not printed or found by the local log
pattern scan. No new GPU job was submitted during review.

## Fixed epoch-2 comparison

| Measure | Original adapter | Weak SFT `1e-6` | Strong SFT `1e-5` |
| --- | ---: | ---: | ---: |
| Target NLL, seventeen training pairs | 2.24465 | 2.21284 | **1.94140** |
| Target NLL, eight development cases | 1.73675 | 1.71356 | **1.51700** |
| Training cases with material issues | 11/11 | 11/11 | **9/11** |
| Training cases within local meaning budget | 0/11 | 0/11 | **1/11** |
| Development cases with material issues | 4/8 | 4/8 | **4/8** |
| Development cases within local meaning budget | 3/8 | 3/8 | **2/8** |
| Training outputs identical to original control | 11/11 | 8/11 | **0/11** |
| Development outputs identical to original control | 8/8 | 6/8 | **0/8** |

Training cases were deliberately selected for known errors; eleven control
errors are by construction, not a general training-accuracy estimate. Target
NLL is token-weighted; it is not generation accuracy or a style metric.

The provisional masked generic-expression comparison favors stronger SFT over
the original adapter in seven training cases, ties three and disfavors one.
Against weak SFT it wins eight, ties two and loses one. On development it wins
four, ties three and loses one against either baseline. These judgments concern
clarity, organization and concision, **not distinctive Fama/coauthor style**.

Median source overlap rises from 0.36681 to 0.42534 on training, and from 0.36861
to 0.42419 on development. Median reference overlap also increases. These are
surface shifts; they do not themselves establish fidelity or learned author style.

## What changed semantically

Two selected training cases no longer have an identified material error:

- The dividend-premium paragraph restores the ordering of premium and payout
  propensity changes, the single substantial lag and the catering-incentive
  conclusion. It meets the local meaning budget.
- The predictability paragraph restores the distinction between risk-premium
  explanations and market inefficiency, rather than treating risk premiums and
  predictive power as alternatives. Two minor residual changes remain: a broader
  term-structure description and a stronger consistent-profitability criterion.
  It exceeds the one-minor budget despite resolving the main material error.

Other outputs show partial repairs, including clearer variance comparisons,
explicit predictive/true-variance distinctions and restoration of an IPO
underpricing endpoint. Remaining attribution, conditioning, scope or hot/cold IPO
errors prevent counting those paragraphs as materially repaired.

On development, the planning paragraph removes invented uncertainty about efficacy
and causal effects and restores the not-whether-planning distinction. It still
has two minor shifts under the existing rubric: the opening cross-paragraph link
is omitted and necessity is weakened to typical framing. It therefore does not
meet the one-minor budget.

Conversely, the bank-data paragraph adds an unsupported **exclusive monitoring
purpose**. The earlier checkpoint review treated unsupported exclusive-purpose
restrictions as material; the same boundary is retained here. Numerical coverage
and on-request access survive, but this new error offsets the planning repair.
Price-mechanism, risk-reflection and ownership-scope errors persist.

Consequently, the unchanged development gate fails: four material-issue outputs
exceed the maximum of three, and two locally eligible outputs fall below four.
Generic-expression wins and zero exact copying do not compensate for those
failures. There is no new detector result or established human/Fama-style benefit.

## Interpretation and next decision

The stronger learning rate changes all nineteen outputs and produces some
targeted corrections. Thus update strength mattered in this controlled setting;
the near-stasis at `1e-6` was not evidence that the model could not respond at all.
However, larger probability shifts do not reliably enforce claim preservation.
This result does not justify another automatic learning-rate increase or a full run.

The next informative comparison is a **matched DPO-only continuation at `1e-5`**,
using the same seventeen chosen/actual-rejected pairs, original starting adapter,
22 grouped updates, fixed reference and beta `0.1`. Compare the fixed final result
with this stronger SFT on the same eleven training and eight development cases.
This tests whether explicit error contrast helps preserve claims at an update
strength that demonstrably changes behavior. The old DPO result at `1e-6` does
not settle that question. Equal updates and learning rate still do not equalize
objective scale or compute.

This is a **proposal only**: no new DPO implementation, job or training approval
is implied. Do not collect detector reports for the current checkpoint. Retain
the original goal of distinctive Fama-style rewriting while separating it from
generic-expression gains in this pooled-author diagnostic.

## Artifacts and limits

Raw outputs, both adapters and log remain private in
`research_runs/sft_strength_v1/kaggle_v1/`. The review folder contains the nineteen
masked triplets, case-wise key, exact-byte-bound judgments, 57 per-output records,
summary, receipt and manifest: `research_runs/sft_strength_v1/review_v1/`.
Public aggregates: [sft_strength_results_v1.json](sft_strength_results_v1.json).
CPU reviewer: `src/fama_style/research/sft_strength_review.py`.

The reviewer knew historical controls and errors; fixed masking does not establish
independence. Material/minor boundaries and expression ranks remain provisional.
The strict meaning budget is applied unchanged, with newly introduced errors
reported alongside repairs. One seed, tiny error-selected training data and reused
development cases do not support broad superiority or impossibility claims.
No reserved-test data or detector services were accessed. The CLI output endpoint
does not enforce its parsed version argument: version 1 is user-reported, while
actual runtime, configuration and adapter content are verified by frozen hashes.
