# Anchor inference: target preference does not guarantee faithful rewriting

Completed as `oliverlds/semantic-anchor-4b`, requested version 1. All **48 first
rewrites, 96 completion scores and 48 label audits** are present. No training
updates occurred. Model revision, runtime source, jobs/config and the reviewed
expanded epoch-2 adapter match the frozen inference packet. The review made no
additional model calls and changed no datasets, targets, checkpoints or settings.

The clearest result is a distinction between two tasks: assigning higher
likelihood to a supplied faithful target and freely generating a faithful
rewrite. Strong performance on these synthetic contrasts does not guarantee
faithful generation. This does not establish that the model can recognize every
error it generates: the counterexamples target one supplied edit, while the
generation review covers the paragraph's full set of claims.

## Matched results

| Model / input | Material-issue outputs | Mean-NLL faithful-target preferences | Total-NLL faithful-target preferences |
| --- | ---: | ---: | ---: |
| Base / original | 9/12 | 12/12 | 11/12 |
| Base / variant | 10/12 | 10/12 | 10/12 |
| Expanded adapter / original | 10/12 | 12/12 | 11/12 |
| Expanded adapter / variant | 8/12 | 11/12 | 11/12 |

Mean NLL is token-normalized completion loss; total NLL is its sum over supervised
completion tokens. Lower is preferred. Ties were retained; none occurred. All
four groups have **zero exact source copies, empty outputs or length-limited
outputs**. There is no new held-out target-loss result in this check.

With original inputs, both models favor the faithful target on all twelve
mean-NLL contrasts, yet respectively nine and ten generated outputs have material
claim issues. With new inputs, the adapter has three improved and one regressed
case relative to its own original-input generations; the remaining eight keep
their issue status. This is a small, mixed input effect, not a robust gain.
Base Gemma has one regression and no improvement under new inputs.

The adapter has generic expression wins/ties/losses of **7/3/2** against base on
original inputs and **8/4/0** on variants. Within the adapter, variants beat its
original-input outputs **6/2/4** on generic expression. These partly masked
organization/concision preferences do not establish distinctive Fama style.

## Errors and exceptions

| Error family | Base original | Base variant | Adapter original | Adapter variant |
| --- | ---: | ---: | ---: | ---: |
| Scope | 3/3 | 3/3 | 3/3 | 3/3 |
| Time | 2/3 | 2/3 | 1/3 | 2/3 |
| Numbers | 1/3 | 2/3 | 3/3 | 3/3 |
| Reasoning | 3/3 | 3/3 | 3/3 | 0/3 |

The promising local observation is the adapter's three variant-input reasoning
cases. They retain the fixed-effort/risk-neutrality reason, the cautious
predictability-versus-inefficiency account and the persistence explanation for
overlap's limited benefit. Minor wording/editorial flags remain. Their
original-input counterparts introduce additional reasons, confuse genuine
predictability with risk-premium explanations or substitute predictive power
for the benefit of overlapping observations. These are training-side cases,
not three independent demonstrations of generalization.

Other failures persist or change under new wording:

- A cycle description turns firm dividend-payment propensity into consumer
  willingness to pay; another output invents a consistent causal response.
- A percentage rewrite retains every numeric value but says the larger 2000
  figures are lower than the 1978 figures. Correct numeric inventory does not
  guarantee a correct comparison.
- The variance example loses the predictive/true-variance distinction or makes
  an illustrative investor's unknown expected return into a universal assertion.
- Underpricing rewrites omit the zero-uncertainty limit, equate hot/cold issues
  with overvalued/undervalued shares or change qualified cost/benefit claims.
- A survey paragraph adds an increasing research trend or comprehensive scope
  instead of a selection of the most significant theories.

Likelihood exceptions are also retained. With variant inputs, base Gemma favors
the unsupported all-theories scope and the past-for-future realized-return
substitution. The adapter fixes the former preference but still favors the
past-return rejection. These are conditional scores on supplied text, not explicit
model explanations of which claims are valid.

Both models show a mean/total ranking disagreement on the original-input
need-not/cannot viability contrast. The faithful completion has **104 supervised
tokens**, versus **103** for the rejection. Per-token loss slightly favors the
faithful text; total loss favors the shorter rejection. Both score forms must
remain visible rather than selecting whichever gives higher apparent accuracy.

## Assessment limits and severity sensitivity

All four conditions were shuffled independently per anchor. All 48 text-bound
claim assessments and twelve style rankings were saved before unmasking or
inspecting likelihood trends. The reviewer created the variants, had seen earlier
outputs and could infer conditions from phrasing. This is partly masked
automated assessment, not independent human review or calibrated semantic gold.

Minor wording shifts, omitted table pointers, some generic editorial caveats and
entailed redundant dating were not automatically treated as material errors.
Four borderline judgments are marked: attribution ambiguity in two variance
outputs, an added further-investigation recommendation, and combined scope/
technical wording changes in an overlap-regression output. Allowing either
classification for only those flagged judgments gives these sensitivity ranges:

| Group | Material-issue sensitivity range |
| --- | ---: |
| Base / original | 9–10 of 12 |
| Base / variant | 9–10 of 12 |
| Adapter / original | 9–10 of 12 |
| Adapter / variant | 8–9 of 12 |

These are judgment-sensitivity bounds, **not confidence intervals**. They overlap,
so the two-case headline difference should not be presented as an established
improvement. Per-case notes and the four flagged decisions remain private.

All targets and original inputs were seen during adapter training. The variants
are new wording of the same twelve training targets. Forty-eight measurements
are repeated paired conditions, not 48 independent evaluation examples. Target
memorization, collocation and synthetic-edit shortcuts can support high likelihood
preference. No reserved or eight-case development records were evaluated here.
No Fama-signature, human-authorship or detector result follows.

## Technical verification

Execution took **706.2 seconds (11.77 minutes)** on one T4, with peak GPU allocation
of **5,785,751,040 bytes**. All full scoring sequences fit within **483/1,024
tokens**; the largest prompt is **275 tokens**, leaving room for the 512-token
generation budget. Prompt masks, decoded completion identities, exact score
coverage, source/target hashes and every mean/total margin were checked or
recomputed. Masked texts, claims and condition assignments are bound back to
the frozen anchor packet.

The log has no Python traceback. A pip resolver warning is present, but model
inference and all planned artifacts completed. No dependency repair, rerun or
GPU job was initiated during review. The downloaded log, runtime and JSON were
scanned without printing credentials; no credential patterns were found.

## Recommended next experiment

The smallest useful next training proposal is a **controlled input-replacement
SFT ablation**: replace the twelve selected original inputs with their faithful
variants, keep all **46 human targets and their frequencies unchanged**, and
train a fresh adapter from the pinned base using the same **46-update** budget,
seed, learning-rate schedule and original data order. Compare against the
existing original-input pilot on the unchanged eight development cases.

This would test whether a more varied input-to-target mapping helps realization
without the data-volume/update-budget confound of the previous expansion. Use
all twelve balanced anchors rather than cherry-picking the reasoning successes;
retain the numeric and scope failures. A fresh adapter is needed for that
comparison—continuing the existing adapter would change the training budget.

That is a hypothesis, not a proven remedy. Recheck the provisional variants,
preserve original uncertainty flags, specify the fixed comparison and run all
token/label preflights before execution. Strong scores on these supplied
negatives do not establish that preference optimization is needed or sufficient.
**No new training experiment is implemented or authorized by this review.**

Raw outputs and log:
`research_runs/teacher_pair_reverse_pilot_v2/semantic_anchor_inference_v1/kaggle_v1/`.
Masked evidence, judgments, provenance, per-output metrics and hash receipts:
`research_runs/teacher_pair_reverse_pilot_v2/semantic_anchor_inference_v1/review_v1/`.
Public aggregates: [anchor_inference_results.json](anchor_inference_results.json).
Offline review code: `fama_style.research.semantic_anchor_inference_review`.
