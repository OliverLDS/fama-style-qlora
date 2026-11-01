# Preference-versus-SFT pilot: successful execution, no generation improvement

User-reported run: `oliverlds/preference-pair-pilot-4b`, version 1.
Both approved continuations completed, but neither passes the
[predeclared quality gate](PILOT_PROPOSAL.md). No additional training or
detector scans were started.

## Execution verified

The frozen Gemma 3 4B / NF4 / FP32 environment used one Tesla T4. There were
22 optimizer updates per arm, 44 finite nonzero gradient checks and 40 completed
development generations across five snapshots. Both arms started independently
from the original expanded-pilot epoch-2 adapter, changed their weights, and
passed final adapter reload checks. All four saved adapters were downloaded;
their tensor headers have the expected 136 F32 language q/v tensors and
2,228,224 parameters. Both final file and tensor fingerprints match the report.
The starting adapter's tensor fingerprint was also recomputed locally.

All 42 processor/token-mask checks passed, the two real manual/built-in likelihood
comparisons agreed, and all eight unchanged-control generations reproduced their
archived texts exactly. The immutable reference cache covers 34 completions;
both arms passed the initial policy/reference identity check. Update ordering,
sibling weighting, linear learning-rate schedule, cache hashes and snapshot
boundaries were verified offline against the frozen packet.

Runtime inside the pilot was **1,216.4 seconds (20.3 minutes)**, including its
model access, reference preparation and evaluation. Peak allocated GPU memory
was **5.10 GiB**. SFT's arm took 402.3 seconds; DPO's took 554.2 seconds, including
their respective checks and evaluation. These measurements exclude the first
notebook cell's dependency setup. The notebook log ends around 21 minutes,
within the one-hour cap.

`evaluation.json` and `state.json` both record completion. The downloaded log
contains nonfatal 401 warnings from PEFT's optional remote base-config lookup
while saving adapters. The warning explicitly says the lookup is ignored;
model loading, inference, optimization and verified adapter saving/reloading
completed. This does not indicate a missing Secret for the main model load.
No credential values were found by the local pattern scan.

## Fixed epoch-2 comparison

Eight reused development cases, four papers; no reserved examples. Epoch 2 was
fixed as primary before execution. Epoch 1 remains diagnostic.

| Measure | Unchanged adapter | SFT continuation | DPO continuation |
| --- | ---: | ---: | ---: |
| Token-weighted development target NLL | 1.73675 | 1.71356 | 1.71476 |
| Generations exactly identical to unchanged adapter | 8/8 | 6/8 | 5/8 |
| Material claim issues, provisional review | 4/8 | 4/8 | 4/8 |
| Within the one-minor/no-material local budget | 3/8 | 3/8 | 3/8 |
| Exact copies of the input | 0/8 | 0/8 | 0/8 |
| Empty or token-limited outputs | 0/8 | 0/8 | 0/8 |
| Generic expression versus control: better / tied / worse | — | 0 / 7 / 1 | 0 / 7 / 1 |

The two final arms tie on generic expression in all eight paired comparisons.
The material errors persist: invented uncertainty/efficacy claims, a changed
efficiency-assessment direction, omitted numerical prospect comparisons, and
weakened ownership quantifiers/relations. The changed price-mechanism paragraph
has a less clear ending in both continuations; no substantive error is resolved.
Other differences are mostly anaphora, tense and connective wording.

Median source word-sequence overlap remains 0.36861 in all three conditions.
Median reference overlap changes from 0.35643 to 0.35238 in both continuations.
These surface measures do not establish semantic alignment or author style.
Identical generations refer to matching the **control output**, not copying
the input; the latter remains zero under the exact-copy definition.

Neither arm reaches the required at-most-three material issues and at-least-four
locally eligible outputs. Neither wins on generic expression, and DPO does not
outperform matched SFT. **No detector scans are warranted for these checkpoints.**
No distinctive Fama-style or human-style milestone benefit is established.

## What the likelihood measurements show

All seventeen training rows improve their chosen-minus-rejected log-probability
margin relative to the frozen starting policy. Mean relative improvements are
3.747 nats for SFT and 3.834 for DPO. Diagnostic final DPO losses are 0.5242 and
0.5207, respectively, versus approximately 0.6931 at initial identity.

However, all seventeen actual rejected responses still have higher absolute
summed log probability than their corresponding human targets in both arms.
This comparison is length-sensitive and does not directly predict greedy output.
The measured probability shifts and verified weight changes show that updates
occurred; they did not produce the desired free-generation changes on development
examples. SFT also improves preference margins, so that movement alone is not a
preference-specific advantage.

An insufficient update magnitude is a plausible explanation, **not an established
cause**. This run cannot separate limited training effect from poor transfer,
label limitations or model/task constraints. It does not show that DPO, QLoRA or
Gemma 4B cannot work in another setting.

## Next diagnostic

Before more optimization, check free-running outputs on one fixed eligible
training prompt from each of the eleven paragraph groups. Compare unchanged
adapter, final SFT and final DPO, reusing exactly bound archived control outputs
where possible. This would need at most 22 new generations, with no training.
It tests whether the updates change behavior even on the examples they optimized:

- Little training-side change would favor a separately authorized bounded
  optimization-strength check over collecting more detector reports.
- Clear training-side improvement with unchanged development behavior would
  instead point toward transfer/data coverage rather than more repetition alone.

This is a recommendation, not a submitted inference job or approval for another
training run. Preserve the completed pilot and its mixed/negative findings.

## Reproducibility and limits

Private raw downloads and log: `research_runs/actual_error_pilot_v1/kaggle_v1/`.
Masked bundle, case-wise key and judgments:
`research_runs/actual_error_pilot_v1/review_v1/`.
The result manifest/receipt records hashes of the packet, approval, raw files,
annotations and review code. Public aggregate data: [pilot_results_v1.json](pilot_results_v1.json).
CPU reviewer: `src/fama_style/research/preference_pilot_review.py`.

The automated reviewer knew the historical control outputs and their semantic
boundaries. Final outputs were read in a fixed masked order; identical outputs
inherit the same prior judgments, and changed wording was checked against those
boundaries. This is not independent or calibrated judging. Generic-expression
ranks are not a Fama-style benchmark. Eight repeatedly used development examples,
one seed and seventeen correlated training pairs do not support a broad
superiority or impossibility claim. The Kaggle CLI's output endpoint does not
enforce its parsed version argument; version 1 is user-reported. Executed runtime,
helper, configuration, decoding and adapter content were instead verified by
their frozen hashes. No new GPU work or reserved-test access occurred in review.
