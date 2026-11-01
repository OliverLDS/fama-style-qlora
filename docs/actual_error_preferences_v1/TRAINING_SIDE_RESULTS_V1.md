# Training-side free generation: targeted errors persist

User-reported run: `oliverlds/preference-training-inference-4b`, version 1.
All 22 generations completed. The two final adapters still fail to correct the
known material errors on these eleven training inputs. No new training or
detector scans were initiated during review.

## Verified execution

- Eleven fixed paragraphs across ten papers: nine original inputs and two
  variants. No selection changes after execution.
- Eleven archived control responses reused unchanged, plus eleven new SFT and
  eleven new DPO responses from the completed pilot's fixed epoch-2 adapters.
- All eleven generation prefixes match the verified training tokens. Executed
  source, configuration, model revision, package versions and decoding match
  their frozen hashes. Both snapshots report all parameters frozen.
- `evaluation.json` and `state.json` record complete inference, zero optimizer
  steps and `training_called=false`. No target paragraphs were supplied to
  generation.
- Runtime: **333.2 seconds (5.6 minutes)**. Peak allocated GPU memory: **4.51 GiB**
  on one Tesla T4. The log ends around 6.2 minutes including setup/conversion.

## Paired results

| Measure | Archived control | Final SFT | Final DPO |
| --- | ---: | ---: | ---: |
| Material claim issues, provisional automated review | 11/11 | 11/11 | 11/11 |
| Within the one-minor/no-material local budget | 0/11 | 0/11 | 0/11 |
| Outputs exactly identical to control | 11/11 | 8/11 | 8/11 |
| Cases with material errors resolved | — | 0/11 | 0/11 |
| Generic expression versus control: better / tied / worse | — | 0 / 10 / 1 | 0 / 10 / 1 |
| Exact input copies | — | 0/11 | 0/11 |
| Empty or token-limited new outputs | — | 0/11 | 0/11 |

The control's eleven errors are **by construction**: these inputs were selected
from actual training-side errors. The counts are not a representative training
accuracy estimate and should not be compared as an error rate with the earlier
eight development cases.

Both new conditions tie in expression on all eleven direct SFT-versus-DPO
comparisons. Most edits are small phrasing or qualifier changes. The larger
return-regression change retains the confusion between overlap/sampling benefits
and predictive accuracy/power; the archived version is slightly clearer. Other
persistent failures include reversal of an economic-justification condition,
loss of an exclusive reason, weakening of temporal/quantifier claims, and
altered risk-premium and IPO-bundling relationships.

Median source word-sequence overlap rises from 0.36681 to 0.39779 for both arms;
reference overlap rises from 0.31868 to 0.33621. The median longest shared source
span remains five words. These small surface shifts do not demonstrate correct
claims or improved style. Exact equality with the **control response** is distinct
from copying the **input**.

| Error family | Selected paragraphs | Material issues: control / SFT / DPO |
| --- | ---: | ---: |
| Scope | 3 | 3 / 3 / 3 |
| Numbers/comparisons | 3 | 3 / 3 / 3 |
| Reasoning | 3 | 3 / 3 / 3 |
| Time | 2 | 2 / 2 / 2 |

## Interpretation and next decision

The earlier development result cannot be explained solely by failure to
generalize: the selected errors also persist on seen training inputs. Combined
with verified weight changes and better teacher-forced margins, this shows
limited behavioral change under the current optimization budget. It does not
identify the unique cause or establish that Gemma, QLoRA or DPO cannot work.

The next useful intervention is **one stronger, bounded SFT-only continuation**:
keep the same seventeen pairs, eleven groups, starting adapter and 22 updates,
but fix learning rate at `1e-5` instead of `1e-6`. This tests update strength
without adding data, another objective or a parameter search. The original
expanded-pilot training used `2e-5`; that context motivates the proposed scale,
but does not establish that it will work for this continuation.

Compare its fixed final checkpoint with the archived control and weak-SFT
checkpoint on the same eleven seen inputs and eight development cases. Keep
error correction, new errors, copying and expression separate. Defer further
DPO experiments until a chosen-only update demonstrably changes the targeted
behavior. Do not buy or collect more detector scans for the present adapters.

This is a **proposal only**, requiring explicit authorization before any GPU
training. No new training notebook, job or experiment result is implied here.
No full-scale training or detector benefit is established.

## Artifacts and limits

Private run and log: `research_runs/preference_training_inference_v1/kaggle_v1/`.
The review folder contains a reproducible masked bundle, key, exact-byte-bound
judgments, per-output measurements and hashed receipt/manifest:
`research_runs/preference_training_inference_v1/review_v1/`.
Public aggregates: [training_side_results_v1.json](training_side_results_v1.json).
CPU reviewer: `src/fama_style/research/preference_training_inference_review.py`.

The same automated reviewer knew the historical errors and controls. Case-wise
labels were fixed before reading outputs, and identical texts must tie; this
does not establish independent or calibrated judging. No human review was
required. The source/reference pairing, inherited context flags and tiny
error-selected sample remain limitations. Generic expression is not an objective
Fama-style benchmark. No reserved examples or detector services were accessed.
As in the prior review, the Kaggle CLI does not enforce its parsed output-version
argument: version 1 is user-reported, while actual runtime/configuration content
is verified against the frozen packet. Credentials remain outside the repository;
the downloaded log passed the local credential-pattern check.
