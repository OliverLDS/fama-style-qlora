# Concise-plan realization: Kaggle version 1

Reviewed `oliverlds/concise_plan_realization`, version 1. All eight generations
completed: four frozen training examples, ordinary instruction (A) versus the same
instruction plus a concise oracle presentation plan (B). Base Gemma 3 4B was used
without an adapter or training. No development-validation or reserved-test rows
were used. This is a small realization diagnostic, not held-out performance.

## Results

| Measurement | A: ordinary | B: concise plan |
|---|---:|---:|
| Exact requested sentence count | 2/4 | 3/4 |
| Mean absolute sentence-count error | 0.50 | 1.25 |
| Mean word-band adherence | 20.8% | 12.5% |
| Semicolon-budget adherence | 2/4 | 2/4 |
| Automated NLI proxy passes | 2/4 | 2/4 |
| Near copies / exact copies | 0 / 0 | 0 / 0 |
| Mean style-distance improvement proxy | -0.584 | -1.478 |
| Mean words per sentence | 41.3 | 67.1 |

Higher style-distance improvement is better; negative values mean movement away
from the frozen training-reference feature profile relative to the input. This
is a surface-feature proxy, not a validated measure of author identity or quality.

| Training paper key | Requested sentences | A produced | B produced |
|---|---:|---:|---:|
| F1965 | 3 | 4 | 3 |
| FF2000 | 2 | 2 | 2 |
| NL2003 | 2 | 2 | 2 |
| VL2018 | 6 | 5 | 1 |

The last planned output is genuinely one 153-word sentence connected by
semicolons, rather than a sentence-segmentation artifact. Thus the better exact
count rate conceals a severe failure. B improves the style proxy for two examples
and worsens it for two; the largest deterioration is the six-to-one collapse.
The NLI pass total also hides a swap: NL2003 changes from pass to flagged,
whereas VL2018 changes from flagged to pass. Its passing label does not establish
that the long sentence preserves all claims or avoids added implications.

## Semantic and experimental limits

Source/reference NLI flags three pairs (F1965, FF2000, VL2018); only NL2003 passes.
The flags mostly reflect low entailment rather than strong contradiction. They
may indicate source information loss, target differences, or limitations of
sentence-unit NLI. They do not establish that three labels are wrong. All four
examples remain in the report; none was removed after seeing outputs.

The independent NLI checker accepted all three equivalent controls and rejected
all seven semantic mutations. These small synthetic sanity checks do not measure
its accuracy on economics prose. Surface syntax, style, and copying diagnostics
also remain limited proxies. No result here validates claim preservation.

The plan jointly controls sentence counts, length bands, coarse clause hints and
semicolon budgets. This experiment cannot isolate which control causes a failure.
It tests a narrow closed-field subset of the discovered taxonomy, not the full
representation or a learned source-to-target mapper. Oracle plans contain
reference-derived structural information and are not deployment predictions.

## Verification and runtime

The uploaded pack SHA-256 matches the frozen local pack:
`83d834c37bfb5233c2a3dbf8b620ff522e8b467c9f0e1440f57c676257b9e0ee`.
Downloaded runtime sources match local sources. All manifest file hashes match;
jobs and prompt hashes match the frozen pack; all eight outputs are nonempty,
non-synthetic and below the output limit. Recomputing metrics and the summary
from saved generations and NLI probabilities reproduces them exactly. The log
contains neither a Python traceback nor a CUDA out-of-memory message.

The runner reports 196.6 seconds total (3.28 minutes), including about 99.8 seconds
of generation and 8.4 seconds for GPU NLI after unloading Gemma. This is runner
time, not Kaggle queue time or the complete notebook dependency-install time.

Private logs, generations and verification hashes are retained under
`research_runs/concise_plan_v1/`; no reference paragraphs are included here.
The preparation document remains a record of the pre-execution protocol.

## Decision

Do not proceed to structural-mapping or additional QLoRA training on this evidence.
The immediate bottleneck remains reliable realization of even a known plan.

The next useful inference-only experiment would separate count control from the
other controls and request a structured array with one sentence per slot. Validate
the array length and sentence boundaries before joining, preserve first-attempt
failures in the results, and report any repair attempts and their costs separately.
Use the same frozen examples and ordinary baseline, with semantic checking that
cannot be passed merely by satisfying the output shape. Structured formatting
alone cannot guarantee preservation of claims or better style. Do not train a
mapper unless realization improves without a semantic regression.

No new GPU job was submitted during this review.
