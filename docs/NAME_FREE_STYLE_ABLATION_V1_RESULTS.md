# Author-name versus concrete style: version 1 results

**The concrete instruction improved length control and some openings, but did not establish a substantial reference-style gain with tolerable semantic drift.** Keep it as a useful presentation baseline; this result does not justify another fine-tuning run.

Run: [oliverlds/name-free-style](https://www.kaggle.com/code/oliverlds/name-free-style), version 1 reported by the owner. [Prepared comparison](NAME_FREE_STYLE_ABLATION.md); [aggregate results](name_free_style_ablation_v1_results.json).

## Execution and reproducibility

All **24 generations and 24 verifications** completed in **431.6 seconds (7.2 minutes)**. All paragraphs were nonempty, valid in format and untruncated. Model and NLI revisions, sampling settings, uploaded pack, embedded runtime identity and every prepared prompt matched. This was inference with **base Gemma 3 4B**, without an adapter or training. The reserved test examples were not read.

All **12 author-name controls exactly reproduced** the previous run's outputs. This supports the matched comparison but does not supply 12 new independent observations of control quality. Historical qualitative judgments were retained only after matching source, reference and output hashes and rereading the text.

The full verifier verdicts were saved successfully, both separately and in the metrics. Local replay using those verdicts reproduced the metrics and aggregate evaluation, within 1e-12 floating-point tolerance. All artifact hashes and verdict-to-source/output bindings passed. This replays scoring; it does not independently rerun NLI or certify semantic correctness.

The installed Kaggle session-output endpoint serves the current session rather than honoring the CLI version suffix. The owner identified version 1, and the exact prepared experiment identity was verified. Downloads and the completion log remain private.

## What improved

| Measure | Author-name instruction | Concrete name-free instruction | Paired references |
| --- | ---: | ---: | ---: |
| Mean paragraph words | 123.75 | 102.67 | 103.50 |
| Mean words per sentence | 32.82 | 23.72 | 23.02 |
| Mean bounded style-benefit proxy | 0.1200 | 0.1179 | — |
| Exact/near source copies | 0/12 | 0/12 | — |
| Valid paragraph formats | 12/12 | 12/12 | — |

The concrete prompt reduced average paragraph length by approximately 17% and sentence length by 28%, bringing both closer to the paired references. Several outputs began directly with an economic variable or tax proposition instead of a general literature-review frame. These are useful changes in presentation. Reference length is a diagnostic comparison, not a target imposed on generation.

Both arms still received positive five-feature style scores on all 12 candidates. Their nearly equal average scores show why these surface proxies cannot establish author-style success. The more direct openings also sometimes assert stronger claims than the source supports: better sentence presentation and better semantic fidelity need separate assessment.

## All-candidate qualitative assessment

All 24 outputs were read against the source and paired reference using the existing [graded policy](style_drift_tradeoff_v1/policy.json). Texts were displayed with hash-ordered labels rather than condition/seed labels, but historical outputs were recognizable and aggregate results were already visible. The same assistant designed and assessed the experiment. This is **exploratory qualitative assessment**, not independent blinded grading or calibrated accuracy.

| Judgment | Author-name instruction | Concrete name-free instruction |
| --- | ---: | ---: |
| Slight reference-style movement / generic formalization | 12 | 12 |
| Moderate or large reference-style improvement | 0 | 0 |
| Minor semantic drift | 2 | 0 |
| Moderate semantic drift | 2 | 6 |
| High semantic drift | 8 | 6 |

The concrete arm has fewer high-drift judgments but more moderate-drift judgments and no minor-drift candidates. This mixed distribution does not establish an overall winner. The two minor-drift named outputs remain eligible under the relaxed objective; their style improvement was slight. No candidate in either arm was judged to offer the substantial style benefit the project is seeking.

Recurring changes in the concrete arm include:

- Omitting the condition under which debt coefficients identify tax effects and turning an observed association into an asserted effect on value.
- Omitting the standard-deviation construction, adding an unsupported bias characterization or changing the statistical object under discussion.
- Strengthening uncertain tax explanations and adding mechanisms involving expectations or capital structure.
- Collapsing distinct tax hypotheses, or changing investors facing equal tax rates into investors preferring equal tax rates.

These are more than small shifts in emphasis. They are recorded separately from the presentation benefit, rather than rejected solely because an NLI threshold failed. The verifier itself flagged 11 named outputs and all 12 concrete outputs; those warnings are not confirmed error counts. The qualitative grades above are not a calibration of that verifier.

## Decision

Use the concrete specification as a baseline for economical expression, **not as a validated style-transfer or faithful-rewriting solution**. The named and concrete instructions change both the author label and instruction specificity/length, so this comparison cannot attribute the difference to the name alone.

The next research decision should concern whether meaningful rhetorical transformation is supported by the paired data and existing validated taxonomy annotations. Inspect those alongside the saved outputs before preparing another GPU experiment; another wording-only prompt variant would not address that question. The full learned three-level mapper was not tested here. No further notebook, model call or training job was created during this review.

## Private provenance

`research_runs/name_free_style_ablation_v1/kaggle_version_1/` retains the original downloaded artifacts and log. `version_1_review/` contains verification checks, all displayed texts, hash-bound assessments, graded metrics, aggregate summary and review manifest. Repeated controls remain identified as repeated evidence. Original packs, implementation and results are unchanged. Public files contain aggregate findings and methodology only; paper paragraphs, generation text, detailed verdicts and credentials remain excluded.
