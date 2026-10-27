# Style tradeoff, Kaggle version 1

**The inference run completed successfully, but this small comparison did not establish substantial Fama/coauthor-style improvement.** Relaxing preservation constraints removed source copying in these outputs; the dominant change was generic academic expansion. The two shallow rhetorical operations did not establish an advantage over direct prompting. No new inference or training was submitted during this assessment.

Run: [oliverlds/style-tradeoff](https://www.kaggle.com/code/oliverlds/style-tradeoff), version 1 as reported by the owner. [Preparation and fixed comparison](STYLE_TRADEOFF_CANDIDATES.md); [aggregate results](style_tradeoff_candidates_v1_results.json).

## Execution and verification

All **24 generations** completed without truncation: four previously exposed development inputs, two prompt arms and three sampled candidates each. All were valid paragraph formats. The recorded runtime was **409.1 seconds (6.8 minutes)**, with approximately **4.50 GiB** peak allocated GPU memory. Base Gemma 3 4B and the NLI model revisions, decoding parameters, prepared pack, runtime code identity and all job hashes matched the frozen preparation. The log records completion. Training was not called; the reserved test split was not used.

Saved artifact hashes were checked. Style features, source similarity, style-distance changes and the arithmetic combining saved semantic-warning pressures were recomputed locally. Floating-point comparisons use a 1e-12 tolerance for differences between Python versions.

Two provenance limitations are explicit:

- The installed Kaggle CLI parses the supplied version suffix but does not pass it to the session-output API. The retrieved artifacts match the exact prepared experiment; the owner supplied the version number. The notebook source and metadata are retained privately alongside the download.
- The experiment saved NLI pressures, status and localized warnings, but **did not persist full verifier verdicts**. The arithmetic can be checked, but the NLI assessment cannot be independently replayed from these artifacts without executing the model again. Future experiment versions should retain those verdicts. The frozen implementation and this run were not changed or rerun.

## All-candidate assessment

Every output was read against its source and paired reference, including candidates with negative style-proxy change. Condition and seed labels were withheld from the displayed texts; hash-ordered labels were used within each case. Run-level aggregates were already visible. This was an exploratory assessment by the same Codex assistant that designed the comparison, **not an independent blinded evaluation**. The fixed [style/drift rubric](style_drift_tradeoff_v1/policy.json) was applied, and assessments are bound to exact source/reference/output hashes.

| Measure | A: direct | B: reference operations |
| --- | ---: | ---: |
| Candidates assessed | 12 | 12 |
| Exact or near copies | 0 | 0 |
| Positive five-feature style proxy | 12 | 11 |
| Mean bounded style-benefit proxy | 0.1200 | 0.0916 |
| Mean source similarity | 0.3167 | 0.3050 |
| Mean words per sentence | 32.82 | 33.91 |
| Style assessed as slight | 12 | 12 |
| Style assessed as moderate or large | 0 | 0 |
| Drift assessed as minor | 2 | 0 |
| Drift assessed as moderate | 2 | 6 |
| Drift assessed as high | 8 | 6 |

The four inputs averaged 17.70 words per sentence; their references averaged 23.02. These are descriptive averages, not a sentence-length target. The five-feature proxy rewards many generated outputs even though the qualitative assessment finds mostly generic formalization. It is insufficient as an author-style measure.

The two minor-drift direct candidates preserve the statistical method, approximate sample size and residual-correlation argument, with small added emphasis. They remain eligible under the relaxed objective. Their apparent style benefit is slight, rather than a substantial improvement rejected for imperfect semantics. Conversely, some outputs alter a theoretical coefficient prediction, replace a standard deviation with a variance, remove an essential identification condition or add an unsupported economic explanation. Those are qualitatively different from small changes in emphasis.

The verifier flagged 23 outputs and marked one consistent. Those counts are **not confirmed error counts or semantic accuracy**. The separate qualitative judgments above neither calibrate the verifier nor certify the remaining text. No best-of-three success rate is claimed.

## What can be concluded

The style result is unchanged even when the semantic-cost weight is set to zero: the qualitative style grades remain slight. This run therefore does not show that strict preservation alone was blocking strong style transfer. It also does not show that style transfer, QLoRA or the proposed learned rhetorical mapper cannot work.

Arm B has fewer high-drift judgments but no minor-drift candidates; it has lower mean surface style gain and slightly higher sentence length. These mixed outcomes on four reused cases do not establish superiority for either arm. One case received an empty plan and is only a wrapper negative control. Two other cases received only a retain-author-voice operation. This is weak guidance, not a test of the full three-level rhetorical representation.

Across arms and seeds, many outputs use inflated method descriptions or literature-review framing instead of the references’ direct statements of hypotheses, methods, comparisons and findings. Sampling produces different paragraphs but often repeats the same generic framing. Low input similarity alone is consequently not evidence of successful style transfer.

## Recommended next step

Do not initiate another fine-tuning run from this result. First isolate whether the shared author-name/formal-academic instruction is encouraging generic expansion. Prepare a very small matched inference ablation that changes only that instruction: author-name prompting versus a name-free, concrete style specification grounded in the existing references. Keep sources, seeds, model, sampling, semantic tolerance and evaluation identical. Retain all candidates and full verifier verdicts.

This would test a narrower mechanism than adding more weak optional guidance or tightening preservation again. If the name-free specification produces a clear style benefit with tolerable drift, it supplies a more useful realization prompt for the subsequent full-taxonomy experiment. This is a recommendation only; no follow-up notebook or GPU job was created during this review.

## Private provenance

Original downloads and logs are retained under `research_runs/style_tradeoff_candidates_v1/kaggle_version_1/`. The `version_1_review/` directory contains the verification checks, full assessment packet, hash-bound judgments, graded metrics, aggregate summary and review manifest. Real paragraphs, raw generations, private annotations and credentials are excluded from the public files. The original pack and results remain unchanged.
