# Source-bound grouping: version 1 results

**The run completed, but the oracle grouping plan was not reliably realized and did not establish substantial reference-style improvement.** The grouped arms were less expansive than the plain-prompt anchor, and the exploratory semantic assessment was more favorable for the oracle arm. Those partial gains do not establish successful structural transfer or support training a mapper yet.

Run: [oliverlds/bound-grouping](https://www.kaggle.com/code/oliverlds/bound-grouping), version 1 reported by the owner. [Prepared experiment](BOUND_GROUPING_REALIZATION.md); [aggregate JSON](bound_grouping_realization_v1_results.json).

## Execution and artifact verification

All **27 generations and paragraph/group diagnostic records** completed in **488.6 seconds (8.1 minutes)**. Outputs were nonempty, untruncated and valid paragraph formats. There were **24 unique same-source/output candidates**: three paired comparisons repeated the exact same output. Repeated text receives one shared qualitative judgment, with all 27 observation views retained.

The uploaded pack hash, prepared jobs, runtime code, pinned base Gemma 3 4B/NLI revisions and sampling parameters matched. All saved artifact hashes passed. Local replay from full paragraph and group verdicts reproduced the saved metrics and evaluation with 1e-12 tolerance for floating-point differences. Group verdicts were checked against their source-unit groups and output-sentence hashes. The log records completion. No adapter or training was used, and neither development-validation nor reserved-test data was used in this diagnostic.

The installed CLI's session-output endpoint serves the current session rather than honoring the version suffix. The owner identified version 1, and the exact prepared experiment identity was verified. Replay reuses saved NLI verdicts; it is not independent NLI re-execution or semantic certification.

## Primary comparison: oracle versus identity groups

| Measure | A: concrete only | B: identity groups | C: oracle groups |
| --- | ---: | ---: | ---: |
| Outputs | 9 | 9 | 9 |
| Requested sentence-count matches | No count requested | **5/9** | **2/9** |
| Count-matched outputs with every positional NLI check consistent | Not applicable | 3/5 | **0/2** |
| Mean paragraph words | 101.1 | 93.3 | 92.2 |
| Mean words per sentence | 20.69 | 18.79 | 19.01 |
| Mean bounded style-benefit proxy | 0.0145 | 0.0217 | 0.0225 |
| Exact/near source copies | 0 | 0 | 0 |

The positional NLI row concerns only count-matched outputs. It is a fallible diagnostic, not confirmed error prevalence or proof that all other bindings failed. Count-mismatched outputs remain visible with positional alignment unresolved. No sentence-boundary ambiguity was flagged in this run.

| Pair | Requested B count | B observed counts, seeds 41/42/43 | Requested C count | C observed counts, seeds 41/42/43 |
| --- | ---: | --- | ---: | --- |
| Managerial human capital | 6 | 5 / 6 / 6 | 4 | 5 / 5 / 4 |
| Dividend initiation | 7 | 5 / 5 / 6 | 5 | 6 / 5 / 6 |
| Premium-distribution horizons | 4 | 4 / 4 / 4 | 2 | 4 / 4 / 4 |

In **3 of 9 matched B/C seed comparisons**, changing only the group assignments produced identical output text. Two of those were the premium example, where all oracle outputs retained four sentences despite the requested two. This establishes limited response to the plan in these cases; it does not identify whether model capacity, prompt representation or another realization factor is responsible.

Both count-matched C outputs also have the same count in A for that input and seed. Count agreement therefore supplies no unique evidence that the oracle plan caused the organization; direct inspection of content placement is necessary. Identity counts and compressed oracle counts also differ in difficulty, so the 5/9 versus 2/9 comparison alone cannot establish general superiority of identity planning.

## Count agreement and claim grouping are different

The two count-matched oracle outputs were inspected directly:

- **Managerial example, seed 43:** broadly follows the intended four-group organization. However, some detail is lost and a qualified statement about immediate pay becomes categorical. Its overall drift was assessed as moderate. The NLI warnings are useful evidence, but do not erase the observed partial grouping success.
- **Dividend example, seed 42:** produces five sentences and preserves most numerical meaning, so drift was assessed as minor. It puts period-average statistics into output sentence 3 although the plan assigned them to position 4; the final-year comparison moves into position 4, while the final conclusion remains separate instead of joining it in position 5. Thus a mostly faithful paragraph can match the count while missing the requested grouping mechanism.

This distinction matters for the research question: a count target is not a substitute for source-to-output claim binding.

## Separate qualitative style and drift judgments

All 27 views were read against the sources and references using the existing [graded policy](style_drift_tradeoff_v1/policy.json). Exact duplicate candidates received shared judgments. Displayed texts used hash-ordered labels without condition/seed labels, but aggregate results were already visible and the assessor designed the experiment. These are **exploratory assistant judgments**, not independently blinded, calibrated style or semantic accuracy.

| Judgment | A: concrete only | B: identity groups | C: oracle groups |
| --- | ---: | ---: | ---: |
| Slight style movement / generic formalization | 9 | 9 | 9 |
| Moderate or large reference-style improvement | 0 | 0 | 0 |
| Minor drift | 0 | 3 | 3 |
| Moderate drift | 5 | 4 | 6 |
| High drift | 4 | 2 | 0 |

The oracle arm's more favorable drift distribution is a useful descriptive observation. Under the saved default ordinal utility, its mean is 0.104 versus 0.076 for B and 0.019 for A. These arbitrary-scale policy utilities are not accuracy or effect-size estimates. With style weight alone, all arms have the same qualitative value: style improvement remained slight. Three selected training pairs cannot establish general superiority.

The dividend outputs often preserve the central expected/actual rates and comparisons. Elsewhere, recurring changes include lost uncertainty about compensation, unsupported causal connectives, and wording that recasts chance negative-premium outcomes as persistence. Some premium outputs also strengthen or blur the distinction between substantial shorter-horizon probabilities and nontrivial longer-horizon probabilities. These were graded individually rather than treating every verifier warning as a confirmed error.

Whole-paragraph NLI marked 1/9 A outputs consistent, 3/9 B and 3/9 C. These counts are not semantic accuracy. The qualitative result gives credit to minor drift, while showing that tolerating it did not uncover a substantial author-style transformation in this sample.

## Decision

Do not train a full structural mapper or launch another QLoRA run from this result. Keep the source-bound plans and full verdicts as useful diagnostic material. The unresolved mechanism is reliable realization of assigned groups, with semantic bindings and stylistic benefit assessed separately.

If the structural branch continues, the next architectural question is whether realizing each source group separately and assembling the paragraph works better than asking Gemma to plan the entire paragraph internally. Such a comparison would need to expose costs and cross-sentence coherence, retain all failed/uncertain cases and keep the graded drift objective. It would test execution of a plan, not learned mapping. This is a recommendation only; no follow-up job or notebook was created during this review.

## Private provenance

Original artifacts and logs remain under `research_runs/bound_grouping_realization_v1/kaggle_version_1/`. `version_1_review/` retains hash checks, the full text packet, 24 unique assessments, 27 graded observation records, inspections of the two count-matched oracle outputs, summary and review manifest. Original packs, code and results were left unchanged. Public files contain aggregate findings and methodology, without source paragraphs, raw generations, detailed semantic evidence or credentials. No model calls or GPU jobs were made during review.
