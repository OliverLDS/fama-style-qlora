# Style–semantic drift tradeoff: saved-output reassessment

The objective now permits minor semantic drift when it buys meaningful movement toward the paired authors’ reference style. Verifier rejection and abstention are retained as evidence rather than used as automatic exclusion rules. No generation, training or Kaggle submission was performed.

## What changed

The separate offline evaluator is `src/fama_style/research/style_drift.py`; its explicit policy is [policy.json](style_drift_tradeoff_v1/policy.json). Historical experiment outputs, prompts, scores and verifier implementations are unchanged.

Two evaluation layers remain separate:

1. **Exploratory numerical proxies.** Style benefit is the change in `1 / (1 + distance)` to the existing training-reference profile. The profile measures sentence length, semicolons, author voice, sentence-initial conditions and parentheses. Semantic-warning pressure combines omission, unsupported-addition and contradiction signals, with localized actor/relation warnings shown separately. NLI confidence is not interpreted as the amount of semantic drift.
2. **Qualitative assessment.** Read source, paired reference and output; grade style as none/slight/moderate/large and drift as none detected/minor/moderate/high/unknown. Minor drift changes nuance or detail while preserving main propositions; higher grades identify material supporting or central changes. These are exploratory assistant judgments, not independent semantic certification.

The ranking is `style benefit − weight × semantic cost`, with four disclosed sensitivity weights: 0, 0.1, 0.25 and 0.5. Qualitative style values are 0/0.2/0.6/1; drift costs are 0/0.15/0.5/1. These numbers express a policy preference, not measured probabilities or an empirically optimal tradeoff. Proxy and qualitative utilities use different scales and must not be pooled. Unknown severity stays unknown. Copy diagnostics remain separate; an unchanged-source control has zero style gain and zero change cost.

No severity category is categorically banned by the ranking. For example, moderate style improvement with minor drift scores `0.6 − 0.25 × 0.15 = 0.5625`, above the unchanged-source control. The moderate-style threshold identifies substantial improvement; it does not exclude weaker candidates from inspection.

## Executed results

The evaluator verified previously recorded artifact hashes and reused development outputs from soft guidance, ledger/grouping ablation and sentence-preserving editing. Repeated controls and derived paragraph views are not independent generations. Reserved test examples were not read.

| Observation | Count |
| --- | ---: |
| Development cases | 21 |
| Recorded observation views | 56 |
| Available paragraph views | 54 |
| Unavailable views from one malformed edit script | 2 |
| Unique candidate paragraphs after deduplication | 48 |
| Previously risky/abstaining candidates retained | 33 |
| Positive five-feature style-proxy candidates | 12 |
| Candidates on a within-case proxy frontier | 8 |

All 12 positive-proxy candidates were read against their source and paired reference. They were graded **slight** in style improvement, predominantly generic formalization rather than sustained reference-grounded rhetorical improvement. Their drift judgments were **3 minor, 4 moderate and 5 high**. None of these 12 met the substantial-style threshold. The other 36 were not qualitatively assessed; this is not evidence that none could have a useful stylistic change missed by the five-feature proxy.

The three minor-drift candidates remain eligible and score above copying under the default qualitative utility. Thus accepting some drift does recover usable small changes; this reassessment does not establish a significant Fama/coauthor-style improvement. Some numerical style gains instead reward wordier phrasing. Other candidates change finding direction, the scope of an economic actor, a regression variable or a qualification, with little apparent stylistic benefit.

The qualitative audit was unblinded, performed by the same Codex assistant implementing the evaluator, and selected after observing proxy gains. It is a diagnostic judgment of a small reused subset, not an independent benchmark, a calibrated model evaluation or a claim about all 48 candidates. Reference text was used for assessment only; no new generator received it. Text hashes bind the saved judgments to their source/reference/output, making reuse reproducible without claiming reproducibility of the original subjective assessment.

## Reproduce locally

Run the CPU-only command from the project root, choosing a fresh output directory:

```sh
PYTHONPATH=src python3 -m fama_style.research.style_drift \
  --output-dir research_runs/style_drift_tradeoff_v1/rescore_new \
  --assessments research_runs/style_drift_tradeoff_v1/assessments_v1.private.json
```

Omit `--assessments` for proxy-only rescoring. Private saved inputs are required and are intentionally excluded from the public repository. Executed artifacts are under `research_runs/style_drift_tradeoff_v1/rescore_v2/`: private candidate paragraphs, source controls, hash-bound assessments, per-case proxy frontiers, aggregate summary and a hash manifest. The malformed script remains invalid; this evaluation does not repair it into a successful output.

Validation: the full CPU unittest suite passed, including six new tradeoff tests; `python3 train.py --check` also passed without accessing model weights or starting training. Tests cover nonzero drift eligibility, unknown severity, scoped role warnings, identity controls, Pareto comparisons and assessment-text binding.

## Recommendation

Use this graded objective for the next inference comparison. Select a small fixed development subset before generation, include direct rewriting alongside less restrictive style guidance, and compare several candidates on separate style and drift axes. Permit minor drift, disclose moderate changes, and examine high-cost changes explicitly. Do not select simply for passing NLI or increasing sentence length. The next useful evidence is a concrete reference-style benefit under this objective; the existing small gains do not justify another fine-tuning run yet.
