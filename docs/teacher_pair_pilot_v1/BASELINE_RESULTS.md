# Teacher-pair Gemma 4B baseline: version 1

The baseline completed successfully in 218.7 seconds on one Tesla T4. The actual
Kaggle slug is `oliverlds/teacher-pair-base-4b`; version 1 is user-reported. The
run produced all eight development outputs and both labelled training probes,
with no empty or output-limited responses. No adapter, optimizer or parameter
update was run. Input configuration, downloaded runtime source, prompt hashes,
model revision and package versions match the frozen experiment.

| Baseline measure | Result |
| --- | ---: |
| Target-token-weighted development loss | 2.96942 |
| Mean per-example development loss | 3.24636 |
| Development generations | 8/8 |
| Training probe generations | 2/2 |
| Development exact source copies | 0/8 |
| Development median source word-sequence overlap | 0.288 |
| Development median reference word-sequence overlap | 0.226 |
| Development outputs with review flags | 8/8 |
| Development material claim issues identified | 7/8 |

The two loss aggregates weight paragraph lengths differently and must not be
interchanged in later comparisons. Loss is neither semantic accuracy nor an
author-style score. All 24 label audits pass prefix masking and decoded target
coverage; full sequences range up to 508 tokens under the 1,024-token cap. The
largest generation prompt has 286 tokens, so its 512-token budget also fits.
These are actual GPU-environment token checks, not CPU estimates.

## Claim and copying findings

The main baseline weakness is unsupported elaboration and scope changes, rather
than literal copying. Seven of eight development outputs have material claim
issues in this automated review. The remaining insurance output retains the
main distinction and conjecture but loses the explicit fixed-probabilities-and-
outcomes clause and has smaller coverage/qualification flags. The classifications
are provisional, unblinded judgments; they are not independently calibrated.

| Development example | Main observed issue |
| --- | --- |
| IPO over-monitoring | Adds management/oversight mechanisms; subsequent investment becomes planned investment |
| Central versus decentralized planning | Invents coordination/accountability reasons and loses the explicit distinction from whether planning occurs |
| Probabilistic versus contingent insurance | Main account retained; explicit outcome-equivalence and some qualifiers/procedural detail are not fully retained |
| Investor ownership incentives | Adds performance assessment, enforcement and uncertainty mechanisms; weakens the strongest-incentive comparison |
| Reflection effect | Drops numeric prospects and majority preference; invents a gain-versus-guaranteed-gain example and extra caveats |
| Owners as patrons | Weakens most to a less precise proportion and ownership to control; changes the author's stated analytical purpose |
| Price-mechanism marvel | Turns a problem/question into a claimed effectiveness explanation and changes authorial/counterfactual claims |
| Bank of Italy loan register | Applies the threshold to loan portfolios rather than individual loans; adds internal/voluntary qualifications |

Both training probes also have material issues. The fee example loses the
agent-versus-principal information assumption. The monthly-price example changes
the object of the distributional comparison and statistical nonsignificance into
an assertion about small numerical deviations. They are diagnostics of training
inputs, not additional held-out validation cases; their counts remain separate.

Surface numerical checks require interpretation. In the bank-register output,
79 becomes seventy-nine and US$52 thousand becomes approximately $52,000. Those
notation changes are not counted as magnitude errors on their own. The portfolio
versus individual-loan threshold is a distinct semantic error. In the reflection
example, by contrast, probabilities and outcome values are actually omitted.

The model often produces longer or more abstract wording, but this does not
establish closer human-author style. The ownership-incentive output is nearly
three times its source length, with unsupported economic explanation. Low copying
alone is therefore insufficient. A later adapter might improve claim retention
merely by copying more; that would also be an incomplete style-transfer result.

## Implication for the prepared pilot

The baseline execution and token/label gates are satisfied. It provides a fixed
comparison for the predeclared two-epoch, 16-update QLoRA experiment, using the
same 16 training pairs and eight development cases. The next meaningful question
is whether this bounded adaptation reduces claim drift while learning useful
expression changes, instead of returning to another neutral-data reconstruction
prompt experiment.

The baseline is not a reason to edit development sources, select easier examples,
change the instruction after seeing outputs or start a full training run. Preserve
the planned base/epoch-1/epoch-2 comparison, separate claim preservation from style
and copying, and retain both training probes. Development is paper/author-held-out
from gradients but already project-exposed, so even a pilot improvement would
need confirmation on fresh material. No Fama-specific or detector claim follows.

At the time of this review, training remained disabled and unauthorized. The
user subsequently authorized the limited pilot on 31 October 2026; the enabled
notebook and exact scope are documented in [PILOT_RUN.md](PILOT_RUN.md).
No training was submitted and no new model calls were made during this review.

Downloaded artifacts are private under
`research_runs/teacher_pair_reverse_pilot_v1/kaggle_base_v1/`. Exact text-bound
assessments, metrics, numeric-notation notes and SHA receipts are in
`research_runs/teacher_pair_reverse_pilot_v1/base_review_v1/`. Original paragraphs,
outputs, configuration and the selection audit are unchanged. Public aggregates
are in [baseline_results.json](baseline_results.json). CPU import/validation is
implemented by `fama_style.research.teacher_pair_baseline_review`.
