# Matched plan-conditioned realizer pilot: version 1 results

The training pipeline worked, and adaptation reduced reference loss sharply. On the four selected development examples, the planned adapter with its plan prompt produced fewer serious semantic errors than the matched direct adapter. **It did not establish better Fama/coauthor-style transformation:** two outputs were near-copies, sentence-count adherence was weak, and no development output reached the predeclared substantial-style grade.

Kaggle: `oliverlds/plan-realizer-pilot`, owner-reported version 1. Total runtime **955.5 seconds (15.9 minutes)**. Both fresh adapters completed **24 updates / four epochs over 12 pairs**, with identical initial adapter hashes, changed trained weights and verified save/reload hashes. All **36/36** generations completed without empty/cutoff outputs. Existing experiments and the reserved test split remain unchanged.

## Execution verification

Retrieved logs, adapters and result artifacts read-only. Verified the prepared pack hash, runtime files, artifact manifest, model/NLI revisions and configuration. Revalidated every update's epoch/example schedule, positive finite gradients, adapter initialization/reload contracts, complete generation matrix, prompt hashes and token-weighted development losses. Recomputed every output metric from raw generations and saved full verifier verdicts; replay matched within floating-point tolerance. No training or inference calls were made during review.

The session-output endpoint does not guarantee immutable historical version selection; version 1 is the owner's designation, supported by pack/runtime/configuration identity and the complete execution contract.

## Four-example development comparison

| Model / prompt | Reference loss | Near-copies of source | Target sentence count matched | High drift, exploratory | Substantial style, exploratory |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base / ordinary | 3.8924 | 0/4 | 0/4 | 4/4 | 0/4 |
| Base / oracle plan | 3.6254 | 0/4 | 3/4 | 3/4 | 0/4 |
| Direct adapter / ordinary | 1.7495 | 1/4 | 2/4 | 2/4 | 0/4 |
| Direct adapter / oracle plan | 1.7677 | 1/4 | 1/4 | 1/4 | 0/4 |
| Planned adapter / ordinary | 1.7961 | 0/4 | 2/4 | 2/4 | 0/4 |
| Planned adapter / oracle plan | 1.7391 | 2/4 | 1/4 | 0/4 | 0/4 |

Near-copy means source overlap at least 0.90 under the frozen similarity function. None was an exact source copy. These are four repeatedly used development rows from one held-out paper, selected for short feasible plans, not an estimate of generalization or statistically significant superiority.

The primary planned condition's loss was only **0.0103 nats lower** than the matched direct condition. Loss measures reference-token prediction, not free-generation style or semantic fidelity. Under the same plan prompt, the planned adapter had lower loss and fewer assessed high-drift outputs than the direct adapter, but this tiny comparison cannot establish an advantage for plan-conditioned training.

## What improved and what remains wrong

**Trimming regressions:** With the plan prompt, both adapters retained the expected-slope distinction, possible standard-error effect, observed comparisons and non-excessive-trimming conclusion. The ordinary-prompt adapters incorrectly attached the unreported evidence to the excessive-trimming clause. The planned output was a close source paraphrase, not a substantial style transformation.

**Regression specifications:** The planned adapter with its plan stayed faithful to the described variables and table organization. Other conditions invented significant investment/value relationships or an incoherent firm-value-on-firm-value issue. This is a useful fidelity improvement on this example, but the successful output was a near-copy and did not reproduce the target sentence organization.

**Dividend signaling:** Both direct-adapter outputs and the planned adapter with its plan produced exactly the same near-copy, removing the source's tentative opening. The planned adapter without a plan instead attributed a conditional premise as an asserted claim and repeated a conclusion. No condition learned the reference's more explicit contrast between costly and cost-free models.

**Debt and taxes:** The ordinary direct adapter incorrectly stated that observed relationships held under fully controlled profitability. The planned condition avoided that error but omitted the key conditional-identification proposition: debt slopes would identify tax effects *if* profitability information were fully controlled. A shorter, apparently safer summary still lost part of the reasoning.

The planned primary condition received two `none_detected`, one `minor` and one `moderate` drift assessments; the direct primary condition received one `minor`, one `moderate` and two `high`. `None_detected` is not proof of equivalence. All grades were assigned by the in-session assistant with conditions visible, using the existing separate style/drift rubric. They are not independently blinded or calibrated domain-expert judgments.

## Style versus preservation

The planned condition's mean source overlap increased to **0.810**, versus **0.675** for direct SFT and **0.224** for the base ordinary prompt. Better preservation here partly came from remaining close to the input. Generic exposition decreased, but distinctive reference-style rewriting did not appear.

The predeclared follow-up trigger—substantial style improvement on at least two development examples without more high drift—**was not met**. This is not a rejection of minor drift; no substantial style gain was observed to justify the tradeoff. The planned condition's sentence-count adherence was only 1/4 and average word-band adherence 0.354, versus 2/4 and 0.479 for direct SFT under its ordinary prompt.

Exploratory style/drift utility also depends on priorities. With the existing bounded-drift weight, the planned primary condition scored 0.059 versus −0.016 for direct SFT. With the more style-first weight they tied at 0.084; with style alone, direct scored 0.150 versus planned 0.100. Thus the planned condition does not dominate the tradeoff. These utilities encode the assessor's ordinal judgments and chosen weights, not validated quality measurements.

Both primary adapter conditions had two `model_consistent` and two risk verdicts from the frozen NLI verifier. The equal counts missed important differences in error severity. Full verdicts are retained as diagnostics; they are not converted into correctness probabilities or selection gates.

## Training probes and cost

The two training probes remain separate from development. Neither adapter consistently reproduced reference structure: the planned dividend probe still used six sentences for a five-sentence plan, and the random-walk probe added content or changed framing. There was no substantial author-style gain on these probes either. Lower training loss did not demonstrate either exact memorization or successful structural realization.

Both arms supervised **5,876 target tokens** across all epochs. Direct training processed 19,768 total sequence tokens in 172.2 measured update seconds; planned training processed 27,240 in 224.7 seconds. Planned prompts therefore added approximately **38% sequence-token cost** and **30% update time**. These times exclude loading, evaluation, generation and NLI; the full notebook took 15.9 minutes. Both arms ran on one visible T4.

## Recommendation

Freeze this result as evidence that supervised adaptation can suppress some unsupported exposition, while the distinctive-style objective remains unsolved. Do not extend epochs or train the rhetorical mapper on the strength of lower loss alone.

The current plan controls only **sentence count, length bands and semicolons**. It is a formatting projection, not the full discovered rhetorical taxonomy. A negative style result here does not establish that richer, claim-bound rhetorical planning is ineffective.

Before another training run, prepare a small set of **reference-grounded rhetorical operations** from the existing audited annotations—for example, conditional identification followed by empirical contrast and a qualified conclusion, or an explicit contrast between alternative explanations. Bind each operation to source-supported claims, distinguish it from mere sentence-length control, and check that the target actually realizes it. Use existing references as targets; do not turn these generated summaries into new labels. This focuses the next experiment on the missing style signal rather than repeating the same shallow-plan training longer.

No follow-up experiment, GPU submission or new training was initiated during this review. The original model and dataset files, previous runs and all 45 reserved test examples remain untouched.

Private artifacts: `research_runs/plan_realizer_pilot_v1/kaggle_version_1/plan_realizer_pilot_v1/`; hash-bound assessments and replay checks: `research_runs/plan_realizer_pilot_v1/version_1_review/`. [Public aggregate results](plan_realizer_pilot_v1_results.json).
