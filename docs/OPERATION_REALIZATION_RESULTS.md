# Grounded-operation realization: completed run

The notebook completed successfully, but the richer programs did **not** establish a meaningful author-style gain. The planned adapter mostly stayed close to the source instead of carrying out the requested reorganization; base Gemma used more generic exposition. The preservation control exposed incorrect citation-to-explanation binding. The predeclared style follow-up trigger was not met.

Retrieved the latest session outputs from `oliverlds/operation-realization`. The user supplied no version number; none is inferred here. Runtime was **567.2 seconds (9.5 minutes)**. All **30/30** calls completed without empty/cutoff outputs. No training occurred.

## Verified execution

Checked the prepared pack, frozen runtime hashes, artifact manifest, saved jobs/prompts, pinned model/NLI revisions, greedy decoding, saved adapter file hashes and loaded adapter-tensor identity. Recomputed all metrics and `evaluation.json` from raw generations and saved verifier/localization evidence; replay matched within floating-point tolerance. The log confirms completion. No GPU/model calls were made during review.

This was four previously inspected training transformations plus one preservation control, all seen by the saved adapter. Every prompt condition shared the same source evidence scaffold. The adapter was trained on shallow formatting plans, not the new operation format. These are reference-informed oracle diagnostics, not held-out performance, deployment results or evidence about a learned mapper.

## Four transformation cases

| Model / guidance | Substantial style gain, exploratory | Near-source copies | Exact source copies | Mean source overlap | Shallow count match |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base / ordinary | 0/4 | 0/4 | 0/4 | 0.339 | 4/4 |
| Base / shallow | 0/4 | 0/4 | 0/4 | 0.401 | 4/4 |
| Base / operations | 0/4 | 0/4 | 0/4 | 0.504 | 1/4 |
| Planned adapter / ordinary | 0/4 | 1/4 | 1/4 | 0.806 | 0/4 |
| Planned adapter / shallow | 0/4 | 0/4 | 0/4 | 0.755 | 2/4 |
| Planned adapter / operations | 0/4 | **3/4** | **1/4** | **0.930** | 0/4 |

Near-copy uses the frozen overlap threshold ≥0.90. Count matches concern the old reference-derived shallow plan; they are **not** an operation-success measure. The base matched all shallow counts in A and B without demonstrating the intended organization. C stages were explicitly not compulsory sentence slots.

All base outputs received only `slight` style grades, largely generic formalization. The adapter outputs received `none`: source-like copying, repetition or generic additions did not establish movement toward the reference's rhetorical organization. These judgments were assigned in-session with conditions visible by the same assistant that prepared the experiment; they are exploratory, unblinded and uncalibrated.

## Requested organization versus actual writing

**Premise → inference → conditional applicability:** The operation-guided adapter kept the premise and inference together in its first sentence, then retained the source's conditional-utility sentence. It was a near-copy. Base Gemma also combined premise/inference and repeated applicability statements, weakening possibility and changing boundary wording. The intended separation was not demonstrated.

**Identification difficulty → criterion introduction → definition → rationale:** The operation-guided adapter copied the source exactly, keeping criterion introduction and definition in the same source sentence. Base Gemma shortened the paragraph, but did not separately introduce and define the criterion. Ordinary base generation additionally invented a statistically significant performance advantage. Cleaner prose was not the specified rhetorical transformation.

**Scope/recommendation → existing count → staged decisions:** The operation-guided adapter reproduced the source's combined list, omitting its footnote marker. Base C introduced first/second/third markers *within one list sentence* and repeated the scope afterward; that is limited surface cue realization, not the requested staged development. Base A/B separated some decisions but put the component-count announcement at the end and introduced unsupported optimality/uncertainty commentary. None provided the full observed reference-grounded organization.

**Distinguish composition/propensity and bind comparison to supported inference:** Both operation-guided outputs retained most numerical values, but still separated expected/actual comparison from the controlled propensity inference. The adapter produced a source-like six-sentence rewrite rather than integrating the relevant claims. Some old shallow outputs also confused relative-percent and gap wording or omitted period-average detail. Correct-looking rates did not establish correct evidence/inference organization.

The core requested reorganization was not demonstrated in any of the eight C transformation outputs. Some lexical sequencing or comparison cues appeared, but there was no substantial paragraph/sentence/language improvement. This finding is about these fixed programs and this realization setup, not a disproof of the full rhetorical-taxonomy approach.

## Claim preservation and the separate control

On the four transformations, exploratory drift grades were:

| Model / guidance | None detected | Minor | Moderate | High |
| --- | ---: | ---: | ---: | ---: |
| Base / ordinary | 0 | 0 | 2 | 2 |
| Base / shallow | 0 | 0 | 2 | 2 |
| Base / operations | 0 | 2 | 2 | 0 |
| Planned adapter / ordinary | 1 | 2 | 1 | 0 |
| Planned adapter / shallow | 0 | 2 | 2 | 0 |
| Planned adapter / operations | 2 | 2 | 0 | 0 |

C was descriptively more faithful, especially with the adapter, but copying contributed to that result. `None detected` is not semantic proof, and these tiny, reused examples do not establish calibrated accuracy or superiority. The accepted allowance for minor drift is unchanged; there was no substantial style gain here to trade against that drift.

The **preservation control is excluded from the tables above**. Its task was to keep two findings and their corresponding hypothetical explanations/citations aligned. With the adapter, all three guidance conditions incorrectly attached Roll's merger-related explanation to proxy contests; some also moved or lost Mitchell/Stafford. The operation-guided output added a causal-direction disclaimer and replaced part of the second alternative-explanation block. That is a substantive attribution/content failure despite fluent prose and cautious-sounding wording. A preservation control is neither a style success nor an excuse to ignore this failure.

Base C preserved more of the two-block structure than the adapter but still broadened actor wording and strengthened evidential framing. Its control result was assessed as moderate drift, not verified preservation.

## Automatic evidence limits

All full paragraph verifier verdicts and claim-to-output-sentence NLI matrices were saved. They are fallible diagnostics: exact copying can yield high semantic consistency without realizing the operation, and composite source claims cannot determine whether a criterion was separately introduced and defined. Stage-position estimates do not measure language actions, citation binding or author style.

For example, the adapter's operation-guided near-copy/exact-copy outputs were semantically consistent on the first three transformation cases; those verdicts do not contradict their failure to reorganize the claims. Final operation assessments used the source, actual output and reference evidence, not sentence counts or an NLI gate alone.

## Decision and next informative check

The predeclared requirement of substantial style improvement on at least two transformation cases was **not met** for either model state. Do not train a mapper or extend the old shallow-plan adapter simply because C preserved more text.

Two explanations remain open: the representation/prompt may be insufficiently actionable, or this 4B realizer and the shallow-trained adapter may not follow it reliably. This experiment cannot isolate those explanations, and it did not train the model on these new operations.

The next informative diagnostic is to try a **stronger instruction-following realizer on the same fixed source evidence and operation programs**, keeping ordinary and operation-guided conditions matched. That can be a small external-model/manual-ChatGPT exercise without another Kaggle training run. Do not provide reference wording; retain the control and separate style/claim assessments. If a stronger realizer realizes the programs, investigate realization capacity or operation-conditioned supervision. If it also fails, revise the operations and representation before investing in mapping-model training. No stronger-model results are claimed here.

No follow-up notebook, external-model request, new training or GPU submission was made during review. Original data, adapters, experiments and all 45 reserved test examples remain unchanged.

Raw artifacts and logs: `research_runs/grounded_operation_realization_v1/kaggle_latest/`. Hash-bound assessments, replay verification and review manifest: `research_runs/grounded_operation_realization_v1/latest_review_v1/`. [Public aggregate results](grounded_operation_realization_v1_results.json).
