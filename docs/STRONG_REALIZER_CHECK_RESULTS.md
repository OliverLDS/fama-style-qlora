# External-realizer check: response results

This is a **provisional positive mechanism result**. The operation-guided responses carried out three of four requested rhetorical reorganizations more clearly than ordinary prompting, while preserving the separate control's citation/explanation bindings. This demonstrates that these programs can be actionable in another realization setup. It does not prove distinctive Fama authorship style, cross-paper generalization or a causal model-size effect.

The user reports **GPT-6 for both batches**. Underlying model revision, decoding, token limits, latency and whether separate new conversations were used are not independently verified. Five-job batch context and a JSONL delivery wrapper differ from the single-job greedy Kaggle run. Possible memory effects and familiarity with the papers remain uncontrolled.

## Import and provenance

All **10/10** records passed strict job-ID, prompt-hash, batch and JSONL checks. No duplicate, missing, repaired or selected-best response occurred. Original files were preserved byte-for-byte. Recomputed CPU surface metrics from exact outputs and fixed source/reference pairs; prepared inner prompts match the frozen Kaggle A/C jobs. The prepared generation files exclude reference paragraphs, although the plans are reference-informed oracle side information. Actual chat contents were not audited.

No model/API/NLI inference or GPU job was run during this review. The reported semantic and rhetorical assessments below are in-session assistant judgments with conditions visible, not independent blinded or calibrated domain-expert evaluation. `None detected` is not proof of semantic equivalence.

## Four transformation examples

| Reference-grounded operation | Ordinary response | Operation-guided response | Incremental finding |
| --- | --- | --- | --- |
| Separate premise, inference and conditional applicability | Premise and inference remain combined | Three distinct moves, retaining the uncertainty and inclusive upper bound | Core operation realized; moderate structural/style gain |
| Introduce criterion separately from definition and rationale | Difficulty, criterion and definition are combined | Separate retrospective criterion introduction; definition and persistence rationale then follow | Core operation realized; moderate gain; exact reference sentence packing differs |
| Stage the existing portfolio decisions | Scope/recommendation and one combined decision list | Separate component-count announcement; first decision distinguished from coordinated remaining decisions | Core operation realized; moderate gain; one minor strength change |
| Bind dividend comparison to its supported inference | Already integrates expected/actual comparison with the controlled inference | Similar integration; final single-year evidence and conclusion remain separate | Partial operation; no material gain over ordinary |

The first three positive cases are **all from Fama's 1965 paper**. The dividend case comes from another paper and supplies no incremental win. This concentration is a major limit on claims about Fama/coauthor style more broadly.

Sentence counts are not the operation criterion. For example, the portfolio response realizes the requested count-introduction/first-versus-remaining-decision organization in three sentences, whereas the reference has five. The analyst response separately introduces and defines the criterion, but combines its definition and rationale with a semicolon. Those are supported rhetorical moves, not exact reproduction of reference wording or boundaries.

## Style, drift and copying remain separate

| Four transformation cases | Ordinary | Operation-guided |
| --- | ---: | ---: |
| Moderate reference-grounded style/structure grade | 1/4 | 4/4 |
| Core operation realized | 0/4 | 3/4 |
| Incremental moderate gain over ordinary | — | **3/4** |
| Drift: none detected / minor / moderate / high | 4 / 0 / 0 / 0 | 3 / 1 / 0 / 0 |
| Near source copies, overlap ≥0.90 | 0/4 | 0/4 |
| Exact source copies | 0/4 | 0/4 |
| Mean source overlap | 0.796 | 0.729 |
| Old shallow target count matched | 0/4 | 1/4 |

The ordinary dividend response already earns a moderate grade because it integrates the comparison and controlled inference while retaining precise economic quantities. Thus the four moderate C grades are **not four incremental victories**. The guidance adds a meaningful difference in three cases.

The minor drift is the portfolio phrase “requires three decisions,” which is stronger than the source's “involves three decisions.” It does not change the decisions, count, scope or footnote, but the strength difference should remain recorded. This is the kind of limited shift the existing tradeoff policy permits when a substantial structural/style benefit is demonstrated.

For random-walk applicability, probability force and the condition that dependence does not exceed the acceptable level remain intact. For the analyst example, sustained market outperformance over many periods and the short-run no-skill counterexample remain. Both dividend outputs retain cohort identity, characteristic/propensity distinction, year/rate roles, 3.8-point gap and the less-than-one-sixth comparison. No invented mechanism or numerical reversal was identified in these ten outputs.

“Moderate” here means movement toward the supplied reference's paragraph/sentence/language features. These are not validated author-identification scores. The operations are also meaningful generic rhetorical devices; following them alone does not establish that the text is recognizably or uniquely Fama-like.

## Preservation control

Both external conditions retain the parallel merger/proxy-contest findings and their hypothetical alternative explanations. Roll stays with the poor-investment merger explanation; Mitchell/Stafford stays with acquirers' pre-merger performance; the proxy-contest alternatives remain separate. Neither response adjudicates the alternatives or adds the causal-direction disclaimer seen in the adapter run.

The CPU citation-context alert did not fire. The positive preservation judgment comes from reading the actual source and outputs, not absence of that limited alert. This control is excluded from the four-case style counts; retaining an already-parallel structure is not a style-transformation win.

## Comparison with the preceding Gemma diagnostic

Under the same *inner* operation prompts, base Gemma and the shallow-trained adapter had no substantial style cases in the prior review. The adapter near-copied three of four transformation inputs and copied one exactly; it also misbound citations in the preservation control. Here the operation-guided external responses have no near/exact copies, realize three core operations, and preserve the control bindings.

This contextual comparison is useful but not causal. Model identity, decoding, batching, outer delivery format, hidden context and source familiarity differ; the adapter was not trained on the new operation format. The result does not establish that parameter count alone caused the difference or that operation-conditioned QLoRA on 4B will succeed.

Under the existing exploratory utility weights, the mean ordinary score is 0.300. Operation-guided scores range from 0.600 with style-only weighting to 0.581 with the largest drift penalty, reflecting the one minor change. These numbers encode unblinded ordinal judgments and policy weights, not validated measurement accuracy.

## Decision

The predeclared provisional trigger is met: three incremental moderate gains, no increase in high-drift cases, and intact control bindings. The supported conclusion is **representation actionability on these reused examples**, not completed style learning or mapper readiness.

Next, expand the three successful operation families into a **small paper-diverse source/plan/reference set**, aiming for roughly 20–30 eligible training pairs if the frozen corpus supports them. Keep original author paragraphs as targets; do not substitute these GPT-generated rewrites as author-style labels. Require explicit source-claim/condition/quantity-role binding and record unsupported candidates rather than forcing a family onto every paragraph.

Then prepare a fresh operation-conditioned realization pilot against a matched direct-SFT arm, using the actual richer operation format at both training and inference. The old shallow-plan adapter need not be trained for additional epochs. Keep existing development data separate for oracle realization checks and all 45 reserved test examples excluded. A learned rhetorical mapper should wait for evidence of realization across papers and more than one example per operation.

No new training set, notebook, API request or GPU job was created during this review. Existing data, adapters and experiments remain unchanged.

Private response import: `research_runs/strong_realizer_check_v1/response_audit_v1/`. Hash-bound assessments, declared model metadata and verification: `research_runs/strong_realizer_check_v1/response_review_v1/`. [Public aggregate results](strong_realizer_check_v1_results.json).
