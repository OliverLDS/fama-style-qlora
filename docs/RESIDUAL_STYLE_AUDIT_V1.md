# Residual style audit: expression differences and information limits

**Sentence packing is only part of the remaining gap.** The six references differ from the twelve generated outputs in grammatical theme, consequence-clause form, label continuity and register. Three bounded control candidates emerge, but none has been tested and none is established as uniquely Fama-like. Several conspicuous differences also involve potentially missing economic meaning or confidence, which a style model should not invent.

This audit reuses the exact six training pairs and twelve responses from the [ordinary-versus-plan comparison](PAPER_PLAN_COMPARISON_V1_RESULTS.md). No generation, training, PDF/OCR verification, annotation request or held-out evaluation was performed. Original data and responses remain unchanged.

## Existing three-level representation retained

The existing annotation schema was inspected before defining observations. Its axes are:

- `paragraph_structure`: main-claim position, moves, progression and qualification placement.
- `sentence_structure`: sentences, clauses, claim references and features.
- `language_features`: dimensions/labels, claim references, evidence and certainty.

The existing grounded-operation catalogue already links sentence actions and language actions to source claims and semantic edges. This audit maps observations to those axes; it does not replace the taxonomy, assign new canonical labels or overwrite earlier annotations. Per-case cards contain observations for all three levels, source/reference/output hashes and the earlier grouping assessments. Feature cards preserve exact source/reference/output evidence spans; potential semantic gaps are kept in a separate file.

These are in-session, reference-visible, unblinded interpretations. Exact quotes and hashes establish inspectability, not semantic correctness or independent author-style measurement.

## Recurring observations beyond sentence counts

| Observation | Selected-paper evidence | What remains after packing matches |
| --- | --- | --- |
| **An economic pattern, relation or institutional function is the grammatical theme** | New Lists 2003, Fama–Jensen 1983, Agency Problems 1980 | References foreground the profitability pattern, system architecture or management function. Outputs more often foreground the table report, organizations or managers performing an action. All three guided outputs already match the reference claim grouping. |
| **An existing consequence receives an explicit finite predication** | Disappearing Dividends 2000, Agency Problems 1980 | References make the consequence subject/verb explicit within an existing argument. Both dividend outputs use a participial result; the planned managers output uses a participial stake consequence. Ordinary managers already states that conclusion finitely, but separately. This is not a universal preference: New Lists uses an appositional inference. |
| **Stable short labels are reused** | New Lists 2003, Disappearing Dividends 2000 | References repeatedly use cohort/quantity labels. Outputs vary between expanded firm descriptions and share/payer paraphrases. Some guided wording already uses the source label. This is a modest continuity preference, not a demonstrated deficiency in otherwise clear alternatives. |
| **Technical prose coexists with plain evaluative or figurative wording** | Market Efficiency 1997, Agency Problems 1980, Volatility Lessons 2018 | References include informal classification imagery, plain wealth imagery and good/bad-news framing. Outputs are more neutral. This contradicts treating the target as uniformly more formal; exact metaphors/evaluations cannot safely be copied without checking meaning. |

The support counts are **two or three of six purposively selected papers**, not corpus prevalence estimates. There is no comparison-author corpus, so these observations cannot establish author specificity. The twelve outputs share one realizer/setup; differences are not necessarily model limitations or caused by the prompt.

### Paragraph level

Much of the broad move progression is already retained: comparison then interpretation, hierarchy then examples then monitoring, favorable evidence then risk, and findings/presumptions then conditional explanation. Guidance closes several grouping gaps, as previously reported. A major new paragraph-tree transformation is not supported by these six residuals.

There is still a change in *what the sentence/paragraph presents as its object of discussion*. A reference can frame an institutional function or empirical pattern directly where an output frames an actor/report. That is a theme/predication choice, not an additional claim, ranking or universal statement to be invented. In the managers paragraph, this difference remains after the previously successful four-sentence organization is achieved.

### Sentence level

Finite result clauses, predication type, apposition and local placement of qualification can differ inside an already matched claim group. In particular, the dividend consequence and the planned managers stake consequence provide two examples of finite versus participial realization. Their relevant causal edges are explicitly present in the existing source maps, with the same claim endpoints.

The reference and generated New Lists paragraphs show the reverse caution: the reference uses an appositional inference. Therefore “make everything a finite clause” is not an evidence-backed global rule. Similarly, reference numerical parentheticals and horizon-first probability framing are useful case observations, not sufficiently recurring evidence for a new universal syntax prescription.

### Language level

Some residual terminology can be supplied from the input itself, such as a previously established cohort label. Other apparent vocabulary gaps involve specialized concepts that may not be equivalent to their plain-English renderings. A source-only style instruction must distinguish these cases rather than receiving a target-derived terminology glossary.

The references are **longer than both outputs in all six cases** under the same regex word counter:

| Paper | Reference | Ordinary | Planned |
| --- | ---: | ---: | ---: |
| New Lists 2003 | 88 | 66 | 63 |
| Disappearing Dividends 2000 | 101 | 90 | 94 |
| Fama–Jensen 1983 | 89 | 55 | 57 |
| Market Efficiency 1997 | 176 | 128 | 142 |
| Agency Problems 1980 | 143 | 86 | 92 |
| Volatility Lessons 2018 | 71 | 68 | 62 |

This does not justify padding, a length quota, blanket nominalization or greater jargon density. Some extra reference wording performs framing or qualification; some may encode information inadequately represented in the input. Length and register alone are not the missing author-style objective.

## Three bounded, untested control candidates

The [machine-readable candidate catalogue](residual_style_audit_v1_controls.json) records finite IDs, existing taxonomy fields, evidence papers and semantic guards:

1. **`FOREGROUND_SUPPORTED_RELATION_OR_FUNCTION`** — Where an existing claim concerns an economic pattern/relation or institutional function, consider making that concept the grammatical theme. Preserve actors and scope; do not add centrality, an importance ranking, universality or a new definition.
2. **`FINITE_SUPPORTED_CONSEQUENCE`** — Where an explicit source edge supports a consequence, consider a finite clause with its existing subject rather than only a participial tail, within the same claim grouping. Preserve endpoints, direction, conditions and force. This is selective; the reference evidence includes a counterexample.
3. **`REUSE_SOURCE_DEFINED_LABEL`** — Use a compact cohort/referent/quantity label already available in the source when continuity benefits. Do not add technical aliases, change units/populations or forbid clear pronouns.

These are **candidate instructions**, not implemented mechanical transformations, learned rules or demonstrated improvements. No new generator prompts contain them, and the ordinary/default prototype has not changed. Mixed register remains an observation requiring care, rather than a fourth automatic operation.

## Four potential meaning-alignment gaps

| Pair | Potential gap | Why it is not automatically style |
| --- | --- | --- |
| New Lists 2003 | Lower post-listing profitability versus a decline in profitability | A relative low level does not itself specify a temporal decline. Wider paper context may matter but is absent from this input. |
| Fama–Jensen 1983 | Approval versus ratification; skills versus human capital | Institutional/technical scope may differ. Treating these as interchangeable target-style vocabulary needs justification. |
| Market Efficiency 1997 | A presumption versus an explicitly safe-to-presume formulation | The reference expresses confidence that should not simply be added to the neutral source. |
| Agency Problems 1980 | Committed human capital/service value versus rented wealth/rental rates | The rental/economic-model representation may carry information beyond a stylistic synonym. |

These are **potential alignment flags**, not independently adjudicated extraction errors or automatic pair rejections. All exact evidence is saved privately. No source, reference, confidence marker or technical term was silently repaired.

This does not overturn the previous source-to-output fidelity assessment: outputs can faithfully preserve their supplied input while failing to reconstruct a reference distinction that the input did not clearly contain. If confirmed, such omissions make some apparent style targets partly underdetermined and may encourage a model to guess missing content. That is a plausible risk, not a demonstrated causal explanation of earlier QLoRA behavior.

## Recommendation

Before testing richer controls or training, resolve these **four flagged source–reference alignments in a separate, versioned repair audit**. Establish whether the existing input supports the reference distinction, needs a meaning-preserving clarification, or remains uncertain. Preserve all originals and do not weaken a reference or add unsupported input facts merely to make the pair fit.

Once input meaning is adequately specified, the three candidate controls can be tested selectively against the current ordinary/shared-evidence baseline. They should not all be applied indiscriminately. The current audit does not justify mapper training, a new QLoRA run, a claim of author specificity or a wider annotation campaign.

## Verification and artifacts

Parent package, audit and review hashes passed; all quoted observations resolve to exact spans. Claim/relation references are valid, and finite-consequence examples require explicit existing source edges. Every original pair/response is unchanged. No executable model/training code changed, so this step adds no tests that merely assert interpretive style grades and does not rerun model evaluation.

Private audit/specification, six three-level case cards, feature cards, semantic flags and verification are in `research_runs/residual_style_audit_v1/review_v1/`. Public files contain aggregate observations and source-independent candidate instructions, not copyrighted paragraphs. Preparation and analysis read only the selected training material; the 21 development and 45 reserved test examples remain outside this audit.

[Aggregate audit results](residual_style_audit_v1_results.json). [Candidate controls and guards](residual_style_audit_v1_controls.json).
