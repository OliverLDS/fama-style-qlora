# Actual taxonomy pairs and the training signal

**There is evidence for specific sentence-grouping operations, but the current annotations do not yet support training a reliable full rhetorical mapper.** The important issues are annotation consistency and matching schemas to the texts actually used in training—not an absence of all wording contrast. This audit used saved training data and responses only. No inference, annotation request or GPU job was submitted.

## What data we actually have

| Material | Available evidence |
| --- | --- |
| Frozen training corpus | 247 pairs from seven papers |
| Nonempty legacy target-taxonomy field | 196 records; heterogeneous target metadata, not paired standardized schemas |
| Discovery sample | 28 actual-source/target LLM annotation pairs |
| Fresh canonical sample | Seven additional actual-source/target annotation pairs |
| Total pairs covered by these paired LLM annotations | **35**, leaving 212 outside this paired collection |
| Proposed replacement inputs | 28 proposals: 13 retain the existing text, 15 revise it |
| Proposed revisions applied to the frozen corpus | **Zero** |
| Proposed catalogue | 132 categories; not all independently validated |

The original 247 pairs have real lexical contrast: the earlier [training-signal analysis](RESEARCH_EXTENSION.md) found median token-sequence similarity 0.495 and no near-identity pairs at its 0.85 threshold. That does not establish that the rhetorical labels measure a learnable stylistic transformation.

The 15 proposed inputs must remain separate from current training inputs. A schema describing a proposed replacement is not a schema for the original input. This audit joined texts by training IDs and hashes, rather than by annotation order, and analyzed actual and proposed pairings separately. Source data and proposals were not modified or accepted as semantically verified repairs.

## Apparent structural differences include annotation variation

Of the 2,508 discovery label occurrences, 720 still require contextual allocation among candidate categories. The earlier 39-case context-resolution exercise did not authorize propagating its decisions to all other occurrences. This audit preserves the unresolved mappings.

The positive-label lists also do not establish absence. A source-only or target-only observed label may reflect feature selection, anchoring or granularity rather than an actual stylistic change. We therefore report observed-set overlap, not binary feature-transition accuracy.

The most informative control is the **13 proposed inputs whose text was retained exactly**. Their annotations were nevertheless generated in a different context:

| Median observed-category Jaccard | Actual discovery source → target, 28 pairs | Identical text, different annotation context, 13 controls |
| --- | ---: | ---: |
| Paragraph level | 0.400 | 0.125 |
| Sentence level | 0.143 | 0.000 |
| Language level | 0.250 | 0.333 |

The main-claim-position observation changes in **six of those 13 identical-text controls**, versus 12 of 28 actual discovery pairs. Text is unchanged in the controls, so those differences cannot themselves be genuine source-to-target transformations. The control is not an independent random sample or a universal noise estimate: paired versus unpaired annotation context, label selection and unresolved catalogue splits can all contribute.

The seven fresh canonical pairs have higher median observed-set overlap: paragraph 0.438, sentence 0.571 and language 0.625. The two fresh masked repeats have medians 0.400, 0.693 and 0.667, respectively. These small sets do not establish broad reliability. The prior fixed-unit check is more encouraging for a limited checklist, but it covers **two texts**, not all 132 categories or cross-text claim alignment. F03/F04/F12 remain disputed there.

## Sentence counts are useful but not a universal rule

Using the existing legacy sentence splitter on all 247 training pairs gives:

| Reference count relative to input | Pairs |
| --- | ---: |
| Same | 133 |
| Fewer | 74 |
| More | 40 |

These are **heuristic counts**. Inspection found a false grouping signal: parenthesized sentences caused one six-sentence reference to be counted as four by the legacy splitter. Its existing annotation partitions reconstruct the text as six sentences. The old parser and historical results were not changed. Candidates with parser/annotation boundary disagreement were excluded from operation selection.

For the 35 paired annotations, after checking that their sentence units reconstruct each supplied text exactly, annotated counts show **16 unchanged, ten fewer and nine more** reference sentences. This is also a deliberately selected sample, not a population estimate. Neither count method supports merging every paragraph.

A CPU-only leave-one-training-paper-out probe predicts the exact target sentence count from source sentence count and a fixed source-length bucket. Conditional counts use five prior-equivalent observations for smoothing; no deployable mapper is saved.

| Probe | Exact-count accuracy | Macro-average across papers |
| --- | ---: | ---: |
| Retain source count | **53.8%** | **52.0%** |
| Fixed target-count prior | 27.5% | 31.9% |
| Smoothed conditional counts | 50.6% | 49.4% |

The conditional probe does not beat identity on exact counts. Its change-direction accuracy is 54.7%, versus identity's 53.8%; that small descriptive difference is not evidence of a useful style mapper. The probe uses heuristic boundaries and measures count prediction, not semantics or author style. It sees neither development-validation nor reserved-test data.

## Concrete operation hypotheses prepared

After the boundary screen, four papers provide grouping candidates in the annotated sample. Three pairs were inspected and given explicit, source-sentence coverage hypotheses:

| Training paper | Input → reference sentences | Proposed grouping of source sentences |
| --- | --- | --- |
| Fama 1980 | 6 → 4 | `[1,2] [3] [4] [5,6]` |
| Disappearing Dividends | 7 → 5 | `[1] [2] [3,4] [5] [6,7]` |
| Volatility Lessons | 4 → 2 | `[1] [2,3,4]` |

These integrate an already supplied qualification, comparison, explanation or inference. They cover every source sentence exactly once in order. They do not authorize inventing a new causal link. The fourth candidate concerns a retrospective industry-comparison summary with temporal material folded into a conclusion; it remains a less specific candidate rather than an executable plan.

The three cards record obligations such as current versus future compensation, expected versus actual dividend initiation rates, and substantial versus nontrivial downside probabilities at different horizons. This is an **assistant-proposed sentence-level alignment**, not independent atomic-claim verification. A coverage check does not prove semantic equivalence, and none of these groupings has been executed or marked training-ready.

The private source-only plans omit reference wording and target hashes. Because the groupings were chosen after inspecting references, they are **oracle diagnostics**, not deployment predictions. They can support a narrowly interpretable realization check if one is subsequently authorized, instead of asking a mapper to learn uncertain label-count differences across the full catalogue.

## Recommendation

Keep direct QLoRA as the existing baseline and retain the concrete rewriting prompt for its presentation benefit. For the structural branch, use the three source-bound grouping cards as the next concrete research material. First establish that each preserves the supplied atomic claims and that realizing its grouping improves writing under the graded style/drift objective. Do not train the full 132-category mapper from the current positive-label differences, silently apply the 15 proposed input revisions, or treat higher annotation counts as stronger style.

This does not reject paragraph- or language-level structure. It identifies which evidence is currently actionable: exact text partitions and inspectable grouping hypotheses are stronger than context-sensitive, incompletely allocated label differences. More detailed structures can follow once measured under a common, fixed-unit annotation procedure.

## Reproducibility and private artifacts

The new stdlib module is `src/fama_style/research/taxonomy_signal.py`:

```sh
PYTHONPATH=src python3 -m fama_style.research.taxonomy_signal \
  --output-dir research_runs/taxonomy_training_signal_v1/new_audit
```

Private input files are required. The final executed numerical audit is `research_runs/taxonomy_training_signal_v1/audit_v3/`. It records input hashes, actual/proposed pairs, identical-text controls, category observations and ranked candidates. Assistant grouping cards and source-only plans are separately retained as `grouping_hypotheses_v2.private.jsonl` and `source_only_grouping_plans_v2.private.jsonl`; `hypothesis_release.json` identifies the active versions. Earlier drafts remain preserved and were never used for generation. [Aggregate JSON](taxonomy_training_signal_v1_results.json) contains no paper paragraphs.

All **218 CPU tests** and the original `train.py --check` passed. New tests cover identical-text annotation differences, unknown absence, the parser boundary artifact, exclusion of ambiguous candidates, paper-held-out prediction, hash tampering and source-coverage guards. No local model weights were accessed. Training data, original annotations, experimental code and results were left unchanged; the reserved test split and development-validation file were not read by this audit.
