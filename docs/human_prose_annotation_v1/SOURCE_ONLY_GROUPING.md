# Source-only boundary retrieval does not beat KEEP on reference layout

A CPU-only audit and whole-paper-held-out evaluation are complete. Of the 60
proposed training pairs, **57** have a unique representation as adjacent source
sentence merges. Two have uncertain claim alignment and one requires splitting;
these three are excluded without changing their annotations or text.

On the 57 eligible paragraphs, the fixed three-neighbor retriever reconstructs
**22 reference layouts**, versus **29 for KEEP** and **5 for fixed pairing**.
This lightweight retrieval baseline does not earn advancement under reference-layout
agreement. It has not generated prose, and neither these results nor the earlier
oracle result establishes source-only style transfer or distinctive author style.

## Merge representation and audit coverage

Source sentences are contiguous units. A dynamic program searches for partitions
whose union of claim IDs within each group equals the target sentence's claim
set in target order. Exactly one partition is required. A source boundary gets
MERGE when it falls inside a group and KEEP when it ends a group. Target sentence
text and wording are not used as input features. Split/reorder cases and ambiguous
partitions cannot be handled by this restricted action space.

| Training annotation audit | Rows |
|---|---:|
| Unique adjacent-merge representation | 57 |
| Uncertain claim alignment | 2 |
| Requires splitting | 1 |
| Total | 60 |

Among eligible rows, 29 need no merge and 28 need at least one. There are 205
source boundaries, with 39 annotated MERGE and 166 KEEP labels. One eligible
paragraph has only one sentence: all methods necessarily match it, so it is
included transparently in paragraph counts but contributes no boundary decisions.
Two eligible rows spread a claim over multiple source sentences; their complete
partitions are nevertheless unique. The search retains at most two solutions to
distinguish unique from ambiguous assignments without unbounded enumeration.

This is mechanical representability, not training-data approval or proof of
semantic equivalence. Forty-one eligible rows carry annotator review flags and
39 require external context. These flags remain visible; they are not silently
cleared or universally equated with a bad pair. Eligibility depends on reference
alignment and is therefore an **evaluation restriction**, not a source-only
runtime gate. The predictor itself does not consult that alignment.

## Frozen baseline and holdouts

The rule/configuration was frozen after the eligibility audit and before the
first baseline scoring; the private `protocol_v1.json` records code and upstream
hashes. There was no parameter search, tuning, feature selection from scores or
second fitted variant. Twelve paper folds hold out every paragraph of the query
paper from the retrieval library.

Prediction receives only source sentences. The paper ID is used for defensive
exclusion, not as a predictive feature. No query reference text, target sentence
count, claim IDs, target taxonomy, paper title or author name enters the boundary
vector or neighbor search. Gold target alignments are used only to construct
training-neighbor labels and to score the held-out query after prediction.

The twelve fixed features are paragraph sentence count, relative boundary
position, left/right sentence lengths, adjacent word-set overlap, right-sentence
reference/reason/example/contrast/addition/condition cues, and a left-sentence
semicolon flag. Count/length features use fixed caps, not all-data scaling. Word
identities are not exported as features; word overlap can still reflect topic or
paraphrase-generation patterns. These are shallow source-text cues, not the full
three-level rhetorical taxonomy.

For each query boundary, the retriever uses mean absolute feature distance,
keeps the nearest boundary from each other paper, and votes with the closest
three papers. At least two MERGE votes are required. Distance ties use fixed
paper/paragraph/boundary identifiers. All query-paper rows are excluded both at
library construction and inside retrieval. Predicted adjacent MERGE decisions
form contiguous groups covering every source sentence exactly once. No semantic
safety of those merged groups is implied before realization.

Controls are KEEP (preserve every source boundary) and fixed pairing (merge
sentences 1–2, 3–4, and so on). All methods are scored on the same eligible rows.

## Results

| Held-out result | KEEP | Fixed pairs | Retrieval, k=3 |
|---|---:|---:|---:|
| Exact reference layouts | 29/57 | 5/57 | 22/57 |
| Macro paper exact rate | 0.513 | 0.092 | 0.388 |
| Exact among 28 merge-needed paragraphs | 0 | 4 | 3 |
| Exact among 29 no-merge paragraphs | 29 | 1 | 19 |
| MERGE precision | undefined | 0.218 | 0.300 |
| MERGE recall | 0.000 | 0.667 | 0.231 |
| MERGE F1 | 0.000 | 0.329 | 0.261 |
| Boundary accuracy | 0.810 | 0.483 | 0.751 |

KEEP's precision is undefined because it predicts no merges. Its high boundary
accuracy reflects the 166/205 KEEP majority, not successful transformation.
The retriever finds nine of the 39 MERGE labels, misses 30 and predicts 21 other
merges. It also changes ten no-merge reference layouts. Fixed pairing recovers
more target merges but proposes 93 other merges. Retrieval beats KEEP in exact
layout on one paper, ties on five and loses on six. None of these summaries is
an independent significance test, and the 205 within-paper boundary decisions
are not independent samples.

The retrieval result is better than indiscriminate pairing at avoiding changes,
but it does not reproduce useful reference choices reliably enough to claim the
mapping problem solved. Only three of 28 merge-needed paragraphs are recovered
exactly. No claim-preservation, generated readability, copying or stylistic gain
was measured here because no prose was generated.

## Limits and next decision

A reference paragraph is one valid organization, not the sole acceptable layout.
An extra merge relative to the annotation is not automatically a semantic error
or a bad stylistic choice. Exact reference matching is a transparent diagnostic,
not a universal quality objective. This experiment does not show that source-only
planning is impossible; it shows that this particular fixed surface retriever
fails to improve layout recovery over KEEP.

The source segmentation and gold claim alignments came from joint GPT-6 paired
annotation, not independent source-only extraction. Therefore “source-only” here
means **query-reference-free prediction conditional on those annotations**, not
an end-to-end independently extracted pipeline. All scored rows are training/
discovery data under held-out-paper simulation. The 20 development rows were not
scored, selected for these folds or used for tuning; both reserved splits remain
outside analysis. The new multi-author pool is not the original 247-pair Fama
training set. Generalization beyond these 12 papers is unknown.

Retain KEEP as the reference-layout baseline; do not send this retriever to a
new generation batch or start GPU fine-tuning on the strength of these scores.
If continuing the mapper research, the next bounded CPU experiment should test a
small regularized boundary classifier with source-only rhetorical features and
whole-paper validation, while keeping the current retrieval result frozen. Any
hyperparameter choice must be confined to inner paper folds. Audit whether its
improvements reflect meaningful presentation choices rather than majority KEEP
or GPT-generated input patterns. A later generation check must judge alternative
plans for usefulness and meaning, rather than treating every reference mismatch
as failure. That classifier has now [completed nested evaluation](BOUNDARY_CLASSIFIER.md):
MERGE F1 improves to 0.447, with 23/57 exact layouts and no generated-style result.

## Reproduction and privacy

Implementation: `src/fama_style/research/source_only_grouping.py`.
Active private results: `research_runs/source_only_grouping_v1/results_v1/`.
Audit decisions, held-out predictions, source sentence index plans and exact
neighbor IDs/votes remain private. The result manifest pins code, source audit
and every generated artifact. Public [aggregate results](source_only_grouping_results.json)
include per-paper counts but no paragraphs, evidence quotes or per-example IDs.
The configuration is included in the aggregates; no credentials or model weights
were accessed. Prior experiments and raw annotations remain unchanged.

```sh
PYTHONPATH=src python3 -m fama_style.research.source_only_grouping --audit research_runs/human_prose_annotation_v1/audit_v2 --output /tmp/source_only_grouping
python3 -m unittest discover -s tests -v
```

Use a fresh output directory. CPU tests cover unique/ambiguous contiguous
partitions, splitting/reordering exclusions, uncertain labels, reference-poisoned
source projection, development/reserved isolation, whole-paper exclusion,
one-neighbor-per-paper voting, source coverage and metrics that expose the KEEP
majority. No external model calls, GPU jobs, neural/adapter training or detector
calls ran. This evaluates a label-retrieval baseline; it is not an LLM fine-tune.
