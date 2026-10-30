# Nested regularized boundary classifier: partial signal, unreliable layouts

The CPU classifier evaluation is complete. With the same twelve source-text
features and the same 57 eligible training/discovery pairs, nested ridge logistic
regression improves annotated MERGE F1 from **0.261 to 0.447** versus the frozen
retriever. It matches **23/57** complete reference layouts, versus **22/57** for
retrieval and **29/57** for KEEP. There is some predictive signal for reference
merge decisions, but this does not establish reliable source-only planning or
better generated writing. No new generation or GPU/LLM training ran.

## Fixed model and nested validation

The previous [representation audit](SOURCE_ONLY_GROUPING.md) is reused unchanged:
57 uniquely adjacent-merge-representable pairs from 12 papers, 205 boundaries,
39 MERGE labels, 166 KEEP labels. Two uncertain-alignment rows and one split case
remain excluded. One single-sentence paragraph is necessarily correct for every
method and contributes no boundary decisions. Eligibility remains conditioned
on target representability and is not a source-only runtime filter.

The classifier uses the same twelve capped features as retrieval: sentence
count, boundary position, left/right lengths, adjacent word overlap, six initial
reference/connective categories and left semicolon presence. No query reference,
claim IDs, target count, target taxonomy, title, author or paper identifier enters
the vector. Paper IDs are used only to split folds. Source sentence segmentation
still comes from joint paired GPT-6 annotation, not an independently extracted
source-only pipeline.

The model minimizes weighted mean logistic loss plus half the regularization
coefficient times the squared slope norm. The intercept is unpenalized. It uses
fixed feature scales, so there is no all-data fitted scaler. The stdlib Newton
optimizer has backtracking and explicit convergence checks; no ML package,
weights download or credential access is required.

Before the first classifier scoring, the private protocol froze:

- Penalties: 0.01, 0.1 and 1.0.
- Class weighting: none or balanced, calculated from each fit's training labels.
- MERGE thresholds: 0.35, 0.5 and 0.65.
- Selection: pooled inner-fold MERGE F1, then precision, macro paper exact rate,
  stronger regularization, higher threshold and unweighted fit as tie-breakers.

There are 18 candidates. For each of 12 outer paper holdouts, the other 11 papers
undergo inner leave-one-paper-out validation. Each inner fit excludes both the
outer paper and its own inner holdout. Thresholds are scored on pooled inner
out-of-fold probabilities. The selected settings are then refit on all 11 outer
training papers and applied to the outer query source. Only afterward are outer
labels used for evaluation. Class weighting changes the fitting objective;
outputs are not calibrated probabilities of semantic safety.

All **792 inner fits and 12 outer fits converged**, in at most five Newton
iterations. Eleven outer folds select balanced weighting, penalty 1.0 and
threshold 0.5; one selects balanced weighting, penalty 0.1 and threshold 0.5.
No classifier candidate or feature was changed after seeing its outer scores.
The earlier retriever results on these papers were already known, so this is
exploratory follow-up research, not an untouched confirmatory benchmark.

## Results

| Paper-held-out measurement | KEEP | Fixed pairs | Retrieval | Nested classifier |
|---|---:|---:|---:|---:|
| MERGE precision | undefined | 0.218 | 0.300 | **0.459** |
| MERGE recall | 0.000 | 0.667 | 0.231 | **0.436** |
| MERGE F1 | 0.000 | 0.329 | 0.261 | **0.447** |
| Boundary accuracy | 0.810 | 0.483 | 0.751 | 0.795 |
| Exact reference layouts | **29/57** | 5/57 | 22/57 | 23/57 |
| Macro paper exact rate | **0.513** | 0.092 | 0.388 | 0.408 |
| Exact among 28 merge-needed paragraphs | 0 | 4 | 3 | **5** |
| Exact among 29 identity paragraphs | **29** | 1 | 19 | 18 |

The classifier finds 17 annotated merges, proposes 20 other merges and misses
22 annotated merges. It changes eleven identity layouts. KEEP gets high boundary
accuracy from the KEEP majority and cannot recover any merge-needed paragraph.
The classifier is selected to improve MERGE F1 rather than maximize that majority
accuracy or complete-layout agreement; these objectives expose different costs.

Compared with retrieval, classifier exact-layout counts improve on three papers,
tie on seven and worsen on two. Compared with KEEP, they improve on one, tie on
seven and worsen on four. The aggregate F1 improvement is descriptive, not a
significance claim: the boundaries and paragraphs within papers are correlated,
and there are only twelve outer papers.

An extra merge relative to the reference is not automatically a bad rewrite or
semantic error. The reference supplies one author-chosen layout among possible
valid alternatives. These scores measure agreement with that layout. No generated
readability, stylistic benefit, copying or claim drift was evaluated. The apparent
improvement over retrieval does not validate a distinctive multi-author or Fama
style signal.

## Decision and next step

The model class matters: regularized linear learning extracts more signal from
the same shallow features than this nearest-neighbor rule. Still, only five of
28 merge-needed layouts are recovered exactly, and roughly half of predicted
MERGE actions differ from the annotations. Do not launch fine-tuning, a larger
mapper or an automatic generation rollout on the basis of this result.

The next useful step is a **versioned audit of the 20 extra and 22 missed merge
boundaries**, grouped by paper. Distinguish annotation/claim-granularity problems,
plausible alternative presentation choices and genuinely unhelpful merges. Treat
that as discovery, not a new held-out score. Freeze this result while auditing;
do not rewrite labels and claim an improved score on the same folds.

That audit can guide targeted collection of additional reliable MERGE and KEEP
examples from new papers and tell us whether richer source-only rhetorical
features are justified. It is premature to say that more arbitrary human papers
would solve the problem, or that these twelve cues prove the structure is
unpredictable. The [42-boundary discovery audit](BOUNDARY_ERROR_AUDIT.md) has now run,
with no labels or classifier scores changed. Additional collection has not run. The 20
human-prose development examples and both reserved splits were not used in fitting,
threshold selection or evaluation. The original Fama data and experiments remain
unchanged.

## Artifacts and verification

Implementation: `src/fama_style/research/boundary_classifier.py`.
Private run: `research_runs/boundary_classifier_v1/results_v1/`.
The protocol pins implementation, inherited feature/label code and audit hashes.
Private artifacts retain outer-fold models, every candidate's inner score,
training-paper traces and query probabilities/plans. No final all-data deployment
model was selected or fitted. Public [aggregate results](boundary_classifier_results.json)
contain counts and per-paper scores, without paragraphs, per-example evidence,
query IDs or model coefficients.

```sh
PYTHONPATH=src python3 -m fama_style.research.boundary_classifier --audit research_runs/human_prose_annotation_v1/audit_v2 --frozen-baseline research_runs/source_only_grouping_v1/results_v1 --protocol research_runs/boundary_classifier_v1/protocol_v1.json --output /tmp/boundary_classifier
python3 -m unittest discover -s tests -v
```

Use a fresh output directory. New CPU tests verify gradients/Hessians against
finite differences, finite regularized fits, the unpenalized intercept, train-only
class weights, held-out-paper exclusion, and invariance of an outer model's
selection/predictions to changes in that outer paper's labels. Existing tests
cover source projection, development/reserved isolation and baseline grouping.
Classical logistic models were fitted on CPU; no external model calls, GPU jobs,
LLM/adapter training, detectors or model downloads were performed.
