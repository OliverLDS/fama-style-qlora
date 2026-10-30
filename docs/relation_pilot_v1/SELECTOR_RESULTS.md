# CPU selector: length baseline wins; learned selector fails

On **35 faithful cases with a decisive readability judgment**, the length-only
selector makes 24 correct decisions, versus 17 for KEEP and eight for the
regularized source-feature classifier. All four outer folds independently select
the same **30-word source-pair cutoff**. This favors retaining a simple baseline,
not expanding the learned mapper on this small sample.

## Locked comparison

The input contains 52 earlier MERGE-preferred proposals from four papers.
Only 35 faithful, decisive cases are fitted/scored in the primary comparison:
18 MERGE and 17 KEEP. Nine faithful ties and eight claim-shift cases are excluded
from every fit and threshold choice, then scored separately. They are not
silently turned into negative labels.

Four outer folds hold out one whole paper, including all its paragraphs and
overlapping local pairs. Inside each outer training set, three inner paper folds
select the logistic probability threshold. The length baseline evaluates a
fixed threshold grid on those same training papers; its rule has no fitted
coefficients, so its fixed-rule inner predictions need no separate fit.
Threshold selection maximizes mean per-paper accuracy, breaking ties by fewer
selected merges and then the more conservative threshold. The KEEP and MERGE-all
controls require no fitting.

The logistic model reuses the earlier twelve source-only features: sentence
count, boundary position, left/right length, lexical overlap, right-sentence
reference/reason/example/contrast/addition/condition cues, and a left semicolon.
No candidate, verifier claim, rationale, judgment, reference paragraph, author
identity or topic vocabulary is supplied as a predictor. Paper IDs route folds;
they are not model features. Source segmentation comes from frozen Stage 2
source-only annotations, not independently verified parsing.

Ridge penalty is fixed at 1.0 with an unpenalized intercept and no class weighting.
Probability thresholds are 0.35, 0.5, 0.65, 0.8 and 1.0; the last allows KEEP
everywhere. Length thresholds are 0, 20, 30, 40, 50, 60, 80 and 160, selecting a
merge when the two source sentences' combined count is at most the threshold.
Counts use the existing `[A-Za-z]+(?:'[A-Za-z]+)?` word regex, not the earlier
whitespace diagnostic. Numeric tokens are not counted by this legacy word
function; the cutoff is specific to this implementation, not a universal rule.

Code, solver, feature implementation, settings and input hashes were frozen
before executing actual predictions. Prior judgments and descriptive length
patterns were already inspected: this is an exploratory locked run, **not a
prospectively preregistered or pristine confirmatory experiment**. No settings
were retuned after seeing the poor classifier result.

## Primary outcomes

| Method | Correct / 35 | Accuracy | Mean paper accuracy | Beneficial merges selected | Worse merges selected |
|---|---:|---:|---:|---:|---:|
| KEEP | 17 | 48.6% | 50.3% | 0 | 0 |
| MERGE all proposals | 18 | 51.4% | 49.7% | 18 | 17 |
| Length only | **24** | **68.6%** | **66.9%** | **11** | **4** |
| Source logistic | 8 | 22.9% | 22.2% | 4 | 13 |

Length-only precision is 11/15 (73.3%) and recall is 11/18 (61.1%); it misses
seven judged-beneficial merges. MERGE-all has slightly higher pooled MERGE F1
(0.679 versus 0.667), illustrating why no single metric establishes superiority.
The locked selection criterion is mean paper accuracy; all metrics are retained.

| Held-out paper | Primary cases | KEEP correct | Length correct | Logistic correct |
|---|---:|---:|---:|---:|
| Grossman–Stiglitz 1980 | 10 | 2 | 7 | 2 |
| Hart–Moore 1990 | 11 | 8 | 8 | 3 |
| Modigliani–Miller 1958 | 6 | 5 | 3 | 1 |
| Spence 1973 | 8 | 2 | 6 | 2 |

The length rule does not beat KEEP on every paper. The logistic selector chooses
threshold 1.0 for Grossman–Stiglitz and Spence, keeping every proposal there,
and 0.35 for Hart–Moore and Modigliani–Miller, merging every proposal there.
These constant held-paper decisions fail to capture the large changes in
paper-level preference rates. All four outer and twelve inner solver fits
converge. Convergence is not evidence of generalization; this negative result
concerns this feature set, regularization and threshold protocol, not every
possible rhetorical mapper.

## Separate tie/drift outcomes

| Method | Selected faithful ties / 9 | Selected claim-shift cases / 8 |
|---|---:|---:|
| KEEP | 0 | 0 |
| MERGE all | 9 | 8 |
| Length only | 2 | 1 |
| Source logistic | 2 | 6 |

The one length-selected shift is judged better separated. Of six logistic-
selected shifts, five favor separation and one is comparable. These selections
are reported as harms/uncertainties outside the primary score, not proof that
source-only features can verify a candidate's semantics.

## Decision and limits

A [replication packet](LENGTH_RULE_REPLICATION.md) is now prepared for four
additional nonreserved papers, using 32 fixed pairs and fresh candidate
generation. No replication outcomes are available yet.

Keep the **30-word heuristic as the next baseline to validate on additional
nonreserved papers**, with the same fixed candidate-generation and verification
procedure. Do not deploy it as a semantic guarantee or tune more classifier
variants until fresh data can test the apparent length effect. The present run
does not justify new Gemma fine-tuning, a neural structural mapper or a claim
that explicit rhetoric has improved distinctive author style.

These are conditional proposal-selection results: all 52 candidates were already
proposed and preferred by GPT-6 in an earlier task. The other 100 boundaries were
not rejudged. New candidate wording could receive different fidelity/readability
judgments at the same source boundary. The primary subset also conditions on a
later faithfulness judgment that will not be known from source alone at deployment.
Side-cohort reporting makes that limitation visible but does not remove it.
The same-model judgments are provisional, paper count is four, and paragraph
pairs are correlated. No independence-based significance or corpus-wide accuracy
claim is made. Whole-paper folds reduce paper/paragraph overlap; they do not
erase earlier project exposure to these papers or annotation/model-family biases.

## Reproduction and checks

Private run: `research_runs/relation_pilot_v1/selector_results_v1/`.
Public [aggregate results](selector_results.json) and
[locked protocol](selector_protocol.json) contain no source paragraphs.
Private features/outcomes, local predictions, fold choices and fitted weights
are pinned in the run manifest. Existing data, judgments, experiments and hashes
remain unchanged. Predictions are not composed into paragraph-wide edits.

CPU entry point: `fama_style.research.verified_merge_selector`, with
`PYTHONPATH=src`, `--verified`, `--sources`, `--protocol` and a fresh `--output`.
Tests alter held-paper labels and side-cohort outcomes to ensure they cannot
change that held paper's fitted model/threshold. They also cover exclusion of
candidate/reference/judge text, source-pair integrity, tied/drift metric separation,
inner/outer paper isolation, conservative tie-breaking and locked-protocol changes.
The original training dry check still passes without downloading or training a
model. This run fits classical classifiers on CPU; it makes no external model
calls and starts no neural/adapter training or GPU jobs. Reserved splits remain
untouched.
