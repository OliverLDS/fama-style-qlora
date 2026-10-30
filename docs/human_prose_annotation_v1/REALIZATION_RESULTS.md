# Human-prose oracle realization: mixed findings, little incremental guidance benefit

All 16 expected responses were received in the two frozen batches. The user
reports GPT-6 with High thinking effort. This is an eight-example, four-paper
**development-only** diagnostic of non-Fama economics prose. Both conditions
receive the same neutral input, source-quoted claim evidence and ordinary rewrite
instruction. One additionally receives reference-informed sentence claim groups
and finite presentation controls, without original wording in the prepared
payload. It is an oracle diagnostic, not a deployable source-only mapping test.

## Results

| Measurement | Ordinary rewrite | Oracle presentation guidance |
|---|---:|---:|
| Outputs | 8 | 8 |
| Normalized input copies | 0 | 0 |
| Median source lexical overlap | 0.771 | 0.723 |
| Median reference lexical overlap, evaluation only | 0.664 | 0.666 |
| Reference sentence count matched | 5/8 | 5/8 |
| Reference claim grouping matched, assistant assessment | 5/8 | 5/8 |

Guidance lowers source overlap in only four of eight paired examples, and raises
reference overlap in three. Median paired output overlap is 0.814. All eight
pairs use the same number of output sentences; seven also use identical claim
groups. Lower overlap measures wording changes, not improved style. Reference
similarity is not proof of fidelity, human authorship or author-specific style.
The near-equal medians do not establish statistical equivalence.

Only **two of the eight** input/reference pairs have different sentence claim
groupings. In one, ordinary and guided generation both recover the intended
three-to-two sentence merge. In the other, both compress to four sentences rather
than the reference's six. The source already has the target grouping in the other
six cases. Thus this sample has limited leverage for detecting benefits from
structural transformation. Guidance was explicitly optional, so failure to match
its groups is not by itself evidence that the model cannot realize a plan.

## Claim preservation and expression

An in-session coding-assistant review read all outputs against the supplied
neutral inputs and original references, binding each of 110 claim instances to
exact output evidence. This was unblinded and is **not independent semantic
verification**. The conservative label “no clear change identified” means only
that no additional change was noticed, not that equivalence was established.

| Evidence-review label | Ordinary, 55 claim instances | Guided, 55 claim instances |
|---|---:|---:|
| No clear additional change identified | 49 | 49 |
| Minor qualification, degree or inferential-force shift | 5 | 4 |
| Uncertain scope risk | 1 | 1 |
| Clear actor-scope broadening | 0 | 1 |

These counts are descriptive judgments, not calibrated semantic accuracy scores,
independent observations or automatic reject thresholds. A modest change could be
acceptable if accompanied by a useful expressive benefit. Here the qualitative
review finds no consistent incremental organization or expression benefit to
justify claiming a successful style/meaning tradeoff.

The clearest guided issue replaces **lenders** with the broader **providers of
capital**, which can include equity investors. Another guided output replaces
explicit expected efficiency with capacity, and probability of success with
feasibility; the latter is recorded as an uncertain risk rather than a definite
error. Both conditions weaken an explicit implication to a suggestion in one
example, and both lose minor frequency/quantity qualifiers in others.

Guidance does help one local fidelity detail: it retains qualified attribution
that ordinary rewriting states more assertively, and keeps a broader environment
reference where the ordinary output narrows it to markets. This supports a local
observation, not a general stylistic win. Several other pairs differ mainly in
connectors, adjectives or near-equivalent wording. Currency formatting from
52 thousand to 52,000 preserves the quantity and is not counted as claim drift.

One neutral input was already known to weaken “probably” to “possible.” That
prior input/reference mismatch remains separately recorded; it is not charged
to the realizer. Generated claims are first compared with the supplied input,
with reference alignment treated as a separate question.

## Interpretation and next decision

Ordinary rewriting already produces varied, compact academic prose in this
small GPT-6 test. The optional oracle guidance does not demonstrate a consistent
additional benefit. Neither result tests Gemma, QLoRA, detector performance or
distinctive Fama/coauthor style. It also does not show that a structural mapper is
useless: most plans here preserve existing grouping and adherence was optional.
Do not train a mapper or scale fine-tuning on the basis of these results.

The next informative diagnostic would use the **five unused development pairs
whose source and target grouping actually differ**, with the same model and
shared semantic input. Compare ordinary rewriting, an explicit source-group
control and an explicit target-group plan, with an actionable request for one
sentence per claim group unless meaning would change. This isolates whether a
nonidentity plan changes realization beyond merely supplying a plan. Keep the
same claim-drift tolerance, score grouping and expression separately, and report
any meaning-related refusal to follow a plan. The follow-up has now [completed](GROUPING_RESULTS.md),
with limited grouping-control evidence. It remains development work; no reserved content is
needed. If guidance still supplies no useful change, retain ordinary rewriting
as the baseline and prioritize pair quality before further structural learning.

The 60 proposed training pairs remain annotations, not certified training data.
The separate 247-pair Fama training dataset and all prior experiments are unchanged.

## Provenance, privacy and reproduction

Raw responses are preserved byte-for-byte under the private run's `review_v1/raw/`.
The manifest pins package, audit, assessment, response and code hashes. Evidence,
per-output assessments and pair observations remain private; this report and
[aggregate results](realization_results.json) contain no reference paragraphs.
User-reported model/effort are recorded without inventing a hidden snapshot,
seed, temperature or current conversation arrangement. Prior cross-chat retrieval
concerns remain; original wording withheld by our payload does not establish
absence of contextual retrieval or pretraining exposure. Same-family annotation
and realization are not independent; there are no repeated generations or
controlled decoding estimates.

```sh
PYTHONPATH=src python3 -m fama_style.research.human_prose_realization_review --package research_runs/human_prose_realization_v1/package_v1 --responses research_runs/human_prose_realization_v1/package_v1/responses --audit research_runs/human_prose_annotation_v1/audit_v2 --assessments research_runs/human_prose_realization_v1/assessments_v1.private.json --output /tmp/human_prose_realization_review
python3 -m unittest discover -s tests -v
```

Tests cover complete/missing/duplicate/wrong-batch response transport, immutable
package keys, malformed output, exact assessment evidence, full claim coverage
and sentence-group evidence bindings. Those checks establish consistency, not
the truth of linguistic judgments. Import/review used no external model calls,
weights, credentials, detector calls, training or GPU jobs. All 20 reserved
responses remain outside development content analysis.
