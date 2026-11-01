# Training coverage: broad concepts present, specific contrasts sparse

Audited all **46 frozen training pairs from 12 papers**, using the four error
families identified in the completed pilot. No development paragraphs, generation
outputs or reserved-test records were deserialized for this audit. Other release
files were hash-verified without parsing their record bodies. No targets,
sources, splits, prompts, prior judgments or training settings were changed.

The central finding is **not a general absence of semantic constraints**.
Quantifiers, time relations, conditions and explanations already appear in many
training pairs. Some exact failure subtypes are absent, however, and most pairs
preserve paragraph claim order while making local wording/clause changes.
These observations suggest targeted supervision, not more undirected epochs.
They do not identify the cause of the observed model errors.

## Content coverage

| Audited family | Pairs | Papers |
| --- | ---: | ---: |
| Quantified scope or ordered comparison | 42/46 | 12 |
| Temporal ordering, aggregation or theoretical/observed distinction | 30/46 | 11 |
| Reported or hypothetical numerical content | 13/46 | 5 |
| Conditions, modality, supplied reasons or limits on inference | 44/46 | 12 |

Families overlap; counts must not be summed. Labels describe constraints visible
in both archived texts. They are qualitative content annotations, not a new
rhetorical taxonomy, model features or independently verified entailment labels.
Citation years, section/table indices and dates alone do not count as numerical
economics. Document roadmap order and vague “recently” language do not count as
an explicit economic ordering constraint. Number words are considered as well
as digits, and hypothetical values are separate from reported figures.

| Specific signal | Pairs | Papers |
| --- | ---: | ---: |
| Scope, frequency, universality or negation | 40 | 12 |
| Ordered comparison or ranking | 31 | 12 |
| Economic event order or boundary | 23 | 9 |
| Aggregation, duration or horizon | 11 | 6 |
| Theoretical/predictive versus observed/realized quantity | 6 | 3 |
| Explicit planned-versus-subsequent investment contrast | **0** | 0 |
| Substantive reported numerical values | 11 | 4 |
| Linked numerical comparisons | 10 | 4 |
| Hypothetical numerical alternatives/counterfactual values | 2 | 2 |
| Numerical probability–outcome choice/indifference tuples | **0** | 0 |
| Stated condition, assumption, threshold or exception | 33 | 11 |
| Qualified modality or epistemic force | 36 | 12 |
| Supplied reason or mechanism | 36 | 12 |
| Explicit limits on association, evidence or predictability | 7 | 5 |

The dataset contains substantial numeric comparisons, including amounts,
percentages and investment horizons. It does not contain the particular
probability/payoff tuple structure that failed in development. Likewise, event
ordering is common and some theoretical-versus-observed distinctions are present;
an explicit planned-versus-subsequent investment contrast is absent. These exact
gaps do not imply that the model cannot generalize or that copying development
topics into training is the remedy.

The expansion increased reported-numeric coverage from **3/16** original pairs
to **11/46** total pairs (eight of the 30 additions). Those 11 come from four
papers; the largest paper contributes 4/11. Temporal-family coverage rose from
8/16 to 30/46. More data therefore added relevant content, even though the
completed pilot did not improve the coarse material-error count.

All 46 payloads remain ordinary positive source/target SFT examples with the
same explicit preservation instruction. They have **no chosen/rejected examples
or per-example semantic-error labels**. Positive-only SFT can teach preservation;
this audit does not establish that preference training would work better.
The 36 supplied explanations are positive examples of bounded reasoning, not
contrasts demonstrating why an unsupported explanation should be rejected.

## Rewriting signal

| Descriptive measurement | Result |
| --- | ---: |
| Median word-sequence overlap | 0.777 |
| Overlap range | 0.437–0.927 |
| Pairs with overlap at least 0.85 | 7/46 |
| Exact word-token copies | 0/46 |
| Sentence-count proxy changes | 23/46 |
| Major paragraph claim-order changes identified | 0/46 |
| Median target/source word-length ratio | 1.045 |

There is lexical contrast and frequent sentence regrouping; the data are not
all copies. The claim-order judgment distinguishes major proposition ordering
from clause fronting, embedding, merging and punctuation changes. Such local
operations can carry useful style, but they provide little direct evidence for
learning a substantially different paragraph argument plan. Targets are typically
slightly longer than inputs, so the data also do not express a general compression
objective. Neither overlap nor sentence counts measure semantic fidelity or a
distinctive Fama signature. These are mixed-author academic paragraphs.

## Alignment and source limits

The text-bound comparison identified no new definite substantive pair-alignment
defect. Forty-two pairs are marked aligned to archived text; one retains a
qualified boundary interpretation; three retain source-wording uncertainty:

- A life-cycle-stage phrase is slightly looser in the input than in the target.
- An unusual share-sale phrase remains an uncertified transcription.
- A multiplicative “times higher” formulation is ambiguous on both sides.
- Rounded/table-derived earnings amounts and percentage figures are identical
  on both sides but have not been independently reconciled.

These are recorded limitations, not silent repairs or new automatic exclusions.
All earlier curation decisions and flags remain attached to the private audit;
30 selected pairs retain their original needs-review flag. Zero numerical-value
set differences were found by the surface screen, but shared ambiguity,
changed roles or identical transcription errors can escape that screen.
No new PDF/OCR certification or semantic gold status is claimed.

## Recommended bounded preparation

Before another training run, prepare **12 training-only anchor cases, three per
error family**, using existing archived targets and explicit bindings for the
actors, values, periods, quantifiers and allowed reasoning. Exclude unresolved
wording from new perturbation exercises. Keep development and reserves fixed.

For each anchor, define a minimal counterexample that changes exactly one
supported relationship—such as scope strength, time order, value-to-group
assignment or an invented explanation—and a faithful but stylistically different
input variant. Independently check those constructions against the archived
target before treating them as training material. Preserve provenance and mark
the synthetic variants/negative examples as such. Do not insert development
generations or references into these records.

That preparation can support a subsequent comparison of stronger faithful SFT
inputs against explicit semantic contrasts. It should first establish that the
proposed examples isolate the intended distinction, rather than automatically
launch a preference optimizer or another epoch extension. A later training
proposal must specify its objective, matched optimization budget and checks
against trading expression gains for unacceptable meaning changes. This audit
does not choose a training method or authorize new GPU execution.

## Reproduction and privacy

Annotations are automated, post-pilot judgments by the same project reviewer,
aware of the prior failure categories. Counts are descriptive and not an
independent causal explanation. Feature definitions are versioned in
[coverage_specification.json](coverage_specification.json); public aggregates
are in [training_coverage.json](training_coverage.json).

Per-pair evidence, source/target hashes, all original flags, notes, provenance
and receipts stay private under
`research_runs/teacher_pair_reverse_pilot_v2/training_coverage_v1/`.
No external LLM calls, paraphrases, negative training examples, dataset repairs,
model weights, detector queries, GPU jobs or training are produced here.

```sh
PYTHONPATH=src python3 -m fama_style.research.teacher_pair_coverage \
  --curated data/teacher_pair_pilot_v2 \
  --annotations research_runs/teacher_pair_reverse_pilot_v2/training_coverage_v1/annotations.private.jsonl \
  --output /tmp/teacher-pair-coverage-reproduction
```

The builder requires all 46 text-bound annotations, validates the positive-pair
schema and labels, reads only selected training-audit bodies and refuses
overwrite. Synthetic CPU tests include an unreadable development file and
unmatched audit body to verify that neither is parsed. The original model,
notebooks, data and experiment hashes remain unchanged.
