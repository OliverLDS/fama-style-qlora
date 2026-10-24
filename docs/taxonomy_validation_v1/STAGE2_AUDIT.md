# Stage 2: fresh canonical annotation and masked-repeat audit

All 16 returned annotations match the prepared files: 14 distinct paragraphs
from seven training pairs, plus two masked repeats. These pairs were excluded
from taxonomy discovery; they remain training data, not development/test data.

## Mechanical checks

JSON Schema, version/hash, ID coverage, sentence/clause coverage, evidence quotes,
local semantic references, canonical level/dimension restrictions and gap/label
cross-links pass. Source responses are unchanged. No repairs, models or GPU jobs
were run, and semantic correctness was not independently verified.

| Batch | Paragraphs | Assigned labels | Unresolved | Not covered |
|---|---:|---:|---:|---:|
| A | 7 | 345 | 1 | 2 |
| B | 7 | 303 | 1 | 0 |
| C: repeats | 2 | 79 | 0 | 1 |

The 14 primary paragraphs have 648 assigned, 2 unresolved and 2 not-covered
label objects, using 81 of the 132 proposed categories. These are counts of
selected annotation objects, not semantic accuracy, completeness estimates or
independent validation of all 132 categories.

## Reported coverage distinctions

Two primary gap instances concern comparative probability expressions with an
implicit baseline: the current comparison-marker definition requires an explicit
comparator. The repeated annotation identifies one additional exemplification
marker distinction at language level, despite an existing paragraph-level
illustrative-move category. The first annotation of that same text did not report
the gap. These are candidate vocabulary clarifications, not automatically accepted
new categories. The catalogue and all original labels remain unchanged.

The two unresolved primary assignments concern passive versus stative/resultative
readings. Abstention is preserved rather than forcing a category.

## What the repeat check actually shows

Comparisons use the private repeat key and identical text hashes. Claim IDs and
list positions are not treated as shared units across independently generated
annotations. Two complementary descriptive comparisons are reported:

| Measure | Repeat 1 | Repeat 2 |
|---|---:|---:|
| Category-set Jaccard overlap | 48.1% | 69.2% |
| Base evidence/dimension groups | 23 | 29 |
| Repeat evidence/dimension groups | 31 | 49 |
| Groups with identical anchors and dimension in both | 11 | 14 |
| Those shared groups with identical category sets | 11/11 | 14/14 |
| Base-only groups | 12 | 15 |
| Repeat-only groups | 20 | 35 |
| Semantic relation count, base → repeat | 0 → 1 | 2 → 4 |

An anchor group is `(level, dimension, exact evidence-span set)`. Clause spans
are matched inside their exact sentence only when uniquely resolvable; no
unanchored clause labels occurred here. Jaccard compares selected category sets,
not classification accuracy. Different span extents count as different units.

The conditional agreement on 25 shared groups is useful but does not establish
overall reliability: most evidence groups are not shared. The evidence points
to variable selection, exhaustiveness and segmentation, rather than demonstrated
category disagreement on those particular shared groups. Relation-count changes
likewise show differences in the recorded graph, not proof that one graph is
correct. Two texts cannot establish general annotation reliability.

## Decision and next check

Do not freeze the vocabulary, infer style differences from all annotation-count
differences, or train a structural mapper yet. The proposed meanings may be usable,
but a repeatable measurement procedure is still needed.

The next small check should reuse ONLY the two repeat texts:

1. Provide fixed sentence/clause evidence units, with stable IDs and explicit
   uncertainty; do not let each annotation redefine its comparison units.
2. Require a short, identical feature checklist per applicable unit, with explicit
   present/absent/unresolved/not-applicable states. The current positive-label
   lists do not make omission distinguishable from absence.
3. Keep separate fields for grammatical scale and rhetorical function. Record the
   two candidate vocabulary clarifications without silently relabeling prior data.
4. Compare decisions on those fixed units before annotating more papers. Retain
   independent semantic checks; syntactic consistency is not claim preservation.

This targets selection and granularity before adding annotation volume. If the
small check becomes consistent, return to the oracle realization experiment to
test whether the structural representation actually improves rewriting.

## Evidence and reproducibility

Private results are in `research_runs/taxonomy_discovery_v1/stage2_audit_v1/`:
`audit_summary.json`, `canonical_label_records.private.jsonl` and `manifest.json`.
The manifest records unchanged input responses and source-packet hashes. Four
synthetic tests distinguish changed evidence selection from a changed category
on the same anchor, validate scope checking, and exercise exact quote occurrences.
