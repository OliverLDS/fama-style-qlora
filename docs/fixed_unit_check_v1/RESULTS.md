# Fixed-unit consistency observations

Both response files contain two complete paragraphs and 55 expected cells.
JSON Schema, actual input IDs, fixed unit geometry, feature/canonical IDs, quote
anchors within unit bounds, claim/relation references and unit-assessment coverage
pass. The source responses were not changed or silently relabeled.

## Session and input provenance

Both files use the prepared `round_2` paragraph/cell IDs. The user confirmed they
were produced in separate new chats without explicitly showing one response to
the other. This supports a record of two separately requested annotations of the
same inputs, not verified information isolation or statistical independence.
No test of memory carryover was performed. These are practical ChatGPT
repeatability observations, with context isolation unverified.

The original distinct-ID round design was not completed. No annotation is discarded
for that metadata issue, and no rerun solely for memory isolation is required.

## Recomputed results

| Measure | Result |
|---|---:|
| Matching states | 51/55, 92.7% |
| Both responses gave present/absent | 53 cells |
| Matching present/absent decisions | 51/53, 96.2% |
| Both marked present | 18 cells |
| Both marked absent | 33 cells |
| Absent → present differences | 2 cells |
| Present → unresolved differences | 2 cells |
| Both units usable and both states informative | 33/35 matching |
| Unit-suitability disagreements | 3 units |

The 55 cells describe only two paragraphs. Correlated features, a small sample,
and unverified session isolation limit interpretation. This is consistency of
outputs, not semantic accuracy or evidence of successful style transfer. The
percentages are not numerically comparable to the earlier optional-label Jaccard
scores, because the measurement representation changed.

## Four disagreements, three boundaries

- **F03 / sequencing:** whether a temporal adverb announcing discussion counts as
  analytic discourse progression rather than merely timing.
- **F04 / comparison:** whether method identity across explicitly listed horizons
  meets the comparison category's explicit-comparator requirement.
- **F12 / specification, two cells:** both responses recognize plausible textual
  elaboration, but differ on whether missing formal relation bindings require an
  unresolved state. This exposes a protocol ambiguity between observing
  presentational structure and verifying semantic bindings/action permissions.

The record retains both proposals without adjudicating which is correct.
Definitions, old semantic graphs and the original 720 ambiguous occurrences remain
unchanged. Explicit evidence and uncertainty still govern any future use.

## Practical decision

The mandatory matrix provides a usable measurement format for a small next
experiment. Keep it provisional, specify the F03/F04 boundary rules, and distinguish
observational presentation judgments from permission to execute a transformation.
There is no need for another broad taxonomy-discovery round or a memory-isolation
exercise on the basis of these files.

Return to a small Gemma realization check with concise, fixed plans on paired
training examples. Compare claim preservation, actual plan adherence and style
separately before training a structural mapper. Agreement on annotations alone
does not establish that Gemma can realize them. No new GPU job is submitted here.

## Artifacts

The final private audit is in
`research_runs/taxonomy_discovery_v1/fixed_unit_audit_final/`: summary, per-feature
counts, disagreements and input/output hashes. Session provenance is explicitly
user-reported; isolation and independence remain unverified. Three synthetic tests
cover input routing, weighted aggregation and exclusion of uncertain-only agreement.
No model calls, semantic repairs, held-out evaluation or training were performed.
