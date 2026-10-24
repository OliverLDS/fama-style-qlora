# Audit of the evidence-based v2 consolidation

Both `taxonomy_v0_1_proposal_v2.json` and `consolidation_report_v2.md` were checked
against the exact 200-item inventory from response audit v2. Unlike the earlier
design scaffold, this response contains observed-label mappings and supporting
inventory IDs. It remains `proposed_not_frozen`.

## Checks and results

| Item | Independently recomputed result |
|---|---:|
| Inventory IDs covered exactly once | 200 / 200 |
| Proposed canonical categories | 132 |
| Paragraph / sentence / language categories | 74 / 35 / 23 |
| Unambiguous one-to-one inventory mappings | 158 |
| Inventory mappings needing full context | 42 |
| Mechanically expandable occurrences | 1,788 |
| Occurrences retaining candidate categories | 720 |
| Descriptive / candidate control / protected categories | 21 / 32 / 79 |

No duplicate canonical or mapping IDs, missing inventory IDs, unknown canonical
or related references, cross-level/dimension mappings, or support-inversion
mismatches were found. Split mappings contain alternatives and disambiguation
rules. Candidate presentation controls specify preconditions and prohibited
effects. Recomputed counts match the machine-readable consistency report.

The support lists include *candidate* links from ambiguous mappings. They must
not be treated as confirmed occurrence assignments. Of 132 categories, 104 have
at least one unambiguous source mapping, including diagnostic B; 95 have at least
one such mapping from the original or proposed-neutral primary contexts. These
counts do not establish independent semantic validity or adequate sample size.

## Derived files and uncertainty

The private audit directory is
`research_runs/taxonomy_discovery_v1/consolidation_audit_v2/`.
It contains the summary, input/output hashes and two derived occurrence files:

- `canonical_occurrences.private.jsonl`: all 2,508 occurrences with the proposed
  vocabulary mapping attached. Source annotations are not rewritten.
- `needs_context.private.jsonl`: the 720 ambiguous occurrences. Their
  `canonical_id` remains null; alternatives and disambiguation rules are retained.

The original taxonomy, report, annotations and neutral proposals remain unchanged.
Mechanical mapping does not establish semantic correctness, annotation agreement,
or a distinctive author-style signal. The 84 annotations still represent only
28 underlying training pairs. Proposed neutral repairs remain unaccepted.

## Recommendation

This is a useful provisional vocabulary, but it should not yet be frozen or used
to train a structural mapper. Its 132 categories are granular relative to the
discovery sample, and 42 input label definitions still need occurrence context.

Start with the ambiguities relevant to presentation controls: 13 split inventory
rows, covering 320 occurrences, have candidate-control alternatives. Examine
representative full training contexts, preserving semantic bindings and uncertainty;
do not assign all alternatives or assume that all instances of a broad label mean
the same thing. Then check the proposed vocabulary on a small fresh training-only
batch through independent annotation. Rare/protected categories can remain
provisional rather than being forced into a deployable control set.

Three synthetic tests pass for mapping coverage, invalid references, required
guards and nonassignment of ambiguous alternatives. No model calls, GPU jobs,
dataset repairs, or held-out development/test evaluations were performed.
