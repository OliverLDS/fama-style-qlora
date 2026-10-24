# Stage 1 context-resolution audit

All six response batches and reports were received. The 39 cases match the
prepared IDs, taxonomy version/hash and per-case candidate sets.

| Reported outcome | Cases |
|---|---:|
| Single category proposed | 30 |
| Distinct components proposed | 4 |
| Supplied candidates do not fit | 3 |
| Ambiguity remains | 2 |

JSON Schema, exact evidence anchors, quote occurrences, case/context identity,
existing claim/relation references and component-duplicate checks pass. These are
mechanical checks, not independent confirmation of the model's interpretations.
All response files, previous annotations and global mappings remain unchanged.

## What the remaining cases imply

The three `taxonomy_gap` outcomes are failures of the *restricted candidate sets*:
an ordering marker had been placed under concision; phrase-internal coordination
had been treated as discourse linkage; and an explanatory critique did not fit
contrast/caveat alternatives. The full catalogue already contains sequencing,
grammatical coordination, and explanatory-premise/assumption-caveat categories.
These are possible avenues for reannotation, not new assignments or proof that
each case has a unique answer. The supplied gap decisions are retained.

The two unresolved cases concern specification versus evidential support with
inadequate relation bindings, and an ambiguous participial/passive/adjectival
reading. They stay unresolved. Decisions on the sample do not resolve entire
inventory groups or the 681 unsampled ambiguous occurrences.

The responses also report possible errors in the old semantic annotations:
word-form/function confusion, dates and quantities bound too broadly, incomplete
claim/clause segmentation, and relation IDs pointing to unrelated propositions.
No semantic graph was silently repaired or accepted for training.

## Stage 2 decision

Proceed with the existing 132-category proposal as a **coverage and consistency
test**, not a frozen standard. There is not yet sufficient evidence to introduce
new canonical IDs. Procedural clarifications now emphasize context over string
matching, grammatical scale, local quantity bindings, and relation endpoints.
They contain no Stage 1 case texts or expected labels.

The updated Stage 2 package is at
`research_runs/taxonomy_discovery_v1/stage1_audit_v1/stage2_ready/`.
It preserves the original seven fresh training pairs, two masked repeats, full
catalogue and output schemas byte-for-byte. Only the instruction text changes.
Use the new `STAGE2_START_HERE.md` beside it for upload and response filenames.

The private audit summary, checked proposals, pending cases and input/output hashes
are in `stage1_audit_v1/`. No new model calls, accepted label repairs, GPU jobs,
or development-validation/reserved-test evaluations were performed. Three synthetic
tests exercise invalid case bindings, quote/reference failures and duplicate
component assignments.
