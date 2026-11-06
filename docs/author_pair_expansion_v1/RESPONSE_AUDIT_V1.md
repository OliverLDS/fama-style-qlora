# Superseding direct-pair assessment

The 3 November [completed salvage](../data_scaling_v1/PAIR_SALVAGE_V1.md)
supersedes the provisional direct-text admission decisions below. It retains
229 of the 249 parents after repairs (230 examples), and combines reassessed
legacy pairs into a balanced 200-pair training cohort. The historical counts and
raw audit remain unchanged. No extra manual QA uploads are requested.

# Broader author-pair response audit v1

Completed 3 November 2026. All **54 batches / 249 expected records** were received.
Every record passes the closed output schema, IDs are complete and unique, and
every original paragraph and its hash match the frozen input. Raw responses and
coverage reports are retained unchanged in a private snapshot.

The user reports GPT-6 with observed High thinking effort throughout. All 249 raw
records report a null effort; 167 name GPT-6 and 82 also leave the model null.
The audit records the user's observation separately, without replacing nulls or
claiming backend verification. Chat independence and hidden runtime settings are
not established. No additional provenance form or manual CSV is required.

## Result

| Disposition | Records |
| --- | ---: |
| Received, schema-valid, unchanged originals | 249 |
| Provisional text pairs after conservative checks | 41 |
| Training-side provisional pairs | 34 |
| Development-side provisional pairs | 7 |
| Provisional pairs also passing structural checks | 13 |
| Quarantined annotations/pairs | 208 |
| Independently verified or approved for training | 0 |

| Group | Received | Provisional training | Provisional development | Papers represented in provisional pairs |
| --- | ---: | ---: | ---: | ---: |
| Fama/coauthors | 21 | 3 | 0 | 3 |
| Shiller | 80 | 13 | 3 | 11 |
| Holmström/coauthors | 71 | 7 | 3 | 8 |
| Baker–Wurgler/coauthors | 77 | 11 | 1 | 8 |

These are the **new expansion inputs**, not a recount or replacement of older
Fama/pooled-author datasets. No old dataset is merged, repaired or relabeled here.
The original 45 Fama and 20 pooled-author reserved examples stay sealed.

## Bounded annotation repairs

Nine raw rows have mechanical errors. In five rows from the new-collection
development batch 19, the complete `source_schema` sentence list matches the
original exactly and the complete `target_schema` list matches plain English.
Swapping those two containers is proven by opposite full-text coverage. It clears
their mechanical and program errors without changing either text, any evidence
quote, claim, relation, transformation or raw provenance.

The repaired copies are separate from raw responses. The private repair audit
records row identities, before/after container hashes, the exact rule, and hashes
of unchanged prose, ledgers and transformations. No semantic endpoint or sentence
binding is guessed.

Four rows retain mechanical flags: two have empty relation endpoints, and two
have sentence-coverage mismatches at fused OCR sentence boundaries. The latter
may reflect the check inserting a space between exact sentence entries; the
source is already flagged for transcription uncertainty. They are held rather
than silently correcting the original or loosening the check. Nineteen rows also
retain transformation cardinality or claim-to-sentence program errors. These
counts overlap other hold reasons.

## Why most rows remain on hold

GPT-6 calls 60 originals usable, 184 uncertain and five unusable. It flags 177 for
external context and 200 as needing review. There are 233 proposed simplified
rewrites and 16 pending-review rewrites: producing a rewrite does not establish
that its claims align.

The gate requires usable standalone content, no unresolved review/context flag,
a present rewrite, preserved annotated claims and relations, no unsupported
rewrite claims, exact mechanical evidence, and no unresolved numeric surface
change. Holds include 43 rows with at least one non-preserved/uncertain claim,
25 with a non-preserved/uncertain relation and eleven needing numeric-scope
checks. These are overlapping annotation signals, not independently established
counts of semantic errors.

**Do not interpret the 208 holds as 208 confirmed bad pairs.** The annotator may
be overly cautious about citations, unreported estimates or surrounding discussion.
Conversely, schema validity and exact evidence quotes can miss substantive drift.
The gate deliberately retains uncertainty instead of silently overriding it.

The median token-sequence overlap among the 41 provisional pairs is 0.756
(range 0.597–0.957). This is a lexical descriptor, not proof of useful style contrast,
claim preservation or Fama specificity. No classifier or style metric was selected
using these results. The 13 structural candidates are mechanically consistent
under this audit, not gold structural labels or validated author signatures.

## Private outputs and reproduction

Audit: `research_runs/author_pair_expansion_v1/response_audit_v1/`

- `raw/`: byte-preserved JSONL responses and supplied coverage reports.
- `audits.private.jsonl`: effective annotations, raw/effective audits and hold reasons.
- `repairs.private.jsonl`: the five bounded container swaps.
- `coverage.json`, `session_provenance.json`, `summary.json`, `manifest.json`:
  coverage, separate user-reported settings, aggregates and input/output hashes.

Provisional data: `data/author_pair_expansion_v1/`

- `training.provisional.private.jsonl`: 34 input/target pairs.
- `development.provisional.private.jsonl`: seven input/target pairs.
- `quarantine.private.jsonl`: 208 retained records with reasons.
- `summary.json`: status and limitations.

Each provisional pair retains paper, exact author list, version, PDF hash,
paragraph/line provenance, text hashes and its annotation-file hash. All exported
pairs explicitly set `training_ready: false` and
`independent_semantic_verification: false`. A text pair can pass this provisional
gate while its optional structural program remains unsuitable for training.

```sh
PYTHONPATH=src python3 -m fama_style.research.author_pair_review \
  --release research_runs/author_pair_expansion_v1/release_v1 \
  --output /tmp/author_pair_review_reproduction \
  --dataset-output /tmp/author_pair_data_reproduction
```

Use fresh output paths. Tests cover proven versus ambiguous swaps, unchanged raw
text/ledgers, preservation of null provenance, original-integrity failures,
context/claim/numeric holds, and separation of structural and text-pair eligibility.
No GPU, downloaded model, API request or detector call is used.

## Next decision

Repair **source/context sufficiency before another training run**. Start with a
bounded sample of the held Fama cases, since this release contributes no
provisional Fama development pair. Check alleged transcription defects against
the original PDF and distinguish genuinely missing economic information from
overly cautious annotation. Use neighboring text as read-only annotation context
where appropriate; do not silently add it to targets or claim that unrelated
paragraphs form one paragraph. Preserve paper roles and record every correction
in a new source/annotation version.

Then re-audit meaning alignment and structural bindings before considering wider
repair or admitting a paired release. This recommendation does not launch a new
annotation batch, inference check, classifier fit or training experiment. The
earlier mixed findings and limits on Fama-style claims remain unchanged.
