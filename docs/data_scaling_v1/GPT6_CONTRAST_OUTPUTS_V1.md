# Brief-first reconstruction: much stronger input contrast

All six Stage 2 files returned **175/175 expected records**, with correct IDs,
target hashes, order and output fields. The new paragraphs show a substantial
change from the earlier close-paraphrase inputs. P002 no longer inherits the
hopes framing; P024 rebuilds the method explanation while retaining its CRSP codes.
All originals and raw responses are preserved.

| Automated disposition | Records |
|---|---:|
| Retain new paragraph as returned | 163 |
| Retain with local content/cleanup repairs | 10 |
| Carry inherited source concern; retain earlier input in staging | 2 |
| Total new versions preserved | 175 |

**173 new inputs** enter a versioned 500-pair candidate set. Its other 327 inputs
are unchanged: 320 outside the replacement queue, five inherited target-cleanup
cases and the two unresolved source concerns. No new independent targets are
added, and no version is discarded. This is a data-quality branch of the same
500-target cohort, not an increase in training-set size.

## Re-expression

Metrics compare each new raw paragraph and its earlier input against the same
human target, using the frozen contrast procedures:

| Median across the 175 selected targets | Earlier input | New input |
|---|---:|---:|
| Word-sequence similarity | 0.756 | 0.163 |
| Ordinary-word shared five-gram coverage | 0.579 | 0.000 |
| Longest aligned ordinary-word span | 15 words | 3 words |
| Input/target word-count ratio | 0.941 | 0.775 |

All 175 have lower word-sequence similarity than their predecessors. None of the
assessed final paragraphs triggers the existing high-copy triage rules. Inspection
also finds fresh sentence construction and regrouping in examples such as P002
and P024, rather than just connective substitutions.

These are surface diagnostics, with heuristic exemptions for fixed material.
They are not validated author-style, grammatical-structure or semantic metrics.
Lower overlap alone is not quality; increased concision can also omit details.
The result supports a stronger rewriting signal in the data. Student-model gains,
Fama-style gains and Pangram improvements have not yet been measured.

## Content assessment and repairs

Codex read the new paragraphs and checked potentially material problems against
the references. This is automated assessment, not independent human/NLI validation
or exhaustive claim-by-claim equivalence. Compression, reordered explanations,
interpretive additions and modest emphasis shifts are accepted under the user's
style-first priority. The workflow does not restore the author's phrasing merely
to make the input closer to the target.

Thirteen exact local patches affect ten paragraphs. They touch 171 of 16,088 raw
source words, approximately 1.1%, leaving the fresh constructions largely intact.
The cases are:

- **P025:** distinguish an all-sample average from the years in which negative-equity
  cases concentrate.
- **P151:** retain repayment contingent on project success and attach a dilution
  threshold to the right argument.
- **P168:** retain the stated order-of-magnitude contrast.
- **P173:** distinguish a return threshold from its significance lower bound.
- **P198:** clarify the inverse shoreline/air-conditioning relationship and the
  resulting downward spurious trend.
- **P223:** correct the interpretation of Federated's $5.85 billion as
  benchmark-adjusted value returned, rather than a hypothetical benchmark value.
- **P319:** assign the reported governance-premium finding to the appropriate survey.
- **P086, P169 and P282:** remove dangling record references or workflow commentary.

P134 and P139 remain source-concern versions. Their output paragraphs discuss
chronological or model ambiguity; that commentary is not promoted as a clean
counterpart to an assertive target. The earlier inputs remain in staging, with
the concern recorded. Alongside P250, P264, P291, P458 and P484, there are seven
source/target follow-ups before a final training-ready release. The existing
mixed release and previous experiments remain historical and unchanged.

## Review and reproduction

Reload the [local pair viewer](LOCAL_PAIR_REVIEW.md). The new version is labeled
**Candidate · fresh brief reconstruction**, beside the prior GPT-6 and Gemini
variants for the same human targets. Identical retained inputs merge their dataset
provenance; the view contains 500 training targets and 1,173 distinct input
variants, including 173 fresh alternatives. Feedback IDs for old pairs remain valid.

Private artifacts are under
`research_runs/data_scaling_v1/gpt6_track_0500_v1/contrast_rewrite_v1/rewrite_quality_v1/`:
text-bound decisions, exact patches, before/after metrics, versioned 500 candidates
and an exact 200-record byte prefix. Every candidate has `training_ready=false`.
The latest candidate catalog is separate from all frozen datasets. Local content
repairs retain Codex authorship rather than being labeled untouched GPT-6 output.

Check with `python3 scripts/data/review_gpt6_contrast_outputs.py check`.
CPU tests verify hashes, coverage, repair replay, unchanged targets, explicit
source holds and the generic viewer adapter. No provider calls, detector scans,
development/test text reads or training jobs are performed by this review.
