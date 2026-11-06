# GPT-6 matched 500: response quality audit

Subsequent finding: this meaning/format audit does not establish sufficient style
contrast. The [contrast reconstruction branch](GPT6_CONTRAST_RECONSTRUCTION_V1.md)
reviews all 500 inputs and prepares brief-first replacement versions for weak
rewrites, including historical reuses. It supersedes the earlier no-recollection
advice where additional rewriting is justified by the user's style priority; raw
responses and this audit remain unchanged as evidence.

All eight manual files were returned and imported: **227 of 227 expected records**,
with correct IDs, target hashes, per-batch order, and five-field schemas. No records
were omitted or duplicated. Together with 273 unchanged historical reuses, all
500 matched candidate pairs are present. No training-ready release or GPU run is
created by this audit.

| Automated disposition of new responses | Records |
|---|---:|
| No material issue identified | 212 |
| Retained with minimal, versioned input repairs | 10 |
| Retained pending inherited target cleanup | 5 |
| Total retained | 227 |

This is an automated assessment, not an independent human or model-based semantic
certification. The eight batches do **not** need to be recollected. The five target
follow-ups are individual curation issues; they do not invalidate the remaining
responses. All raw files and frozen human targets remain unchanged.

## Meaning and contrast

The audit checks every new record's identity, field types, nonempty input, numeric
surfaces, input length, copying, and correspondence to the frozen target. Automated
changed-span inspection covers the new pairs. All 44 cases flagged by numeric
differences, substantial shortening, or the generator's uncertainty received full
pair inspection, along with additional follow-ups. The reviewer is Codex; no API,
independent NLI model, detector, or human review was used.

There are 26 numeric-surface flags and 21 shortening flags, with overlap. Many
are harmless: percent versus `%`, digits versus words, consolidated duplicate
citations, anaphoric references, expanded year ranges, or squared notation. They
are assessed in context rather than rejected by a token-matching gate.

Ten inputs receive 14 small exact-substring repairs, preserving all other neutral
wording. These restore modality or scope, a missing novelty claim, an exclusivity
claim, attribution, literal actors, and commodity specificity; one removes an
unsupported averaging-window direction. The clearest economic error is **P340**:
the rewrite makes stock demand more price-sensitive where the reference describes
a more steeply downward-sloping demand curve. The repair retains the reference's
relationship between arbitrage barriers and sentiment-driven price movements.

Raw neutral inputs have median token-sequence similarity **0.513** to their human
targets, median word-length ratio **0.858**, zero exact normalized target copies,
and 227 distinct normalized inputs. This supports useful surface contrast. It does
not certify semantic equivalence, prove human-style quality, or establish a
detector outcome. No minimum edit-distance target is imposed.

## Five inherited target issues

| ID | Finding | Disposition |
|---|---|---|
| P250 | Milwaukee's mean is reported as both 6.8% and 5%; archived PDF native text repeats the contradiction. | Preserve uncertainty; do not silently infer that the second statistic is a median. |
| P264 | Target begins mid-sentence on page 17; page 16 contains the preceding text specifying corporate tax rates. | Resolve the cross-page paragraph boundary consistently. |
| P291 | BIS Working Papers publisher boilerplate entered the assigned author's narrative pool. | Correct target selection; the GPT-6 rewrite itself is faithful. |
| P458 | A standalone Conclusion heading is attached to prose. | Move heading to metadata in a versioned cleanup. |
| P484 | A Retail Investor Trades subsection label is attached to prose. | Separate heading metadata consistently. |

These issues are present in the frozen mixed targets and therefore affect both
matched teacher tracks. They are separate from the earlier duplicate-counting
audit. None is a reason to regenerate all eight batches. Source evidence uses
archived native PDF extraction with original PDF hash checks, not a new independent
visual verification. P379 also retains the published target's literal actors
rather than guessing what the author intended.

Resolve these shared target issues before freezing final teacher-specific training
releases. The immutable mixed release and earlier experiments remain historical.
Any target correction must be versioned and applied consistently to both tracks;
do not quietly alter just one track or relabel Codex-written replacements as GPT-6.

## Artifacts and checks

Private audit folder:
`research_runs/data_scaling_v1/gpt6_track_0500_v1/response_quality_v1/`.
It contains response hashes, lexical screens, text-bound adjudications, raw/final
input hashes, exact repair anchors, source evidence, and all 500 candidate pairs.
The first 200 candidates are an exact byte prefix of 500; all targets and historical
reuse inputs remain unchanged. Candidate records explicitly set `training_ready`
to false. Requested current settings are GPT-6/High; the UI's actual per-batch
effort is not independently observable.

Run `python3 scripts/data/audit_gpt6_track_500.py check` to verify the import,
prepared source bindings, audit hashes, and nested candidates. CPU tests cover
wrong identities/hashes/fields, missing text, uncertain statuses, local patch replay,
nonunique/whole-paragraph replacement rejection, immutable targets, and historical
reuse preservation. All copyrighted prose and generated response files stay private.
