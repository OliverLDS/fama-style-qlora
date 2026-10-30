# Fixed-candidate verification: mixed organization results

All thirteen files cover **52/52 fixed candidates** and pass schema, exact-text,
evidence-occurrence and internal-consistency checks. No candidate, source text,
unit alignment, label or raw provenance was repaired. User-reported settings are
**GPT-6, High thinking effort**. All raw model/effort fields remain null, with the
user confirmation recorded separately. This does not retroactively resolve
Stage 2's earlier effective High/Medium uncertainty.

## Returned judgments

| Fidelity judgment | Merge reads better | Separation reads better | Comparable | Total |
|---|---:|---:|---:|---:|
| Faithful | 18 | 17 | 9 | 44 |
| Local shift | 0 | 7 | 1 | 8 |
| Material shift / unresolved | 0 | 0 | 0 | 0 |
| Total | 18 | 24 | 10 | 52 |

These are automated second-assessment judgments, **not semantic certificates**.
All candidates had previously been selected as MERGE-preferred by GPT-6.
The verifier reconfirms a readability advantage for only **18/52 (34.6%)**;
24 are now judged better separated and ten comparable. This exposes substantial
preference instability across the two annotation tasks. It is not a randomized
repeatability test: the prompts, task, candidate visibility and reported effort
differ. The original Stage 2 labels remain unchanged.

Every candidate judged to improve readability is also judged faithful. The eight
local-shift cases add **no** readability improvements in this selected sample.
Thus a relaxed drift budget would not rescue those candidates here. This does
not establish that all style transfer avoids a fidelity tradeoff; this check
concerns local organization of neutral economics prose, not distinctive author
style or a fine-tuned Gemma model.

The eight local shifts concern added explanatory/inferential relations,
substituted access mechanisms, an entitlement-versus-share distinction and a
less explicit qualification. Unit annotations contain 175 preserved, two omitted
and three changed units; some relation shifts appear in added-content fields
rather than as changed units. Matching unit words alone would miss those effects.
Zero material-shift verdicts is a property of this selected sample and judge,
not a validated safety rate.

## What appears useful

Coding-assistant reading of the returned rationales finds useful merges often
connect a short dependent explanation, definition, qualification or contrast.
Separation is often favored when merging crowds several logical roles into one
sentence, buries a qualification or adds referent burden. These observations
are exploratory, not independently validated rules.

| Returned readability judgment | Median source-pair words | Median candidate words |
|---|---:|---:|
| Merge better | 27.5 | 25.0 |
| Separation better | 41.5 | 40.5 |
| Comparable | 38.5 | 35.0 |

Counts use whitespace-delimited words. Some longer merges are favored, so this
does not establish a universal length threshold. Length is confounded with
paper, content and candidate construction.

| Paper | Candidates checked | Merge better | Separation better | Comparable |
|---|---:|---:|---:|---:|
| Grossman–Stiglitz 1980 | 11 | 8 | 2 | 1 |
| Hart–Moore 1990 | 12 | 3 | 9 | 0 |
| Modigliani–Miller 1958 | 13 | 1 | 9 | 3 |
| Spence 1973 | 16 | 6 | 4 | 6 |

The paper differences reinforce the need for whole-paper validation. Repeated
paragraphs and shared model judgments prevent treating all 52 cases as
independent replications. The nine surface-flagged cases include four faithful
readability improvements and three local shifts; the 43 unflagged cases include
five local shifts. Surface flags should neither veto all flagged candidates nor
certify unflagged candidates.

## Limits and next step

The proposed [CPU selector comparison](SELECTOR_RESULTS.md) is now completed;
the recommendation below records the next step identified at this audit.

This is a selected-positive check: the other 100 Stage 2 KEEP-preferred boundaries
were not rejudged. It cannot establish corpus-wide selection accuracy, calibrated
fidelity rates or deployment performance. No source-only mapping model has been
tested on these judgments. The full neutral paragraph was supplied as context;
original author wording and earlier labels were excluded from the verifier
payload. Shared GPT-6 generation/verification, pretraining and possible cross-chat
retrieval still limit independence. All current responses mark external context
unnecessary for their local comparison; this does not resolve the earlier
source paragraphs' missing-context cautions.

The next useful check is a **small CPU selector comparison**, against KEEP and
a length-only baseline, with whole-paper validation. Use source-only features;
keep ties and claim-shift judgments separate. The eighteen faithful/merge-better
cases are provisional analysis seeds, not approved fine-tuning data. With only
four papers and selection restricted to earlier proposed merges, any result
must remain exploratory and conditional on that proposal set. More QLoRA
training is premature on the strength of this check alone.

## Audit artifacts

Private snapshot: `research_runs/relation_pilot_v1/verification_audit_v1/`.
Raw response/coverage bytes, model-confirmation notes, unit evidence and per-item
checks are hashed and preserved. Supplementary group IDs, length calculations
and interpretation limits are pinned in `verification_content_notes_v1/`.
The [aggregate summary](verification_results.json) contains no paragraphs.

CPU importer: `fama_style.research.relation_pilot_verification_review`, using
`--packet`, `--output` and `--session`; output must be fresh. Checks cover all
fixed text, sequential unit IDs, source/candidate/context/addition/readability
quotes, missing required evidence and faithful-verdict contradictions. These
checks cannot establish that the extracted units exhaust the source meaning.
Tests exercise corrupted text/evidence, false faithful labels, missing/duplicate
IDs, raw provenance preservation, snapshots and nonapproval of training data.
No reserved examples, new model calls, classifier fits, training or GPU jobs
were used by this import and analysis.
