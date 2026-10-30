# Stage 2: source-only relation judgments audited

All eight files cover **39 paragraphs, 191 sentences and 152 boundaries**. Every
record passes the frozen schema and mechanical audit. Source paragraphs are
unchanged; fresh D-claim bindings and exact evidence resolve; every adjacent
sentence boundary is covered once with evidence from both sides. Preferences
agree with permitted actions, and every BOTH judgment includes a concrete
candidate. No text, claims, verdicts or raw provenance were repaired.

## Settings and limits

The user confirms **GPT-6**, with **High requested but effective High/Medium
uncertain** because the interface sometimes switched effort. All 39 raw records
declare GPT-6 with null thinking setting. Raw values remain intact; the user's
report is stored separately. No per-batch effective effort, hidden model
snapshot or controlled decoding setting is inferred.

This uncertainty does not require discarding the discovery annotations or
rerunning solely to change their provenance label. It prevents describing this
as a consistently High-effort experiment or attributing differences to effort.
Shared model, pretraining and possible cross-chat retrieval also remain
uncontrolled. The raw `reference_supplied=false` declaration records annotation
provenance, not proof of hidden-context independence.

## What the labels say

| Annotation | Boundaries |
|---|---:|
| BOTH permissible | 130 |
| KEEP only | 22 |
| MERGE only / UNRESOLVED | 0 / 0 |
| Presentation preference: KEEP | 100 |
| Presentation preference: MERGE | 52 |

Thus **KEEP is permissible at all 152 boundaries under these annotations**.
Permission-only accuracy would reward keeping everything and would not show
useful structural learning. No classifier was evaluated here. The nontrivial
signal is the separate presentation preference: 52 preferred merges versus 100
preferred separations. These are GPT-6 judgments of compact economics prose,
not established author-style features or uniquely correct reference layouts.
Mandatory merging may also be intrinsically rare under a specification that
allows reasonable alternative organizations; no class balance is forced.

The judge supplies 325 source claims. Thirty-two paragraphs flag review needs
and thirty flag external-context needs. Claims are newly indexed from source,
but neither claim completeness nor economic validity is independently verified.
Previous Stage 1 uncertainty, the held-back pending rewrite and the separate
antecedent interpretation caution remain unchanged.

## Candidate audit

All **130 candidates remain semantically unverified**. Mechanical consistency
does not make the candidate a faithful realization or validate its own verdict.
Surface checks flag **26 candidates**, including **9 of the 52 preferred merges**:

| Overlapping warning categories | Candidates |
|---|---:|
| Changed guarded-word inventory | 14 |
| New explicit causal cue under a bounded keyword rule | 10 |
| More than 60 whitespace-delimited words | 5 |
| Changed numeric-token inventory | 1 |

These are triage rules, not semantic decisions. The causal rule misses synonyms
such as an existing explanatory connective; a new keyword need not add a new
relation. Guard paraphrases may preserve meaning. The numeric warning concerns
one fewer repetition of a numbered actor, not a demonstrated changed quantity.
The length threshold is a recorded readability heuristic, not a safety boundary.
No possible multiple-sentence candidates are flagged after masking common
abbreviations; that rule is not a full sentence parser.

Coding-assistant inspection of all preferred merges and all flagged candidates
records a potentially strengthened explanatory connection and a potentially
weakened assertion/condition, alongside the numeric false-alarm explanation.
Private notes pin their IDs and source/output provenance. These observations are
not independent verification, and zero flags does not establish fidelity.

Candidate median length is 39.5 words (range 10–81); 13 use semicolons. Median
token sequence overlap with the source pair is 0.894. High reuse is compatible
with a local grouping edit and does not by itself indicate either author-style
learning or a failure of a paragraph rewriting model. All judgments are local;
overlapping pairwise merges must not be chained without a new global check.

## Recommended next experiment

The [verification packet](VERIFICATION.md) for this check was subsequently
completed; see its [verification results](VERIFICATION_RESULTS.md). The
recommendation below records the next step identified at the Stage 2 audit.

Check **the 52 preferred merge candidates** against their actual source
paragraphs for claim fidelity and readability, with source-only evidence and
explicit abstention. Keep fidelity, preference and permission separate. Do not
accept the annotator's own candidate as proof that a merge is permissible.
Retain flagged/unflagged status for sensitivity analysis rather than discarding
examples automatically. Do not regenerate the corpus merely to force MERGE-only
labels or remove uncertain settings.

After those checks, a small source-only **preference** mapper can be tested with
whole-paper validation, against KEEP and a fixed-plan baseline. Four papers
support only an exploratory comparison. Source features must exclude returned
judgments, rationales, candidates and reference layouts. This report does not
authorize or implement that fit, new generation or GPU training.

## Artifacts and checks

Private snapshot: `research_runs/relation_pilot_v1/stage2_audit_v1/`.
Raw responses and coverage reports are copied byte for byte; user-reported
settings, per-row errors, candidate flags and hashes remain private. Additional
content notes are frozen separately in `stage2_content_notes_v1/`.
The [public summary](stage2_audit_summary.json) contains aggregate counts only.

CPU implementation: `fama_style.research.relation_pilot_stage2_review`, with
`--packet`, `--output` and `--session` arguments. Use a fresh output directory.
Tests cover source/evidence corruption, broken claim bindings, boundary coverage
and adjacency, evidence on both sides, verdict/candidate consistency, surface
false alarms, missing/misplaced IDs, immutable snapshots, uncertain effort and
the absence of semantic/training certification. No reserved data, reference
layout scoring, model calls, classifier fits, Kaggle jobs or training were used.
