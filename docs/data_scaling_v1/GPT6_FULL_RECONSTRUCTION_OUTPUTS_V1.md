# Full-cohort reconstruction expansion: output assessment

The eleven returned Stage 2 files contain all 320 expected training-side
reconstructions. IDs, order, target hashes and required fields pass validation.
All responses report `rewritten`, with empty review-note arrays. Raw responses,
human targets, earlier inputs and experiments remain unchanged.

**305 paragraphs are retained unchanged; 15 have limited local repairs.**
All 320 support candidate pairs after this assessment. No manual recollection is
requested. This is automated text assessment with targeted reference checks,
not independent semantic certification or exhaustive claim alignment.

| Diagnostic, median over these 320 targets | Prior GPT-6 input | Reviewed reconstruction |
| --- | ---: | ---: |
| Word-sequence similarity to human target | 0.500 | 0.154 |
| Ordinary shared five-gram coverage | 0.154 | 0.000 |
| Longest aligned ordinary-word span | 6 | 3 |
| Input/target word-count ratio | 0.848 | 0.647 |

All 320 have lower sequence similarity than their previous input. None triggers
the existing contrast triage rules after repairs. These findings support a
substantially different expression of content, not proven distinctive author
style, student-model improvement or detector-score gains.

## Repairs and tradeoffs

Eighteen exact patches affect fifteen paragraphs:

- P039: constrain a statement about the disappearance of alternative contracts.
- P121: merger fractions concern firms in cohorts, not fractions of exits.
- P226/P236: separate citations for word-of-mouth versus SIR parameters, and for
  elimination evidence versus the negative idiosyncratic-beta result.
- P304: distinguish c1–c3's twelve cases from the consistently significant c4 result.
- P308: distinguish M&A affiliate-count denominators from FDI investment positions.
- P352/P396: distinguish integrated-market accounts from an independent
  cheap-finance mechanism; Froot–Stein is a contrasting theory in this passage.
- P368: retain the negative-expected-value lottery observation. Its new gross/net
  payoff calculation is consistent with the supplied probabilities and is kept.
- P393: remove an unsupported within-industry interpretation of the benchmark.
- P403: avoid describing credit deterioration as weakening ABX spreads.
- P433: remove an unsupported ranking of variation between two periods.
- P441: attach the named authors to the separated-control puzzle.
- P442: the October 2005 article reports an earlier slump and recovery, not a
  January–April period following the article.
- P490: clarify the state-dependent cap on an otherwise constant real transfer.

Five derived briefs also receive versioned clarifications so Route A does not
inherit ambiguity revealed by realization: P236, P308, P393, P441 and P442.
Their original Stage 1 review and Stage 2 upload packets remain intact.

Moderate interpretive additions, compression and shifts of emphasis are accepted.
The repairs do not restore expressive target wording or enforce a word-change
percentage. Fifty-eight inputs are shorter than 45% of their target's word count;
this is an advisory observation, not a rejection rule. Some material is compressed
at the brief stage. Both proposed routes may therefore teach expansion from
notes or concise prose as well as style transformation. Report this confound in
any pilot; low lexical overlap alone is not a sufficient quality metric.

## Candidate datasets and review

Private output folder:

`research_runs/data_scaling_v1/gpt6_track_0500_v1/full_reconstruction_v1/rewrite_quality_v1/`

- `assessed_rewrites.private.jsonl`: all 320 raw/final versions and replayable decisions.
- `adjudications.private.jsonl`: exact prose patches and reasons.
- `brief_clarifications.private.jsonl`: ten exact edits across five briefs.
- `reviewed_briefs_0495.private.jsonl`: combined brief lineage, including two earlier concerns.
- `candidates_0500.private.jsonl`: 493 fresh prose inputs plus seven prior-input placeholders.
- `candidates_0200.private.jsonl`: the unchanged-order first 200 of that candidate version.
- `brief_human_candidates_0493.private.jsonl`: matching usable briefs and human targets
  for Route A. Brief inputs are deterministic JSON strings of task/points/anchors;
  a future training implementation must use an explicitly matched inference format.

Both routes currently cover the **same 493 human targets**. Five targets still
require source repair (P250, P264, P291, P458, P484); two earlier reconstructions
retain concerns (P134, P139). The goal remains 500 complete targets. No historical
human target or split has been changed, and no candidate is marked training-ready.

The local review catalog includes this new candidate version under
“GPT-6 full brief reconstruction v1.” Earlier versions remain available for
comparison. The interface is read-only and exposes no feedback-export workflow.

Verify with `python3 scripts/data/review_gpt6_full_outputs.py check`.
Manifest hashes bind responses, edits, derived briefs, prior candidates and outputs.
No API calls, GPU jobs, detector scans, development examples or reserved-test
examples were used in this assessment. Resolve the seven outstanding cases before
freezing the complete matched release; no further bulk regeneration is needed
for the 320 just reviewed.
