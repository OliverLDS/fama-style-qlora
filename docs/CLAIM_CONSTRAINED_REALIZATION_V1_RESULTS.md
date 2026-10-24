# Claim-constrained realization: version 1 results

Reviewed `oliverlds/claim_constrained_realization`, version 1. All sixteen
first-attempt outputs and NLI assessments completed. The experiment used four
previously inspected training examples, pinned base Gemma 3 4B, no adapter,
no training and no held-out examples. Runner time was 334.9 seconds (5.58 minutes),
including 223.4 seconds of generation and 16.8 seconds of NLI. Queue, installation
and other notebook setup time are excluded.

## Matched results

| Measurement | A: count paragraph | B: claims + paragraph | C: count slots | D: claims + slots |
|---|---:|---:|---:|---:|
| Normalized valid format | 4/4 | 4/4 | 3/4 | 4/4 |
| Valid format and requested count | 2/4 | 0/4 | 3/4 | 4/4 |
| NLI proxy passes among valid outputs | 1/4 | 4/4 | 2/3 | 2/4 |
| Joint format/count/NLI proxy passes, all examples | 0/4 | 0/4 | 2/4 | 2/4 |
| Mean count error among valid outputs | 0.50 | 1.00 | 0.00 | 0.00 |
| Near copies among valid outputs | 0/4 | 0/4 | 0/3 | 0/4 |
| Mean style-gain proxy among valid outputs | -0.734 | -1.043 | -0.134 | -0.013 |

The count and JSON controls are reference-derived oracle information. The source
ledger is generated only from input evidence. These are diagnostic outcomes,
not deployment or generalization estimates. All four examples remain reported.

The ledger improves the paragraph condition's semantic-check pass count, but
sentence counts become source-like rather than target-like: the four B outputs
contain 2, 1, 1 and 5 sentences versus requests for 3, 2, 2 and 6. They all miss
by one. With slots, the ledger fixes count adherence on the fourth example, but
the joint success count does not increase.

| Training paper key | Requested count | A count / NLI | B count / NLI | C count / NLI | D count / NLI |
|---|---:|---|---|---|---|
| F1965 | 3 | 3 / flagged | 2 / pass | 3 / pass | 3 / flagged |
| FF2000 | 2 | 2 / flagged | 1 / pass | 2 / flagged | 2 / pass |
| NL2003 | 2 | 1 / flagged | 1 / pass | 1 / pass, invalid count | 2 / flagged |
| VL2018 | 6 | 5 / pass | 5 / pass | 6 / pass | 6 / pass |

The invalid NL2003 C array still receives diagnostic NLI on its joined text, but
is excluded from valid-only metrics and cannot pass the joint gate. Its row
remains in the overall denominator of four.

## Format, copying and style

All eight slot responses used outer Markdown JSON fences. The predeclared
normalizer removed exactly those wrappers: C has three valid payloads and D has
four. Raw JSON-contract compliance is zero in both slot arms. No content repair,
new generation, extra-key deletion or slot splitting occurred.

No valid output crosses the 0.90 near-copy threshold. This does not mean copying
pressure is absent: mean source lexical similarity rises from 0.261 to 0.356 in
the paragraph comparison. Slot-arm means are 0.352 for C's three valid outputs
and 0.484 for D's four, so those means have different example sets and must not
be interpreted as a clean paired effect.

Style gain is improvement in distance to the frozen training-reference feature
profile; higher is better. It is not a validated measure of Fama/coauthor style.
All arm means are negative. D's mean is close to zero rather than evidence of a
substantial style improvement. B worsens the proxy despite its NLI pass gain.
C and D style means also have unequal valid-output denominators.

## Remaining claim failures

Inspection of D's F1965 output shows a source upper-bound condition rewritten
as a lower-bound condition in one sentence and then expressed as an upper bound
again in another. Thus the explicit warning and evidence ledger did not prevent
a reversal or internal inconsistency. The NLI checker flags this output.

D's NL2003 output omits the explicit statement that most new listings are small
and loses the input's annual qualification. Its count is correct, but it is
NLI-flagged. Source-evidence retention is not guaranteed by matching slot counts.

Conversely, a pass is not proof: B's F1965 output adds a statement about limitations
on predictive power beyond the supplied input's claim about exact description,
yet passes NLI. A high entailment score can miss a stronger or added assertion.
These are qualitative counterexamples, not a newly validated claim-accuracy score.

All three equivalent controls, seven original semantic mutations and four targeted
mutations behaved as expected. The bound-reversal control reached approximately
0.990 maximum contradiction. Passing these simple controls does not establish
accuracy on longer economics paragraphs. The unchanged source/reference checks
still flag three pairs and pass one; no labels were silently repaired or dropped.

## Verification

The uploaded input pack matches the approved SHA-256
`83d834c37bfb5233c2a3dbf8b620ff522e8b467c9f0e1440f57c676257b9e0ee`.
All manifest hashes and the nineteen exported Python runtime sources match local
files. Jobs and prompt hashes match the frozen input pack. Recomputing saved
metrics and summary from generations and saved NLI probabilities reproduces them
exactly. All sixteen outputs are nonempty, non-synthetic and below the token limit.
The log contains neither a Python traceback nor a CUDA out-of-memory message.

Private downloads, raw generations and verification hashes are retained under
`research_runs/claim_constrained_v1/`. No copyrighted reference paragraphs are
reproduced here. Earlier runs and the preparation protocol remain unchanged.

## Decision

Source constraints provide a useful semantic-check signal in paragraph mode, and
sentence slots make count control actionable. Their combination does not yet
solve faithful style transfer: the joint proxy remains two of four, explicit
semantic errors persist, and the style proxy provides no clear success.

Do not start mapper training or another QLoRA run on this evidence. The next work
should use these saved generations to strengthen automated claim-level checks,
with explicit treatment of bound direction, omitted qualifiers and added claims.
Report unresolved cases rather than declaring them faithful from aggregate NLI.
This can proceed without generating more text or requiring user review.

After establishing a useful semantic gate, the next generation comparison should
relax exact sentence quotas so the model is not rewarded for filling slots at the
expense of claims. Test source-ledger and empirical structural guidance with
faithfulness as the primary criterion. The current experiment does not prove
that quotas caused the observed drift or that any such revision will succeed.

No new GPU job was submitted during this review.
