# Source-sentence editing: Kaggle version 1

Reviewed `oliverlds/sentence_preserving_edit`, version 1. All four model
generations completed and both baseline outputs exactly reproduced their prior
hashes. The runtime verifier canary passed. **There is a useful structural result,
but no verified style-transfer success.**

## Primary outcomes

| Measurement | A: baseline | B: all proposed edits | B: filtered edits |
|---|---:|---:|---:|
| Available paragraphs | 2/2 | 1/2 | 1/2 |
| Source sentence count retained | 0/2 | 1/2 | 1/2 |
| Verifier-consistent paragraphs | 0/2 | 0/2 | 0/2 |
| Semantic risk signals | 1 | 1 | 1 |
| Semantic abstentions | 1 | 0 | 0 |
| Invalid scripts | Not applicable | 1 | 1 |
| Exact / near copies among available outputs | 0 / 0 | 0 / 0 | 0 / 0 |
| Joint semantic/style/non-copy proxy successes | 0/2 | 0/2 | 0/2 |

The two B variants come from the same script per case; they are not additional
independent model generations. The invalid script remains in the denominator
and produces no paragraph, rather than a fallback copy counted as success.

## Valid script: useful sentence structure, unresolved meaning

On the longer case, the source and reference each have seven sentences. The
baseline generates six; the edit script preserves all seven source positions.
It proposes seven changed sentences. Six pass the local/contextual proxy checks;
one receives an abstention and retains its original source sentence.

| Same valid case only | Mean words per sentence | Style-gain proxy | Source lexical similarity |
|---|---:|---:|---:|
| Source | 22.0 | Reference point | 1.0 |
| Reference, diagnostic only | 23.0 | Not a generated output | Not used for generation |
| A baseline | 35.7 | +0.219 | 0.424 |
| B proposed | 26.1 | +0.231 | 0.688 |
| B filtered | 26.0 | +0.224 | 0.696 |

The sentence length is much closer to the reference. However, the aggregate
style proxy is only marginally better than the matched baseline on this case.
Do not compare the baseline's two-case mean of -0.309 with B's one-available-case
mean of +0.224 as though they were matched samples. Sentence-count compliance
is imposed by the edit application, not evidence that the model learned style.

Copying increases in lexical similarity, although neither B variant crosses
the 0.90 near-copy threshold and neither is an exact/token-normalized copy.
Fallback affects one of seven proposed edits; it is explicitly recorded.

## Local checks do not guarantee paragraph consistency

The edited second sentence changes the source actor category from **other
shareholders** to **other security holders**. This is a broader technical term,
not a harmless formatting substitution. The source paragraph also discusses
transfers involving bondholders, making the distinction material.

The sentence-level verifier nevertheless accepts this edit: its minimum local
entailment is approximately 0.981, and full-source-to-replacement support is
approximately 0.991. At paragraph level, the NLI check flags coverage of the
second source sentence: minimum entailment is about 0.670 for the proposed
paragraph and 0.605 for the filtered paragraph, with contradiction about 0.319
and 0.385 respectively. The new scoped guards report no conflict in this case.

The filter therefore lets a problematic role edit through. Both B paragraphs
remain risk-flagged. Per-sentence screening must not override the paragraph
check, and the locally accepted six edits must not be reported as six verified
meaning-preserving edits. The withheld fourth edit is an abstention associated
with wording outside the relation guard's supported scope, not a confirmed error.

## Invalid script: structural quotation marks

The shorter case's model output uses typographic opening/closing quotation marks
where JSON requires ASCII string delimiters. This leads to an invalid-control-
character parse error. It is not a missing model output or a training/runtime
failure. No derived paragraphs or semantic scores are assigned to it in the
primary results.

A separate **post-hoc syntax diagnosis** changes exactly four quotation-mark
delimiters around replacement values. It preserves all three replacement prose
strings character for character, keeps the same source IDs and then passes the
existing structural parser. This is not a global smart-quote replacement, a
rewriting operation or a retroactive change to the primary failure count.

The diagnostic copy and exact character-position patch record are retained under
`version_1_review/`. No new model calls were made. **The recovered script has not
received semantic verification**, so no recovered success is claimed.

## Verification and cost

All pack, prompt, implementation, model/control and recursive manifest hashes
match. Saved probabilities reproduce every per-edit decision, accepted-edit
application, derived paragraph verifier result, source/reference check and all
168 runtime-canary verdicts exactly. Metrics and summaries also recompute exactly.
All four raw generations are nonempty and below the token limit. Logs contain
neither a Python traceback nor CUDA out-of-memory error.

Runner time is 211.2 seconds (3.52 minutes). Baseline generation takes about
32.6 seconds; edit-script generation 47.8 seconds. Canary and verification stages
take about 20.1 and 16.2 seconds; the rest is loading/overhead. Queue and dependency
installation are excluded. Output tokens are 392 for the two baseline calls and
597 for the two script calls. No cost is counted as an extra generation for
the derived filtered variant.

Gemma and NLI peak allocated memory are about 4.73 and 0.78 GiB on one T4, loaded
sequentially. Raw text, edit decisions, caches, the format diagnostic and integrity
records remain private under `research_runs/sentence_preserving_edit_v1/`.
Public aggregates are in
[sentence_preserving_edit_v1_results.json](sentence_preserving_edit_v1_results.json).
Previous experiments, source pairs and frozen verifier code remain unchanged.

## Next step

Keep fine-tuning deferred. Before generating more rewrites, perform a small
**verification-only recovery pass** on the saved artifacts:

1. Reassess the delimiter-corrected script as a separately labeled post-hoc variant.
2. For the valid case, restore the globally flagged second source sentence and
   reassess paragraph consistency. Use the source-sentence ID mapping; do not
   invent a replacement or silently accept local proxy scores.
3. Report all rollbacks, retained source text, copying, style and remaining unknowns.

This requires no new Gemma generation and should not replace the frozen primary
results. It would test whether transparent format recovery and paragraph-level
rollback produce a useful edit proposal, not establish generalization on these
two repeatedly inspected development cases. No new job was submitted here.
