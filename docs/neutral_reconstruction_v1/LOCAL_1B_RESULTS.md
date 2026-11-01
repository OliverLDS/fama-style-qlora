# Local quantized 1B comparison: results

Executed nine first-attempt requests against the user's existing local Gemma
server. All completed in 228.9 seconds (including metadata checks, model-file
hashing, preflight and generation). No training, Kaggle run, paid API or remote
model call occurred. Every output is nonempty and stopped normally, with no
length-limit finish. The server model was `ggml-org/gemma-3-1b-it-GGUF:Q4_K_M`,
server build `b1-c589f0e`. The loaded GGUF file's SHA256 is recorded with the
chat-template hash, server defaults and explicit request settings.

The three previously exposed cases were tested under direct rewriting, detailed
brief reconstruction and compact-claim reconstruction. All conditions used the
same prose-only contract and greedy settings. The detailed brief includes the
versioned confidence/media-authority correction. No new source or target text
was generated to prepare the compact list; it deterministically preserves every
proposition and nonempty qualifier while resolving actor IDs and removing empty
fields and predicate labels. See [protocol](LOCAL_1B_PROTOCOL.md).

| Measure | Direct rewrite | Detailed brief | Compact claims |
| --- | ---: | ---: | ---: |
| Outputs | 3 | 3 | 3 |
| Material claim issue flagged in automated review | 3 | 3 | 3 |
| Median input tokens | 297 | 1,552 | 1,080 |
| Median word-sequence overlap with human reference | 0.326 | 0.123 | 0.176 |
| Empty, blocked or length-limited outputs | 0 | 0 | 0 |

The compact format reduces median input size by about 30% relative to the detailed
brief but shows no demonstrated fidelity benefit in this sample. Low overlap
cannot count as success when claims drift or disappear. These qualitative
classifications are unblinded automated judgments, not calibrated error rates or
independent verification. Three selected development examples do not establish
population performance, Fama style or generalization.

## Observed errors

- **Liquidity wording:** direct rewriting adds unsupported historical trend and
  comparison details. The detailed brief places a housing boom in the wrong
  decade and adds an origin story. Compact reconstruction loses the 2004 housing
  episode and changes which period had the highest usage.
- **Commentators and contagion:** direct rewriting adds unsupported economic
  judgments. The detailed brief loses the monetary explanation and confidence
  claim, while weakening asserted poor understanding into a potential deficiency.
  The compact version omits the market-efficiency alternative and adds frequencies.
- **Sentiment and returns:** direct rewriting reverses the equal-arbitrage
  assumption and changes the type of demand. Detailed reconstruction invents a
  rise in arbitrage difficulty and observed return behavior. Compact reconstruction
  drops a demand-shock condition and supporting analogy, while introducing
  observed/causal language beyond the hypothetical claim.

Rendered prompts used 279–1,835 tokens in preflight. Actual API prompt counts were
one token higher in every request, within the eight-token safety margin. Even
with the 512-token output budget, every request fits the 4,096-token slot context.
Cached-token counts were zero. Context overflow, output truncation and the previous
Kaggle JSON-wrapper problem therefore do not explain these outputs.

The local 1B endpoint is useful for checking API transport, request construction,
formatting and gross failures. These results do not support using it as a faithful
annotation writer or as a semantic evaluator. Its failure is not an automatic
rejection of the compact format for 4B: model size, quantization, engine, template
and output contract differ from the earlier Kaggle experiment.

Recommendation: avoid further repeated 1B prompt tuning on these three cases.
The next model-facing step, if authorized, is a matched, prose-only 4B comparison
of the same three conditions. Retain direct rewriting as the baseline, report
all claim and copying tradeoffs, and do not start post-training or scale the
brief-based data construction based on the present evidence. A later fresh sample
is still required for generalization. No new 4B job has been prepared or submitted
as part of this local run.

Private inputs, exact raw responses, server snapshot, token audits and hashes are
under `research_runs/neutral_reconstruction_v1/local_1b_v1/`. `package_v2` is the
executed frozen packet, `run_v1` contains all nine outputs, and `review_v1` contains
per-output judgments, metrics and receipts. The earlier `package` is an unexecuted
draft. Original Kaggle results and source data are unchanged. Public aggregates
are in [local_1b_results.json](local_1b_results.json).
