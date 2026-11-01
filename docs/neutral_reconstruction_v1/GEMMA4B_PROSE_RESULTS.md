# Matched Gemma 4B prose comparison: version 1 results

All nine first outputs completed on one Tesla T4 in 196.3 seconds, with no empty,
blocked, output-limited or formatting-flagged responses. The actual Kaggle slug
is `oliverlds/reconstruction-prose-4b` (the title contains underscores). Downloaded
runtime code, model revision, input configuration and prompt hashes match the
frozen packet. The pinned model revision is
`093f9f388b31de276ce2de164bdc2081324b9767`. No new inference or training was run
while reviewing these saved outputs.

The earlier JSON-wrapper failure is absent because the output contract is now
prose only. Execution and contract completion do not imply semantic success.

| Measure | Direct rewrite | Detailed brief | Compact claims |
| --- | ---: | ---: | ---: |
| Outputs | 3 | 3 | 3 |
| Outputs with claim/coverage flags in automated review | 3 | 3 | 3 |
| Median input tokens | 297 | 1,552 | 1,080 |
| Median reference word-sequence overlap | 0.433 | 0.257 | 0.255 |
| Median longest shared word run | 9 | 6 | 7 |
| Missing or added unique numeric values | 0 | 0 | 0 |

Compact claims use about 30% fewer median input tokens than the detailed brief,
but show no reliable fidelity advantage in these three cases. Both brief formats
have lower reference overlap than direct rewriting. Lower overlap is not success
when content is omitted or altered. Numeric-value checks pass despite changed
comparisons, binding and relationships; mere retention of dates is insufficient.

These counts combine issues of different severity: an omitted supporting analogy,
an ambiguous comparison and an explicit actor reversal are not interchangeable.
They are unblinded automated judgments, not calibrated error rates, independent
semantic gold or a ranking of all three methods. The same three failure-selected
training-side cases do not establish generalization or Fama/coauthor style.

## Case-level findings

**Liquidity phrase usage.** Direct rewriting loses the explicit statement that
usage during the housing boom exceeded the late-1990s level. Detailed
reconstruction retains that comparison and the correct search actor, but omits
the standalone claim that usage surged during the late-1990s boom. Compact
reconstruction retains the late-1990s rise but creates an unsupported or ambiguous
comparison involving the mid-1980s; its opening is also awkward. All retain the
numeric dates, so numeric matching alone misses these issues.

**Commentators and contagion.** Direct rewriting turns two alternative fallback
views into a conjunction and adds epistemic/frequency qualifiers. Detailed
reconstruction reverses who hears whom: economists hear commentators instead of
commentators hearing economists. It also loses psychological/sociological
foundations and changes the stated cause of low confidence. Compact reconstruction
keeps the correct confidence dependency but omits avoidance of sociological views,
weakens both the all-events monetary view and categorical media-authority barrier,
and adds a disclaimer about causation from an annotation guardrail.

**Sentiment and returns.** Direct rewriting does not retain the equal-arbitrage
condition, replacing it with inconsistent opportunities. Detailed reconstruction
retains the main conditional demand/return relationships and equal-arbitrage
assumption but drops the lottery analogy and adds an unsupported identification
of a framework component. Compact reconstruction drops the near-definitional
sentiment-to-demand relationship as well as the analogy, while repeating assumptions
and identifying the two components with scenario assumptions not explicitly
identified that way in the paragraph. Detailed is stronger on this particular
core relationship, which is not a general win across cases.

## Interpretation and next gate

The experiment establishes a working 4B inference pipeline and localizes failure
modes in using Gemma as a **neutral-data writer**. It does not evaluate its ability
to learn the reverse training direction, neutral input to human author paragraph.
In particular, student ability to reproduce a teacher's brief-to-neutral-data
construction process is **not a prerequisite** for ordinary text-pair SFT. Treating
that diagnostic as a necessary readiness gate was too strong. The student can be
trained on teacher-produced, source-checked pairs without receiving or generating
the teacher's briefs. Plan-conditioned rewriting remains a separate hypothesis
that must be tested in its own deployment format.

Recommendation: stop repeated brief-format tuning on these three examples and do
not use their current Gemma reconstructions as accepted training inputs. Return
to the intended training task: select a small source-checked set of existing
teacher-produced neutral/human pairs, freeze paper-level development examples,
and prepare a matched base-versus-QLoRA pilot. Evaluate claim preservation,
copying and meaningful stylistic change independently. Planning and pair curation
can be CPU-only; no training should be submitted without explicit authorization.
This review creates no training-ready pairs or new training packet.

The private download is
`research_runs/neutral_reconstruction_v1/gemma4b_prose_kaggle_v1/`.
Per-output text-bound assessments, metrics, original references and hash receipts
are in `research_runs/neutral_reconstruction_v1/gemma4b_prose_review_v1/`.
Original outputs, earlier failed runs, local 1B results and source archives remain
unchanged. Public aggregate results are in
[gemma4b_prose_results.json](gemma4b_prose_results.json). The CPU review importer is
`fama_style.research.reconstruction_prose_4b_review`.
