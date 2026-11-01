# Matched stronger DPO: objective improvement without a generation advantage

User-reported notebook: `oliverlds/dpo-strength-4b`, version 1.
The fixed `1e-5`, beta `0.1` DPO continuation completed successfully, but does
not improve on matched stronger SFT in the current automated comparison.
**The development quality gate still fails. No detector scans or further GPU
training were initiated during review.**

## Execution verified

The run completed 22 grouped updates, all finite/nonzero gradient checks,
42 response-token/mask audits and 27 generations, including nineteen fixed final
outputs. All eight original control generations reproduced. The reference cache
contains the original starting adapter's 34 completion scores; every score
matches its archived value within the declared tolerance. Seventeen initial
policy/reference identity checks passed at approximately `log(2)` loss.

Reference policy, model revision, adapter, data, decoding, package, group-order
and learning-rate hashes/checks passed. Every update records the immutable cache
hash. Both downloaded checkpoints match their file hashes and expected 136 F32
language q/v LoRA tensors; the final tensor fingerprint matches the report.
Final evaluation used the reloaded, frozen epoch-2 adapter. The emitted runtime
and two helpers match the frozen source bytes.

Worker runtime was **902.9 seconds (15.0 minutes)**, with peak allocated GPU
memory **5.10 GiB** on one Tesla T4. Stronger SFT's corresponding worker time was
587.5 seconds. This is an observed end-to-end cost difference, including gates
and evaluation, not a controlled throughput benchmark. DPO performs additional
reference, identity and rejected-response work. No outputs were empty,
token-limited or exact copies of the input.

## Fixed final comparison

| Measure | Original adapter | Stronger SFT | Stronger DPO |
| --- | ---: | ---: | ---: |
| Training material-issue cases | 11/11 | 9/11 | **9/11** |
| Training cases within local meaning budget | 0/11 | 1/11 | **1/11** |
| Development material-issue cases | 4/8 | 4/8 | **4/8** |
| Development cases within local meaning budget | 3/8 | 2/8 | **2/8** |
| Training target NLL, seventeen pairs | 2.24465 | 1.94140 | **1.95933** |
| Development target NLL, eight cases | 1.73675 | 1.51700 | **1.52575** |

DPO exactly reproduces **8/11 SFT training outputs and 5/8 SFT development
outputs: thirteen of nineteen total**. The six differing outputs have mostly
small lexical or scope/roadmap changes; none resolves a remaining material-error
case relative to SFT. Byte-identical outputs inherit the earlier exact semantic
assessment; five previously unseen text strings were separately inspected, while
one output reverted to an already assessed original control.

The same two selected training cases lose their material-issue designation under
both methods. Only the dividend-timing case meets the full local meaning budget;
the predictability/risk-premium case retains two minor shifts. The same planning
repair and unsupported exclusive monitoring-purpose error offset each other on
development. Numerical-prospect omissions, ownership-scope problems and other
remaining errors persist. These eleven training cases were selected for known
errors and are not a representative accuracy sample.

Generic-expression comparisons against the original adapter are identical:
seven wins / three ties / one loss on training and four wins / three ties / one
loss on development. **DPO versus stronger SFT ties on all nineteen cases.**
These are provisional clarity/organization judgments, not distinctive Fama-style
or human-authorship measurements. No claim of general objective equivalence is
supported by this tiny, reused sample.

Median source overlap is 0.40288 for DPO versus 0.42534 for SFT on training,
and 0.41132 versus 0.42419 on development. Reference overlap medians are equal
between the two arms. Lower surface overlap alone is not better transformation
or fidelity. Exact input copying remains zero in both arms.

## Why the low DPO loss is not a successful rewrite

All seventeen training pairs improve their chosen-minus-rejected log-probability
margin relative to the starting policy. The mean relative improvement is
**37.42 nats**, and final mean diagnostic DPO loss is **0.03451**, versus
approximately 0.6931 at initial identity. These are measured model scores, not
synthetic algebra tests.

Nevertheless, all seventeen archived wrong responses still have higher absolute
summed response log probability than their full human targets after training.
That comparison is length-sensitive and does not directly predict greedy output.
The distinction matters: DPO rewards improvement in the chosen/rejected log ratio
relative to the original policy; a large relative improvement can coexist with
the original response remaining much more likely. Better preference margins and
a low surrogate loss therefore do not establish error correction in free generation.

The pipeline and optimization moved as intended. In this bounded setting,
explicit contrast with full human-reference targets did not add a demonstrable
generation benefit over showing those targets through SFT.

## Recommended next step: isolate the semantic preference signal

Pause automatic learning-rate/epoch sweeps and additional detector scans.
The next useful work is a **training-only, versioned minimal-repair preference
audit**: make each positive response a minimally corrected version of its actual
wrong model response, changing the unsupported or lost economic claims while
preserving unrelated wording. Keep the original human reference as a separate
style target and semantic evidence; do not relabel a repaired model response as
human writing or a Fama-style gold paragraph.

For that proposed branch, record the exact edits, source/claim evidence, unresolved
cases and lexical/length contrast before considering any new optimization. Hold
ambiguous repairs rather than forcing labels. Compare the new contrast with the
current full-paragraph contrast at CPU level first. The hypothesis is that a more
localized pair better isolates claim preservation from large stylistic/length
differences. This is **not a proven cause or implemented solution**.

This is a fidelity diagnostic supporting the main style-transfer objective,
not a replacement of the human-style corpus or authorization to train. No
reserved/development paragraphs should be used to build repair preferences.
No new GPU run, external annotation job or detector evaluation has been started.

## Artifacts and limitations

Private outputs, both adapters and log: `research_runs/dpo_strength_v1/kaggle_v1/`.
Fixed masked bundle, key, judgments and results:
`research_runs/dpo_strength_v1/review_v1/`.
The receipt records hashes of downloaded files, annotations, packet and shared
review code. Public aggregates: [dpo_strength_results_v1.json](dpo_strength_results_v1.json).
CPU reviewer: `src/fama_style/research/dpo_strength_review.py`.

The reviewer knew previous controls and SFT results. Masked labels were fixed
before reading the new wording; this does not establish independent or calibrated
judging. The existing one-minor/no-material budget and exclusive-purpose boundary
are retained. One seed and reused, tiny training/development subsets do not
establish broad superiority, impossibility or author-specific style. No reserved
examples or detector services were accessed. The downloaded log passed credential
pattern checks. As before, Kaggle's output API does not enforce the CLI's parsed
version argument: version 1 is user-reported, while actual source, configuration
and adapter content are bound to the frozen hashes.
