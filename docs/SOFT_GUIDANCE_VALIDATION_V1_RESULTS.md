# Soft-guidance comparison: Kaggle version 1

Reviewed `oliverlds/soft_guidance_validation`, version 1. All 42 first-attempt
generations completed, covering the same 21 development examples in both arms.
All formats were valid paragraphs. The runtime verifier canary passed. No adapter,
training, reserved-test access or new GPU submission occurred during this review.

**Conclusion: this fixed soft plan does not show an overall style-transfer
improvement.** Some examples improve, but the average style proxy worsens and
semantic screening does not establish a fidelity gain.

## Matched aggregate results

| Measurement | A: claim guidance | B: same guidance + soft plan |
|---|---:|---:|
| Verifier-consistent outputs | 6/21 | 9/21 |
| Risk signals | 9/21 | 10/21 |
| Abstentions | 6/21 | 2/21 |
| Joint format/semantic/style/non-copy proxy successes | 0/21 | 2/21 |
| Exact / near copies | 0 / 0 | 0 / 0 |
| Mean source lexical similarity | 0.451 | 0.538 |
| Mean style-distance improvement proxy | -0.966 | -1.487 |
| Mean paragraph-level words per sentence | 75.1 | 81.7 |
| Single-sentence outputs | 8/21 | 11/21 |
| Total output word/number tokens | 2,718 | 2,445 |

Higher style gain is better; negative values mean movement away from the frozen
training-reference feature profile relative to the input. It is not a validated
Fama/coauthor-style score. The style profile and planning operation share training
features, so these are not independent measures of authorship.

B's text is shorter in aggregate than A's but is packed into fewer sentences.
The source and reference mean paragraph-level sentence lengths are approximately
19.4 and 24.6 words, respectively. B's mean of 81.7 overshoots that range rather
than matching the observed target prose. One B output has a single 213-word
semicolon-linked sentence; this is a real output, not a segmentation artifact.

B improves the style proxy in 11 paired examples and worsens it in 10, with a
negative mean difference of approximately -0.520. Positive standalone style gain
occurs in five A outputs and four B outputs. Two B outputs pass the joint proxy,
but that small count does not outweigh the overall mixed results or certify
faithful author-style transfer. Higher lexical similarity is not a near copy at
the declared 0.90 threshold, but warrants reporting rather than calling copying
solved.

## Why 6 to 9 semantic passes is not a clear fidelity gain

Before v3's additional scope-abstention layer, the v2 verifier reports consistency
on **12/21 A outputs versus 11/21 B outputs**. The global/sentence NLI component
alone reports consistency on 14/21 versus 12/21. All six A and two B abstentions
come from unresolved scope guards, not token-context overflow.

Thus B's higher final consistency count is driven by fewer scope abstentions,
not improved underlying NLI results. This may reflect different surface wording
and guard applicability; it is not evidence that three more paragraphs preserve
their claims correctly.

Five examples move into final consistency under B, including two previously
abstained examples, while two previously consistent examples become risk signals.
Only four examples are verifier-consistent in both arms. All examples and all
unknowns remain in the report; none was dropped after observing outputs.

Source/reference checks classify four pairs as consistent and seventeen as risk
signals. These can reflect real alignment problems or verifier limitations on
economics prose. They are not seventeen confirmed label errors and do not justify
automatic exclusion or correction. Passing the synthetic canary did not establish
domain accuracy.

## Observable prompt/output failure

One planned output contains metalinguistic commentary about modal cue words and
stray quoted qualifiers rather than ordinary economic exposition. It also treats
a citation year as supporting quantitative evidence, which is not supplied by
the source. The cue-rich ledger lists isolated protected terms and quantities;
the observed artifact is consistent with those annotations being treated as
material to reproduce. That causal explanation requires a prompt ablation and
is not established by this single case.

The global NLI component nevertheless assigns minimum entailment about 0.957 to
this case. The final v3 verifier abstains; it does not certify the output. This
illustrates both a domain blind spot in NLI and the value of keeping abstention
separate from success. No source or output paragraphs are reproduced publicly.

The optional grouping operation also appears poorly calibrated for realization:
both arms already produce much longer sentences than the references, and B
increases the concentration into long sentences. More grouping is not a useful
default here. This tests one fixed, narrow prior, not the full discovered taxonomy,
a learned rhetorical mapper or a QLoRA/structure hybrid.

## Integrity and cost

The approved pack SHA-256, all recursive manifest hashes, model revisions,
generation controls, prompts and exported Python sources match. Using only the
saved probability cache reproduces all 42 output verifier decisions, all 21
source/reference checks and all 168 runtime-canary verdicts exactly. Metrics,
paired summaries and the canary gate also reproduce exactly. The logs contain
neither a Python traceback nor CUDA out-of-memory message.

Runner time is 742.7 seconds (12.38 minutes): generation takes about 284.1 seconds
for A and 268.7 for B; the canary stage takes 23.2 seconds and the post-generation
verifier stage 70.7 seconds. Runner time includes additional loading/overhead;
Kaggle queue and dependency installation are excluded. Input tokens increase
from 11,221 to 15,254 under B; output tokens decrease from 3,302 to 3,019.

Peak allocated memory is approximately 4.79 GiB for Gemma and 0.78 GiB for NLI,
loaded sequentially on one T4. There are 3,633 cached NLI probability requests.
All raw generations, probability caches, logs, hashes and paired diagnostics stay
private under `research_runs/soft_guidance_validation_v2/`.

Public aggregate metrics are in
[soft_guidance_validation_v1_results.json](soft_guidance_validation_v1_results.json).
The frozen preparation protocol, older blocked package and all prior runs remain
unchanged.

## Next step

Do not begin QLoRA or mapper training on this evidence, and do not treat the soft
plan as the winning method. Pause the default sentence-grouping idea rather than
extending it to more examples unchanged.

The next focused diagnostic should separate two prompt factors on a small,
explicitly post-hoc set of failing development cases: source-sentence evidence
alone versus the cue-rich ledger, crossed with grouping guidance off versus on.
Keep the source, master rewrite instruction, model and decoding fixed. Two cases
would require eight generations. Measure annotation leakage and sentence
inflation alongside semantic warnings, copying and style. This tests the suspected
prompt mechanism without another full comparison or training run.

These already-used development cases cannot establish generalization. No new
diagnostic notebook or GPU job was created or submitted as part of this review.
