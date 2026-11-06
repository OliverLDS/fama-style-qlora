# GPT-6 purpose-only 500: completed pilot, mixed behavioral findings

Reviewed Kaggle [`oliverlds/gpt6-purpose-500-4b`](https://www.kaggle.com/code/oliverlds/gpt6-purpose-500-4b), version 1.
The notebook executed successfully. This is **not yet a successful rewriting
checkpoint**: the adapted model copies more input and sometimes produces
repetitive or severely shortened prose despite much lower validation loss.

## Execution verified

- All 500 training pairs seen once; 250 optimizer updates, exactly the frozen
  seed/shuffle and learning-rate schedule.
- One Tesla T4, approximately 32.4 minutes for the recorded runtime, peak
  allocated GPU memory 5.55 GiB. Installation time is not included in that
  runtime measurement.
- All 524 label checks passed without truncation; target-only likelihood agreed
  with the built-in masked loss at the saved sanity-check point.
- All 250 recorded gradient norms were finite and positive.
- The saved adapter changed and reloaded exactly: 136 FP32 tensors, 2,228,224
  parameters. Downloaded adapter file hashes match the report.
- `evaluation.json` and `state.json` both mark completion. All 24 base and
  24 adapted generations exist; none is empty or hits the output cap.
- Downloaded runtime, model revision, package versions and dataset/config match
  the frozen preparation. Canonical-PDF split checks pass. The log passes the
  credential-pattern screen; no credentials were printed during local review.

Kaggle's output API rejected numeric `version_label=1`. The downloader instead
verified `current_version_number == 1` through SDK metadata both before and
after retrieval. It saved all eleven files and a hash receipt, without storing
signed URLs. The notebook remains private. No new GPU or provider call was made
to generate prose during this review.

## Loss and copying

| Measurement | Base | Adapted |
| --- | ---: | ---: |
| Token-weighted development NLL, 24 variants | 3.587 | 0.900 |
| Existing-input NLL, 16 variants | 3.369 | 0.779 |
| Gemini-input NLL, 8 variants | 4.042 | 1.155 |
| Median input/output word-sequence overlap, existing inputs | 0.330 | 0.750 |
| Median input/output word-sequence overlap, Gemini inputs | 0.266 | 0.821 |
| Exact word-token copies across 24 outputs | 0 | 3 |
| Outputs with word-sequence overlap ≥ 0.8 | 0 | 10 |
| Outputs with an exactly repeated sentence of ≥ 6 words | 0 | 3 |

Word-sequence overlap is a token sequence similarity, **not a percentage of
words copied**. Token-copy counts normalize punctuation and case. Exact
sentence duplication misses repeated clauses and lightly reworded repetition.

Reference overlap also rises (existing-input median 0.298 → 0.612; Gemini
0.255 → 0.515), but many inputs already closely resemble their references.
Copying them therefore increases reference overlap too. This does not establish
author-style learning.

Teacher-forced loss scores each next reference token while providing earlier
reference tokens. Free generation must produce its own continuation. Its
copying, circular explanations and omissions are directly relevant even when
teacher-forced loss improves substantially.

All 24 base outputs are byte-identical to the previous mixed-500 pilot on the
same generation inputs, references and prompts. Relative to that prior adapted
model, median input overlap rises from 0.668 to 0.750 on existing inputs and
from 0.513 to 0.821 on Gemini inputs; exact word copies rise from zero to three.
Both training data and training instruction changed, so this comparison cannot
isolate the cause.

## Targeted reading

The private case notes record specific failures rather than claiming an
exhaustive semantic error rate. Examples in the frozen development order:

- Case 7 loses the Nexis sample counts and coverage description, invents an
  inappropriate characterization of air-bag inflation and repeats its opening.
- Case 12 reduces a paragraph about innovation investment and NPV advice to
  a circular complaint that the advice is not good.
- Case 14 substitutes a repeated contrast with prediction ability for the
  original fund-performance and regulation discussion.
- Case 19, a Gemini input, adds an unsupported causal chain involving money
  supply, purchasing power and falling prices.

These are unblinded same-agent observations, not independent judgments. Some
other outputs preserve useful content but mostly retain input wording. The
problem is not merely a strict claim-preservation gate: copying and repetitive
prose directly fail the desired style transformation.

## Important evaluation mismatch and next decision

Using the same lexical metric, the median source/reference overlap is **0.153
for the new 500 training pairs**, versus **0.783 for the existing development
inputs** and **0.647 for the Gemini development inputs**. Training inputs are
fresh brief reconstructions; these reused development inputs were primarily
direct paraphrases of the human references.

This mismatch is observable; its causal responsibility for the failures is
unproven. Near-copy inputs remain a legitimate robustness test and the observed
failures remain failures. However, the run does not show how this adapter
performs on held-out inputs made by the same fresh-reconstruction procedure.
It does not establish that the 500-pair collection is unusable or that more
epochs, data or a larger model would solve the problem.

The recommended next step is an **inference-only matched diagnostic**, retaining
this adapter and all data: add fresh purpose-only reconstructions for a small
fixed, author-balanced subset of the same development content, then run base
and this saved adapter with the identical simple instruction and decoding.
Retain the current direct-paraphrase inputs as controls. Do not show human
reference wording to Gemma. Compare copying, repetition, readable transformation
and major content changes across the input routes before deciding whether to
change training strength or expand to 1,000 pairs.

No new annotation batch, GPU run or detector scan has been initiated. Pangram
scans are deferred while the visible generation failures are diagnosed; no
repeat human-reference calibration is needed. No Fama-specific style improvement
or detector reduction is established by this run.

## Private artifacts and reproducibility

`research_runs/data_scaling_v1/gpt6_purpose_500_4b_v1/` contains:

- `raw/`: execution log, runtime, model adapter and all result JSON files.
- `download_manifest.json`: version checks, notebook metadata and hashes.
- `review_v1/`: verified execution summary, all 48 outputs with descriptive
  metrics, and its manifest.
- `behavior_audit_v1/`: training/development contrast, copying/repetition counts,
  targeted private case notes and its manifest.

CPU verification and audit scripts:

```bash
python3 scripts/build_gpt6_purpose_500.py --check
python3 -m unittest discover -s tests -p 'test_gpt6_purpose_500*.py' -v
```

The versioned review builders refuse to overwrite existing reports. Their
`verify` function can recheck downloaded files without model weights being
loaded. The original pilot preparation, historical mixed-500 reports and all
raw teacher responses remain unchanged.
