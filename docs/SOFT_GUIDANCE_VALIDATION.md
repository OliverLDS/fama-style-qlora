# Claim guidance versus soft rhetorical planning

Status: prepared, input-hash checked and CPU-tested; **not run on Kaggle**.
This comparison uses the passed v3 synthetic verifier calibration. It creates a
new package; the earlier blocked package, all results and the original Gemma 3
1B training workflow remain unchanged.

## Upload and run

1. Upload `kaggle/soft_guidance_validation.ipynb` into a **new private notebook**.
   Suggested title: **Soft Guidance Validation**.
2. Create a **new private Kaggle Dataset** containing only this file:
   `research_runs/soft_guidance_validation_v2/input_pack/soft_guidance_pack.json`.
   Suggested dataset title: **Soft Guidance Validation Inputs**. Attach it to
   the notebook. This is a new 21-example pack; the old four-example
   `concise_plan_pack.json` is not an input for this notebook.
3. Enable internet, **GPU T4 x2**, and the notebook's **HF_TOKEN** Kaggle Secret.
   Unlike the verifier-only calibrations, this run needs gated Gemma access.
   Only GPU 0 is visible. Never put a token in code or the dataset.
4. Select Save & Run All. Share the notebook slug and version after completion.

Keep the notebook, attached dataset and outputs private. The pack includes source
and reference paragraphs for evaluation. Upload only the pack, not prompt files,
reports, checkpoints or credentials. Nothing was uploaded or submitted by the
implementation or builder.

Approved pack SHA-256:
`07151d65d962d8a82f0e18afac15c511859534a582e85e066bcc7abd67b3d0a5`.

## Matched comparison

The frozen corpus supplies the same **21 development-validation examples** used
in earlier work, from one held-out paper. There are 42 first-attempt generations:

| Condition | Generator input |
|---|---|
| A: claims only | Frozen rewrite instruction, source text and source-only preservation ledger |
| B: claims + soft plan | Exactly the same prefix, followed by an optional source-bound presentation plan |

Both arms choose their sentence count freely. There are no target sentence slots,
length quotas, target schemas, reference words, reference claims or adapter weights
in generator jobs. Pair order alternates A/B and B/A. Both use pinned base
Gemma 3 4B, NF4 with FP32 compute, greedy decoding, seed 42, 768 new tokens and
a 4,096-token context limit. No truncation, reranking, automatic repair or
optimizer is used. Empty/cutoff outputs stop the run with preserved partial files.

The plan uses the sentence-length direction fitted only on the **247 training
references**, plus source-derived argument order and organization. All 21 sources
are eligible for optional adjacent grouping under the frozen profile. Grouping
must not invent a causal link, condition or other relationship. There is no
learned lexical prescription, and language-style changes remain constrained by
source meaning.

This is a **fixed soft prior**, not the complete discovered taxonomy, a trained
source-to-target mapper, or an oracle target plan. The planning feature profile
and style-distance measure share training features, so feature improvements are
not independent evidence of authorship. The 45 reserved test examples are never
read or used for selection. No fresh generalization estimate is claimed.

## Runtime sequence and calibration evidence

The notebook verifies pack bytes, train/validation identities, frozen inference
source hashes, job hashes and verifier code/fixture hashes before any model or
Secret access. The package records the reviewed
`oliverlds/verifier_calibration_v3`, version 1, evaluation and verification hashes.

On Kaggle it then:

1. Loads the pinned public NLI model and repeats the frozen v3 synthetic calibration
   as a **runtime canary**, using the original thresholds. This is no longer fresh
   validation. A failed canary stops the run before Gemma Secret access/generation.
2. Unloads NLI, retrieves `HF_TOKEN` without printing it, token-audits every prompt,
   and loads Gemma on GPU 0. It generates all 42 outputs once.
3. Unloads Gemma, reloads the same NLI model, and assesses all 21 source/reference
   pairs and all 42 outputs with the unchanged v3 verifier. A probability cache
   avoids duplicate model calls and supports offline metric reproduction.

The run never keeps both model weights on the GPU at once. ML imports remain
inside the inference runner named `run_training`; no trainer is invoked.

## Evaluation and limits

Report all examples, including malformed paragraphs, risk signals and abstentions.
The format check rejects JSON/fences, headings/lists and multiple paragraph blocks;
it is a surface contract, not a semantic judgment. It does not repair outputs.
Report separately:

- Verifier `model_consistent`, `risk_signal` and `abstain` counts, with 21 examples
  per condition as the overall denominator. Valid-format semantic proxy passes
  are distinct from diagnostic consistency on malformed output.
- Exact/token-normalized copying, near-copy threshold 0.90 and lexical similarity.
- Frozen reference-profile style-distance gain, its individual features and paired
  B-minus-A changes. Invalid-format denominators remain explicit.
- A joint **proxy** count requiring valid format, verifier consistency, no near
  copy and positive style-distance gain. It is not verified faithful style transfer.
- Input/output tokens, generation/canary/verifier/runner time and GPU memory.

Source/reference checks flag potential label alignment issues without removing any
pair after generation. The actual sources have three to seven surface sentences.
A CPU source-only scope check finds an unsupported relation expression in one of
21 sources; all examples remain included. Outputs may introduce further guard
abstentions. The verifier's twelve-unit, 256-pair and 512-token NLI limits can also
cause abstention; no evidence is truncated or silently treated as a pass.

The synthetic calibration does not establish paragraph-level economics accuracy.
NLI false alarms, unrecognized paraphrases, nested scopes and missed additions
remain possible. Treat an increase in warnings as diagnostic, not automatically
an increase in real claim errors. Do not tune on the reserved paper test split.

The research question is whether B improves meaningful style-proxy change without
increasing copying or worsening semantic screening relative to matched A. A mixed
or negative result remains informative. `training_ready` stays false: successful
execution or proxy gains do not automatically authorize QLoRA or mapper training.

## Outputs and reproduction

Kaggle output directory: `/kaggle/working/soft_guidance_validation_v2/`.
Main files are `evaluation.json`, `generations.private.jsonl`,
`metrics.private.jsonl`, `nli_probability_cache.private.jsonl`,
`jobs.private.jsonl`, `token_audit.json`, `run_status.json` and a recursive file-hash
manifest. `runtime_calibration/` retains the repeated synthetic canary results.
The manifest covers nested artifacts as well as the private generations.

Based on earlier short runs, roughly 10–20 minutes inside the runner is a planning
estimate, not a measured result for this experiment. Queue, installation and cold
model downloads are additional. Heartbeats identify slow model loading/generation.
Use a fresh session/output path for reruns; completed or partial runs are not
silently overwritten.

```sh
python3 scripts/build_soft_guidance_notebook.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m fama_style.rhetoric.soft_guidance_experiment \
  --dry-run --output-dir /private/tmp/soft-guidance-new-dry-run
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

All 180 CPU tests passed, including matched prompts, reference exclusion, input
and source-code pins, failed gates, context limits, output identity/cutoff checks,
unknown/invalid-output denominators and self-contained notebook dry execution.
The original CLI dry check also passed without downloading or training a model.
Synthetic runs are wiring checks only; no model-quality results are reported.
