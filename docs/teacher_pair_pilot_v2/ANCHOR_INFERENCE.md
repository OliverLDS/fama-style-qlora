# Semantic anchors: a matched Gemma 4B inference check

The user executed this notebook as `oliverlds/semantic-anchor-4b` version 1.
The [completed review](ANCHOR_INFERENCE_RESULTS.md) confirms all 48 rewrites
and 96 scores, with zero updates. Original-input mean-NLL preferences favor
faithful targets in 12/12 cases for both models, while generation still has
material issues in 9/12 base and 10/12 adapter outputs. The specification below
preserves the executed scope and does not authorize a rerun. This probes the
[12 provisional anchors](SEMANTIC_ANCHORS.md) on base Gemma and the saved
expanded-pilot epoch-2 adapter. Use the same twelve training-side cases in both
conditions: their original teacher input and their new neutral faithful variant.
The original human target and one-edit counterexample are fixed across inputs.

## Questions and scope

Does the new input wording make faithful rewriting harder? Does the existing
adapter preserve the focused quantities, conditions and relations under that
change? Separately, does conditional likelihood favor the faithful human target
over its minimally altered, unfaithful counterpart?

| Factor | Fixed scope |
| --- | --- |
| Anchors | 12 from ten training papers; three per error family |
| Input conditions | Original teacher input; new faithful variant |
| Models | Pinned base Gemma 3 4B; same base with reviewed expanded epoch-2 adapter |
| First-attempt rewrites | 48 = 12 × 2 inputs × 2 models |
| Teacher-forced completion scores | 96 = 12 × 2 inputs × 2 candidates × 2 models |
| Likelihood contrasts | 48 paired chosen/rejected decisions |
| Tokenizer/label audits | 48 unique full scoring sequences, reused across models |
| Optimizer updates | **0** |

All human targets and original inputs were seen during adapter training. The
variants are new wording of those same targets, not new held-out content. This
is a training-side robustness/realization diagnostic, not generalization evidence,
a new independent semantic benchmark or a distinctive Fama-style evaluation.
The cases, ledgers and rejections have same-reviewer provisional validation.

## Run manually in Kaggle

1. Create a **new private Dataset**, for example “Semantic Anchor Inference V1”.
   Upload only the two files from `data/semantic_anchor_inference_v1/upload/`:
   `semantic_anchor_jobs.jsonl` and `semantic_anchor_config.json`.
   Keep `private_key/`, raw anchor drafts and other datasets out of the upload.
2. Import `kaggle/semantic_anchor_4b.ipynb` as a **new private notebook**.
   Attach the new Dataset.
3. Attach the **version 1 notebook outputs from
   `oliverlds/teacher-pair-expanded-pilot-4b`**, including
   `teacher_pair_pilot_v2/epoch_2/adapter/adapter_config.json` and
   `adapter_model.safetensors`. No retraining, new adapter upload or manual
   editing is needed. The exact reviewed hashes are checked before model access.
4. Enable **HF_TOKEN in Kaggle Secrets**, **Internet** and **GPU T4 x2**. Start
   a fresh session; the first cell exposes GPU 0 only.
5. **Save & Run All**. The notebook runs inference automatically under the
   fixed scope. It has no training switch or optimizer. Send the actual slug
   and version when complete.

The notebook source embeds only code and hashes. All copyrighted paragraphs,
variants, rejections and outputs remain private. It does not use the reserved
test or eight development paragraphs. No CLI push or automated submission is
performed by the preparation tools.

## Implementation and safeguards

Use `google/gemma-3-4b-it` at revision
`093f9f388b31de276ce2de164bdc2081324b9767`, the executed pilot's NF4 double
quantization/FP32 compute, SDPA attention, seed 1729 and greedy 512-token limit.
All model parameters are frozen, and execution uses inference mode. The existing
root 1B training CLI, executed pilot runtime, data and results are unchanged.
ML imports remain inside the repository-convention `run_training` entrypoint;
in this new evaluator that function performs inference only.

Generation receives **only the ordinary rewrite instruction and input text**.
It never receives the reference, rejected target, claim ledger, error label or
oracle rhetorical plan. Targets are separately supplied as assistant completions
for teacher-forced loss; that is scoring, not training or rewrite prompting.
Input-condition order alternates across anchors and is the same for both models.
Independent completion scoring does not present chosen/rejected labels to the
model as an A/B selection task.

Before loading weights, the tokenizer verifies every original/variant prompt,
both full completion boundaries and decoded targets. Prompt tokens are masked
from loss. End-of-turn supervision follows the same chat template as the pilot.
No truncation or silent row dropping is allowed, and prompt plus 512 generated
tokens must fit the 1,024-token context cap. A failure preserves job ID, candidate,
token counts and partial audits in `state.json`.

The base model is evaluated first. Then the exact reviewed rank-8 epoch-2 adapter
is attached with `is_trainable=False`; every parameter is rechecked as frozen.
No LoRA initialization, trainer, optimizer, backward pass or weight update exists
in the evaluator. Each generation and score is persisted as it finishes. There
is no automatic retry or budget extension after failure.

HF_TOKEN is retrieved from Kaggle Secrets without printing it or writing it to
reports. Secret-loading and execution exceptions suppress underlying message
details; diagnostic state records phase/job and exception type. Runtime package,
model, adapter, data and source hashes are recorded for the review.

## Scoring and interpretation

For each candidate, record mean negative log likelihood (NLL), supervised-token
count and total NLL. Define each margin as **rejected NLL minus chosen NLL**:
a positive margin favors the faithful target, negative favors the rejection and
an absolute margin at most 1e-8 is a numerical tie. Report mean and total margins
separately, including any disagreement. Mean loss has normalization bias; total
loss can prefer shorter sequences. Neither is a calibrated truth probability.

Report contrasts by model, input condition and error family, with ties and
per-case margins. These repeated measurements are paired views of twelve cases,
not 48 independent semantic-validation examples. Target memorization, collocation
and synthetic-edit shortcuts may raise apparent accuracy. Faithful ranking alone
cannot justify a preference-training claim.

For the 48 rewrites, assess claim preservation and generic expression separately.
Use the frozen ledgers and target only during post-generation review. Record
material additions/omissions, changed scope, time order, numeric roles and causal
claims, as well as minor wording flags. Compare each variant input to its original
input within the same model; compare base/adapter on the same input condition.
Report exact source copies, empty/length-limited outputs, source/reference overlap
and generic organization/concision preferences. Preserve every first output.
Partly masked automated review remains provisional, not independent human proof.

Interpret outcomes without automatically choosing an optimizer:

- Worse rewriting on new wording would motivate a bounded paraphrase-robustness
  intervention, after rechecking the variants themselves.
- Weak faithful-target likelihood preference could motivate testing the
  contrastive signal, but not establish that preference optimization will help.
- Good target ranking with drifting generations would distinguish ranking from
  free realization; an easier scoring task is not a solved rewrite problem.
- Improvements here would still need evidence beyond these training targets.

This notebook does not select or execute the next training method. The 45-example
Fama and 20-example multi-author reserves remain sealed.

## Outputs and reproduction

Download all of `semantic_anchor_4b_v1/`: `state.json`, `evaluation.json`,
`label_audit.json` and `generation_config.json`, plus the emitted runtime source
and notebook log. The report contains base/adapter snapshots, 48 generations,
96 completion scores and 48 contrast records. Runtime and memory costs have not
been measured. With 48 autoregressive generations, this may take longer than
the earlier training pilot despite performing no updates; progress is saved
per job. Keep partial outputs if it fails.

[anchor_inference_plan.json](anchor_inference_plan.json) pins the packet/runtime
and exact adapter hashes. No unexecuted results are reported.

```sh
PYTHONPATH=src python3 -m fama_style.research.semantic_anchor_inference_packet \
  --anchors data/teacher_pair_semantic_anchors_v1 \
  --adapter research_runs/teacher_pair_reverse_pilot_v2/kaggle_pilot_v1/teacher_pair_pilot_v2/epoch_2/adapter \
  --pilot-review research_runs/teacher_pair_reverse_pilot_v2/pilot_review_v1/results \
  --output /tmp/semantic-anchor-inference-reproduction
python3 scripts/build_semantic_anchor_4b.py
PYTHONPATH=src python3 -m fama_style.evaluation.semantic_anchor_4b \
  --jobs data/semantic_anchor_inference_v1/upload/semantic_anchor_jobs.jsonl \
  --config data/semantic_anchor_inference_v1/upload/semantic_anchor_config.json \
  --check
```

The CPU check does not load a model or tokenizer. Tests cover changed adapter
files, prompt/reference contamination, matched-condition bindings, target masking,
loss-margin signs and length disagreement, secret-error suppression and absence
of training calls. The notebook is self-contained, free of saved outputs and
below Kaggle's source-size limit. Actual model-tokenizer/GPU preflight occurs
only when the user runs it in Kaggle.
