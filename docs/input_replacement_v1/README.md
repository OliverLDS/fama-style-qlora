# Controlled SFT input replacement

The approved experiment has completed as `oliverlds/input-replacement-4b`
version 1. See [results and limits](RESULTS.md): final claim issues remain six
of eight and direct expression preferences have one win versus two losses;
the predeclared gate fails. No further run is authorized. The instructions
below preserve the executed setup.

The user approved **one fresh two-epoch / 46-update experiment** after reviewing
the [anchor inference results](../teacher_pair_pilot_v2/ANCHOR_INFERENCE_RESULTS.md).
It replaces twelve selected training inputs with their neutral faithful variants
and preserves every human target. The preparation tools enabled manual Kaggle
execution; the user subsequently ran the notebook.

## Comparison and scope

| Factor | Original-input control | Input-replacement treatment |
| --- | --- | --- |
| Training targets | 46 human paragraphs | Exact same 46 paragraphs |
| Target occurrence count | Twice across two epochs | Same |
| Training input texts | 46 original teacher inputs | 34 unchanged, 12 replaced one-for-one |
| Training row order / IDs / papers / authors | Frozen | Same |
| Epochs / optimizer updates | 2 / 46 | 2 / 46 |
| Adapter starting point | Fresh base Gemma 3 4B | Same pinned base, fresh adapter |
| LoRA / optimizer / schedule / seed | Rank 8 q/v, LR 2e-5 linear, seed 1729 | Same |
| Evaluation | Eight development cases and two training probes | Same inputs, targets and decoding |

The control is `oliverlds/teacher-pair-expanded-pilot-4b` version 1. The new
adapter must **not** continue from that checkpoint. Source wording and resulting
input token lengths are the intended treatment; the number of targets, their
sampling frequencies and optimization budget do not increase. This avoids the
volume/update confound of the earlier data expansion. A single-seed comparison
on reused development examples still does not establish generalization or a
distinctive Fama signature.

The twelve replacements remain balanced across scope, time, numbers and reasoning
with three per family; all twelve are retained rather than selecting only the
reasoning-family inference successes. Original human targets and instructions
are unchanged. There are no rejected completions, preference labels, claim
ledgers or oracle plans in the training payload. This tests positive SFT with
more varied faithful inputs, not preference optimization.

## Manual Kaggle run

1. Create a **new private Dataset**, for example “Teacher Pair Input Replacement
   V1”. Upload only the two files from
   `data/teacher_pair_input_replacement_v1/upload/`:
   `teacher_pair_replacement.jsonl` and `teacher_pair_replacement_config.json`.
   Keep the audit directory and earlier datasets private and separate.
2. Import `kaggle/input_replacement_4b.ipynb` as a **new private notebook**.
   Attach this new Dataset.
3. Attach the **version 1 outputs of
   `oliverlds/teacher-pair-expanded-base-4b`**, including
   `teacher_pair_base_v2/evaluation.json`. This is the reviewed **baseline**,
   not the trained pilot's adapter output. No prior adapter is needed.
4. Enable **HF_TOKEN in Kaggle Secrets**, **Internet** and **GPU T4 x2**. Use a
   fresh session; the first cell exposes only GPU 0.
5. **Save & Run All**. `RUN_PILOT=True` under this approved scope; no code edit
   or additional enabling step is needed. Send the resulting notebook slug
   and version.

The notebook pins the exact data, config and reviewed baseline-report hashes.
It uses the existing baseline rather than requiring another standalone baseline
run: training inputs are not used in those ten baseline generations. The eight
development rows and two old probes are untouched, and the frozen control config
is validated separately from the new source-data hash.

Before training, all 54 rows undergo full-token/target-only label checks, decoded
target matching and generation-context checks, with no truncation or dropped
rows. A fresh base snapshot must reproduce **all ten generation records exactly**;
each development loss and supervised-token count must match the control, with
only **1e-6 absolute per-example loss tolerance**. A mismatch stops execution
before adapter creation or gradients. Actual tokenization for the new data has
not yet been GPU-tested.

The runtime then initializes new LoRA weights and verifies that all fresh B
matrices are zero before training. No previous adapter is loaded; the only
adapter load is the final reload of this run's own saved epoch-2 checkpoint.
The method, model revision, NF4/FP32 precision, batch size, accumulation, optimizer,
learning-rate schedule, dropout and decoding are unchanged from the control.
The original root `train.py` and its 1B configuration remain unchanged.

It saves adapters and evaluates at updates **23 and 46**. The complete run has
31 generations: ten at base, ten per epoch and one reload probe, plus eight
development losses at each snapshot. Gradient checks, changed adapter weights
and exact reloaded tensors are required. No larger budget, automatic retry,
extra seed, full training or reserved-test evaluation is authorized.

## Predeclared interpretation

Compare both new epochs with the original-input control at matched **23/46
updates**, and with their common base. Report development claim issues, generic
expression preferences and copying separately; do not choose an epoch solely
by lower loss. Keep all eight development cases and the two separately labeled
training probes, including negative results.

The control's final development loss was **1.89590**, with **six of eight**
material-issue outputs, six generic expression wins over base and zero exact
copies, empty outputs or length-limited outputs. A promising primary result
requires **fewer than six** material-issue outputs, at least six generic-expression
wins against the common base, and no increase in copying/empty/limited responses.
Also require at least as many direct generic-expression wins as losses against
the control's matched final checkpoint. Record ties and disclose every new
numeric, conditional or causal error. Passing the
counts on eight exposed cases would remain exploratory, not proof of style or
semantic reliability.

Use partly masked automated review of saved outputs, keeping the known limits
of the prior judgments explicit. Do not train on development generations or
replace difficult development inputs after seeing outcomes. The 45-example Fama
and 20-example multi-author reserves remain sealed. The source variants have
same-Codex-reviewer provisional validation and carry the original uncertainty
flags; no independent semantic gold or new PDF/OCR certification is claimed.

## Artifacts and validation

Download all of `input_replacement_pilot_v1/`: `evaluation.json`, `state.json`,
`label_audit.json`, `generation_config.json`, `gradient_checks.json`,
`training_log.json` and both epoch adapter directories. Preserve the emitted
runtime and notebook log. If execution fails, retain partial files for diagnosis
instead of automatically rerunning or raising the step cap.

The private packet includes `private_key/replacement_audit.private.jsonl`, with
all twelve source changes, unchanged-target hashes, original construction review
and curation flags. Thirty-four unselected training rows and all eight development
rows retain their original bytes. All 54 targets and the row sequence are exact.
Earlier data, experiment hashes, source files and model artifacts remain intact.

[experiment_plan.json](experiment_plan.json) pins the treatment and control.
[execution.json](execution.json) records the user's limited approval and exact
runtime/data/config/baseline hashes. The immutable packet retains its initial
authorization-false preparation field; the separate approval enables this one
run. Source wording is private. The notebook source contains code/hashes only,
has no saved outputs and stays below Kaggle's size limit. Credentials are neither
embedded nor printed: Kaggle retrieves HF_TOKEN from Secrets.

Reproduce the data packet into a fresh directory:

```sh
PYTHONPATH=src python3 -m fama_style.research.teacher_pair_input_replacement_packet \
  --control data/teacher_pair_expanded_experiment_v2 \
  --anchors data/teacher_pair_semantic_anchors_v1 \
  --baseline research_runs/teacher_pair_reverse_pilot_v2/kaggle_base_v1/teacher_pair_base_v2/evaluation.json \
  --baseline-review research_runs/teacher_pair_reverse_pilot_v2/base_review_v1 \
  --pilot-review research_runs/teacher_pair_reverse_pilot_v2/pilot_review_v1/results \
  --output /tmp/input-replacement-reproduction
python3 scripts/build_input_replacement_4b.py
PYTHONPATH=src python3 -m fama_style.training.teacher_pair_input_replacement \
  --data data/teacher_pair_input_replacement_v1/upload/teacher_pair_replacement.jsonl \
  --config data/teacher_pair_input_replacement_v1/upload/teacher_pair_replacement_config.json \
  --check
```

Mac CPU checks require no GPU, weights or tokenizer downloads. Tests cover
one-for-one text changes, target/order preservation, the control-baseline gate,
unchanged optimizer/LoRA constructor arguments, explicit approval and notebook
hash guards. No CLI upload, local training or automated Kaggle submission is
performed by these preparation tools.
