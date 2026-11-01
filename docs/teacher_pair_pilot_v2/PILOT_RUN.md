# Run the authorized expanded teacher-pair pilot

This authorized run has completed as `oliverlds/teacher-pair-expanded-pilot-4b`
version 1. See [results and limits](PILOT_RESULTS.md). The instructions below
preserve the executed setup; no additional run is authorized.

The user authorized **one two-epoch / 46-update pilot** after review of
`oliverlds/teacher-pair-expanded-base-4b` version 1. This authorization does not
include full training, automatic retries, a larger budget or other experiments.
The notebook was enabled for manual execution and the user subsequently ran it.

1. Import `kaggle/teacher_pair_expanded_pilot_4b.ipynb` as a **new private Kaggle
   notebook**. Keep the completed baseline notebook unchanged.
2. Attach the **same private expanded Dataset used for that baseline**. It must
   contain `teacher_pair_expanded.jsonl` and `teacher_pair_expanded_config.json`.
   The local originals are in `data/teacher_pair_expanded_experiment_v2/upload/`;
   they have not changed. Do not attach the earlier 16-pair Dataset.
3. Attach the **version 1 notebook outputs from
   `oliverlds/teacher-pair-expanded-base-4b`**. The mounted output must contain
   `teacher_pair_base_v2/evaluation.json`. Attach actual output files, not just
   notebook source, and avoid multiple copies of the baseline output.
4. Enable **HF_TOKEN in Kaggle Secrets**, **Internet** and **GPU T4 x2**. Use a
   fresh session. The first cell exposes GPU 0 only, before importing Torch.
5. Select **Save & Run All**. `RUN_PILOT=True` for this approved scope; no manual
   switch edit is needed. Send the resulting notebook slug and version.

The launch cell validates the exact data, config and reviewed baseline report
hashes before starting the model process. The unchanged runtime independently
checks the pinned model revision, fixed settings, baseline coverage and all 54
token/target-label boundaries. The model is `google/gemma-3-4b-it`, using the same
NF4/FP32 compute and language-only rank-8 q/v LoRA as the earlier successful
pilot. The original root `train.py`/1B workflow remains unchanged.

The run produces a fresh base snapshot and epoch snapshots at updates **23 and
46**, each with eight development generations/losses and two old training
probes. It saves both adapters, checks all adapter weights after disk reload,
and generates one reload probe. Keep training-probe findings separate from the
eight development cases. No reserved-test examples or detector calls are used.

Download the entire `teacher_pair_pilot_v2/` directory, including `state.json`,
`evaluation.json`, `label_audit.json`, `generation_config.json`,
`gradient_checks.json`, `training_log.json` and both epoch adapter directories.
Preserve the emitted runtime source and notebook log as well. All raw text,
outputs and adapters remain private. If execution fails, retain partial files
for diagnosis; do not automatically rerun or increase the cap.

The earlier 16-pair pilot had 16 updates; this pilot has more data and a larger
optimization budget, so differences cannot be attributed solely to data volume.
Keep the existing [evaluation criteria](EXPERIMENT.md#evaluation-and-decision-rule)
and compare both epochs, claim drift, expression/style and copying separately.
A lower loss or a successful run alone does not establish improved fidelity,
Fama-specific style, human authorship or detector performance.

[pilot_execution.json](pilot_execution.json) records the approved scope and
frozen hashes without credentials or source paragraphs. The immutable packet
and preparation plan retain their original authorization-false fields as
historical preparation records; this session authorization supersedes their
execution instructions. The disabled notebook and authorization are archived
privately under `research_runs/teacher_pair_reverse_pilot_v2/pilot_execution_v1/`.
No CLI push, automatic Kaggle run or local model execution was performed.
