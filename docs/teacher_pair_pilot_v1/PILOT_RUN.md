# Run the authorized limited QLoRA pilot

This authorized run has completed as `oliverlds/teacher-pair-pilot-4b` version 1.
See [results and limits](PILOT_RESULTS.md). The instructions below preserve the
executed setup; no additional run or extended budget is authorized by the review.

The user authorized this previously prepared pilot on 31 October 2026, after
review of `oliverlds/teacher-pair-base-4b` version 1. Scope is **two epochs and
sixteen optimizer updates**, on the same 16 training pairs and eight development
pairs. No full training, data expansion or hyperparameter search is authorized.
The notebook was enabled for manual execution under that scope.

1. Import `kaggle/teacher_pair_pilot_4b.ipynb` as a **new private Kaggle notebook**.
2. Attach the **existing private Teacher Pair Reverse Pilot Dataset** used in the
   base run. The two original JSON files are unchanged; do not create a revised
   dataset or attach other pilot datasets.
3. Attach the **notebook outputs from `oliverlds/teacher-pair-base-4b`, version 1**.
   The mounted output must include `teacher_pair_base_v1/evaluation.json`.
   Attach the output files, not the notebook source alone.
4. Enable **HF_TOKEN in Secrets**, **Internet**, and **GPU T4 x2**. Start a fresh
   session. The first code cell exposes only GPU 0.
5. **Save & Run All**. No code editing or switch change is needed: `RUN_PILOT`
   is now true for this approved scope. Provide the new notebook's actual URL
   slug and version when it finishes.

The notebook verifies the SHA256 of the data, configuration and exact reviewed
baseline report before the GPU process starts. This prevents accidentally using
a later baseline or different dataset. The runtime independently validates the
model revision, fixed hyperparameters, baseline coverage and target-only labels.
It recomputes a base snapshot, then saves adapters and outputs at epochs 1 and 2.
A disk reload checks all adapter tensors and saves one development probe.

Download the entire `teacher_pair_pilot_v1` output folder, including
`evaluation.json`, `state.json`, `label_audit.json`, `gradient_checks.json`,
`generation_config.json`, `training_log.json`, and both epoch adapter folders.
Keep all generated text and model artifacts private. Preserve partial outputs
if the run fails; do not automatically rerun or increase the step cap.

Interpret development outputs separately from the two training probes. A lower
loss or successful training run does not establish preserved meaning or improved
style. Compare base, epoch 1 and epoch 2 under the already fixed claim/style/copying
criteria; retain negative results and do not select better-looking cases.

[pilot_execution.json](pilot_execution.json) records the approval scope and
checksums without source paragraphs or credentials. The immutable data packet's
initial authorization flags and `START_HERE.md` describe preparation time; this
run instruction supersedes those initial execution instructions. Model/data text,
hyperparameters and the old baseline remain frozen. The prior disabled notebook
is archived privately under
`research_runs/teacher_pair_reverse_pilot_v1/pilot_execution_v1/`.

No CLI push, automatic Kaggle run, local model execution or training was performed
by the enabling tools. The user subsequently executed the pilot on Kaggle.
