# Expanded teacher-pair follow-up: baseline first

Prepared on 31 October 2026. The [new baseline has completed](BASELINE_RESULTS.md)
as `oliverlds/teacher-pair-expanded-base-4b` version 1; all 54 token/label checks
pass, and all ten outputs and eight loss rows reproduce v1 exactly. The pilot
has now [completed](PILOT_RESULTS.md) as
`oliverlds/teacher-pair-expanded-pilot-4b` version 1. Loss falls 36.15% from base,
but six of eight development outputs retain material claim issues at both epochs;
the predeclared success gate fails. The frozen v1 experiment remains unchanged.
The specification below preserves the bounded experiment; it is not authority
for a rerun or full training.

## Question and budget

Does a practical expansion from 16 to 46 provisionally aligned teacher pairs
improve meaning-preserving academic rewriting on the same eight development
examples? Preserve the two old training probes. Inputs, targets, ordinary rewrite
instruction, pinned Gemma 3 4B revision, NF4/FP32 compute, language-only q/v LoRA,
rank 8, alpha 16, dropout 0.05, batch one, accumulation two, seed 1729 and greedy
512-token generation match v1.

The prepared budget is **two epochs / 46 optimizer updates**, with base evaluation
at zero and adapter snapshots at **23 and 46 updates**. Learning rate remains
2e-5 with a linear schedule over the new budget. Each snapshot has eight
development generations/losses and two training probes; a final disk reload adds
one development generation and verifies all adapter tensors. The baseline has
ten generations; the pilot has 31 including the reload probe.

Both dataset size and optimization budget change from v1's 16 updates. The
longer linear schedule also changes the learning-rate trajectory by update.
This is a practical improvement experiment, **not** a causal estimate of data
quantity. A step-matched experiment would be needed for that claim. No search,
automatic extension, extra seed or reserved-test evaluation is included.

The [preparation record](experiment_plan.json) pins data, config, runtime and
curation hashes. It records training authorization as false. Preparation and
local CPU checks did not authorize a GPU training job. Subsequent user approval
is recorded separately in [pilot_execution.json](pilot_execution.json), without
altering the frozen preparation record.

## Manual Kaggle handoff

1. Create a **new private Dataset**, for example “Teacher Pair Expanded V2”.
   Upload only these two files from `data/teacher_pair_expanded_experiment_v2/upload/`:
   `teacher_pair_expanded.jsonl` and `teacher_pair_expanded_config.json`.
   Keep the old v1 Dataset and notebooks intact; do not upload audit files.
2. Import `kaggle/teacher_pair_expanded_base_4b.ipynb` as a **new private notebook**.
   Attach only the new Dataset. Enable **HF_TOKEN in Kaggle Secrets**, Internet
   and **GPU T4 x2**. The first cell exposes GPU 0 only. Run in a fresh session.
3. Save & Run All. This baseline performs **no training**. It checks frozen file
   hashes, all 46 training and eight development target-label boundaries, full
   sequence lengths and generation context budget before loading weights. There
   is no truncation or silent row removal. If a row fails, retain the partial
   `state.json` and `label_audit.json`; state identifies its ID and measured token
   lengths without printing source text or credentials.
4. Provide the actual notebook slug and version, or download the entire
   `teacher_pair_base_v2/` output directory. Keep `evaluation.json`, `state.json`,
   `label_audit.json`, `generation_config.json` and the emitted runtime source.
   Review completion, all 54 label checks, losses and generations before training.
5. The baseline is now reviewed and the user has authorized the limited pilot.
   Follow [PILOT_RUN.md](PILOT_RUN.md) for the enabled notebook and exact version-1
   baseline output attachment. Its hash is pinned before model execution.

Kaggle Secrets is the only token source on Kaggle. The runtime neither writes
the token into output metadata nor prints its value. No CLI upload, submission,
local model inference or GPU execution is performed by the preparation tools.

## Evaluation and decision rule

Assess all eight development cases separately from the two training probes.
Keep claim drift, expression/style preferences and copying as separate outcomes.
Record omissions, additions, changed actors, quantities, comparisons, conditions,
negation, modality, attribution and causal direction. Distinguish material changes
from wording choices; do not penalize every paraphrase or equate overlap with
faithfulness. Report truncation, commentary and empty outputs separately.

Compare the new base, epoch 1 and epoch 2 under masked condition labels, retaining
ties and all cases. The reviewer has already seen earlier outputs, so this is
automated, partly masked assessment, not independent human evaluation. Report
token-weighted development loss, exact source copies, source/reference overlap,
and per-paper generic organization/expression preferences. Compare with v1's
epoch 2 on the identical eight cases: six material-issue outputs, zero exact
copies, and five generic style preferences over its base. New baseline results
must also be compared against the archived v1 base for reproducibility; disclose
differences before attributing gains to training.

A promising outcome requires fewer material-issue development outputs than v1's
six, with at least as many generic-style wins against the corresponding base as
v1's five, and no increase in exact copies, empty or length-limited outputs.
Report any new material numeric, condition or causal errors even if aggregate
counts improve. Mixed results remain mixed. Both epochs must be reported; loss
alone cannot choose a winner. These exploratory gates on eight reused cases do
not establish significance, training readiness, a Fama signature, human
authorship or a detector improvement. No detector queries are needed here.

## Runtime, costs and reproduction

The 4B runtime is a separate fork of the executed v1 implementation because v1's
source hash and notebook are frozen. It changes the protocol, 46-pair count,
23/46-step boundaries, output names and baseline preflight guards. The root
`train.py` and its original `google/gemma-3-1b-it` model remain untouched.
All ML imports remain within `run_training`; Mac CPU checks need no weights.
Dependencies use the same pinned versions as the completed v1 notebooks.

V1 took about 219 seconds for baseline and 607 seconds for its complete pilot.
The new pilot has 2.875 times as many optimizer updates, with the same number
of generation snapshots; wall time cannot be scaled directly from either number.
Plan for a longer T4 session. Exact time, token lengths, memory and costs remain
unmeasured. Memory settings and maximum length are unchanged; longer new rows
may still fail preflight or require more memory. Stop and diagnose a failure
without automatic retries or increasing the cap.

Build into a fresh output directory (existing releases refuse overwrite):

```sh
PYTHONPATH=src python3 -m fama_style.research.teacher_pair_expanded_packet \
  --curated data/teacher_pair_pilot_v2 \
  --output /tmp/teacher-pair-expanded-v2-reproduction
python3 scripts/build_teacher_pair_expanded_pilot.py
PYTHONPATH=src python3 -m fama_style.training.teacher_pair_expanded_pilot \
  --data data/teacher_pair_expanded_experiment_v2/upload/teacher_pair_expanded.jsonl \
  --config data/teacher_pair_expanded_experiment_v2/upload/teacher_pair_expanded_config.json \
  --check
```

The notebook builder reads the frozen local packet and embeds the runtime and
file hashes. It does not submit anything. Dataset text, audits, raw generations
and adapters remain private; public documentation contains only aggregate plans
and hashes. No new experiment results are reported here.
