# Training-side free-generation diagnostic

The diagnostic has now completed; see [results and limitations](TRAINING_SIDE_RESULTS_V1.md).
The specification below preserves the setup used for that run.

**Prepared, not executed. Inference only; zero optimizer steps.**
The [completed pilot](PILOT_RESULTS_V1.md) changed weights and likelihoods, but
did not improve its eight development generations. This diagnostic asks whether
its final adapters change behavior on the actual training examples before we
consider further optimization.

## Frozen comparison

- Eleven paragraphs across ten papers, one included preference row per original
  paragraph group. Prefer its original input when included; otherwise use its
  included variant. Selection is sorted and independent of new outputs.
- Nine original inputs and two variants. Error families: three scope, three
  numbers, three reasoning and two time examples. They are not equally balanced.
- Eleven unchanged-adapter responses reused from the earlier semantic-anchor
  inference run. Each is byte-identical to the selected actual rejection, with
  matching source, prompt, target hash, model revision, adapter and environment.
- Eleven new responses from final SFT and eleven from final DPO: **22 new
  generations, 33 total comparison outputs**. Both checkpoints are fixed epoch 2.

Keep the same ordinary instruction, greedy decoding, 512-new-token cap, pinned
Gemma 3 4B revision, NF4/FP32 compute, package versions and one T4 as the completed
pilot. The generation prefixes are taken from the verified training token
sequences and must match the actual processor before model loading. The largest
prompt is 275 tokens; none requires truncation within the 1,024-token limit.

There is no optimizer, backward pass, teacher-forced scoring, adapter saving or
new preference construction. Both adapters are loaded frozen into the same base
model, one at a time, with unloading rather than merging between conditions.
File hashes are checked before and after inference. The thirty-minute parent
timeout includes dependency setup; GPU memory/runtime remain unmeasured for this
new notebook. Credentials are read only from Kaggle Secrets, never printed or
stored by the code.

## Files and running in Kaggle

1. Upload `kaggle/preference_training_inference_4b.ipynb` as a **new private notebook**.
2. Create/attach a private Dataset containing just these two files from
   `data/preference_training_inference_v1/upload/`:
   `preference_training_jobs.jsonl` and `preference_training_config.json`.
3. Attach notebook outputs of `oliverlds/preference-pair-pilot-4b`, version 1,
   including both `sft/epoch_2/adapter/` and `dpo/epoch_2/adapter/`.
4. Select **GPU T4 x2**, enable Internet and **HF_TOKEN** in Secrets, then
   **Save & Run All** in a fresh session. GPU 0 alone is exposed.
5. Keep the complete `preference_training_inference_v1/` output directory and log,
   including partial outputs on failure. Send the notebook address and version.

Do not attach the original pilot's training Dataset or this packet's `private_key/`.
The new upload contains only source prompts and prefix IDs. The human targets and
archived control responses remain in the local private evaluation key. Ordinary
source/reference wording overlap remains a property of the task; no separate
reference completion is supplied to generation.

CPU runtime check:

```sh
PYTHONPATH=src python3 -m fama_style.evaluation.preference_training_inference \
  --jobs data/preference_training_inference_v1/upload/preference_training_jobs.jsonl \
  --config data/preference_training_inference_v1/upload/preference_training_config.json \
  --check
```

Builders are `src/fama_style/research/preference_training_inference_packet.py` and
`scripts/build_preference_training_inference.py`. The packet includes a plan,
private reference key, receipt and manifest. All six files reproduce byte-for-byte
in a fresh directory. CPU checks validate both real saved adapter hashes without
loading model weights. The notebook is self-contained, below Kaggle's source
limit, and has no saved cell outputs. No upload or GPU job was initiated by the agent.

## Assessment after execution

First verify 22 complete generations, eleven matching token audits, exact adapter
and decoding hashes, all parameters frozen, and zero optimizer steps. Then join
the eleven cached controls and assess paired outputs with masked condition labels.
Keep meaning and generic expression separate, using the existing one-minor/no-material
meaning budget. Record exact changes from the control, resolved versus persistent
material issues, new errors, source copying/long shared spans, empty/truncated
outputs and generic-expression preferences. Report by paragraph and error family;
do not substitute training likelihood for generation quality.

Little training-side change would make a separately authorized optimization-strength
check more informative than further detector scans. Training-side improvement
without development improvement would instead suggest limited transfer or data
coverage. Neither observation uniquely diagnoses a cause. No numerical success
claim or new result is reported before execution, and no subsequent training is
authorized by this diagnostic.

These are seen, error-selected training cases, not held-out evaluation. The frozen
prior pilot packet's eight development rows are parsed only to validate its
existing contract; none is used to select these prompts, and no development
generation is scheduled. No reserved-test data are accessed. Automated judgments
remain provisional; this check cannot establish generalization, reduced
detectability, human authorship or distinctive Fama style. No detector calls.
