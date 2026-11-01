# Bounded preference-versus-SFT implementation

The manually executed pilot has now completed; see
[verified results and limitations](PILOT_RESULTS_V1.md). The setup and authorization
details below record how that completed run was prepared; they do not authorize a
repeat or further training.

**Approved for the bounded pilot; not executed by the agent.** The user explicitly
approved the 44-update, one-T4, 60-minute pilot. Approval is retained separately in
`research_runs/actual_error_pilot_v1/execution_approval_v1.json`, bound to the frozen
packet manifest. The prepared notebook now has `RUN_APPROVED_PILOT=True`.
Full training is not authorized. The CPU tokenizer
gate passed previously; this extension implements the
[frozen pilot design](PILOT_PROPOSAL.md). It reports no new model results.
The original 1B entrypoint, datasets, experiments and results are preserved.

## What will be compared

| Condition | Starting weights | Additional updates |
| --- | --- | --- |
| Control | Original expanded-pilot epoch-2 adapter | 0 |
| Chosen-only SFT | Independently reset to that same adapter | 22 |
| DPO | Independently reset to that same adapter | 22 |

Seventeen actual-error pairs cover eleven training paragraphs and ten papers.
Each original-paragraph group receives one update per epoch, averaging its
one or two sibling rows. Both arms use the same two group permutations, fresh
AdamW state, learning rate `1e-6`, seed 1729 and disabled dropout. The linear
schedule decays over 22 updates, without warmup. SFT averages response-token
NLL per row; DPO uses summed response log probabilities with beta `0.1`.
Equal updates and learning rates do not equalize gradient scales or compute.

The model is pinned `google/gemma-3-4b-it`, revision
`093f9f388b31de276ce2de164bdc2081324b9767`, with NF4 double quantization, FP32
compute, SDPA, and the original rank-8 language q/v adapter. The starting
adapter is from `oliverlds/teacher-pair-expanded-pilot-4b`, version 1,
`epoch_2/adapter/`; the input-replacement adapter is a different experiment.
Both source adapter files must match their archived hashes.

The primary checkpoints are epoch 2. Epoch 1 is diagnostic; no best-epoch
selection is planned. Forty generations are scheduled: eight unchanged-control
outputs and eight for each of the four arm/epoch combinations. These use the
original milestone development subset, not the later teacher-pair eight-case
subset. Reference paragraphs are used for likelihood evaluation, never included
in generation prompts. Unchanged input remains a separate analysis baseline.
No reserved examples were read when building this packet.

## Explicit training loop and runtime gates

The implementation uses a small grouped PyTorch/PEFT loop rather than adding a
TRL release to the previously recorded environment. The current TRL reference
behavior and STAMP were reviewed in the proposal; this implementation does not
claim tested compatibility with a TRL trainer. This choice preserves the verified
IDs, response masks, group updates and initial-adapter reference policy directly.

Each DPO row first obtains its two policy scores without gradients. It then
recomputes chosen and rejected responses separately and backpropagates each
score multiplied by the detached derivative of the sigmoid DPO loss. With
unchanged parameters and deterministic forwards, these are the chain-rule
gradients of the joint loss. A CPU finite-difference test checks the derivatives,
including shared parameters and sibling averaging. Actual forwards must also
agree between the no-gradient and gradient passes; otherwise execution stops.
This avoids retaining two response graphs, at the cost of extra forwards.

Before any optimizer updates, the notebook must pass:

1. Exact package, model revision, adapter and one-visible-T4 checks.
2. Forty-two processor/tokenizer equivalence checks: 34 training responses
   plus eight development references. No truncation or row removal is allowed.
3. Manual causal response likelihood versus the model's built-in loss on a
   training response and development reference.
4. Exact reproduction of all eight archived control generations.
5. Immutable caching of 34 reference scores from the unchanged starting adapter.
   Dataset/token/model/adapter hashes bind these scores to the frozen sequences.
6. Initial policy/reference equality and approximately `log(2)` DPO loss for
   both independently initialized arms.

During training, stop on nonfinite or zero gradient norms, nondeterministic
two-pass scores, changed reference cache or incorrect update coverage. Each arm
must change adapter weights, save checkpoints and reload the final adapter with
bitwise-equal adapter tensors. Record losses, outputs, gradients, learning rates,
forward/token counts, elapsed time and peak allocated GPU memory.

GPU fit and model-level gates remain **unmeasured**. The proposal's 20–40 minute
estimate is a planning estimate, not a benchmark. A parent-process timeout caps
the notebook at one hour from its first code cell, including dependency setup.
There is no automatic retry or silent reduction of the experiment.

## Files and CPU verification

- Runtime: `src/fama_style/training/preference_pair_pilot.py`.
- CPU packet builder: `src/fama_style/research/preference_pilot_packet.py`.
- Notebook builder: `scripts/build_preference_pair_pilot.py`.
- Prepared notebook: `kaggle/preference_pair_pilot_4b.ipynb`.
- Private frozen payload: `data/actual_error_pilot_v1/upload/` (two JSON files).
- Private plan, receipt and manifest: `data/actual_error_pilot_v1/`.

The packet reuses the successful real-tokenizer recovery without modifying the
seventeen pairs. All eight development prefixes reproduce archived token hashes.
All five packet files reproduced byte-for-byte in a fresh temporary directory.
The runtime's real-packet `--check` also verifies the original adapter hashes,
without importing ML packages or loading weights. Unit tests cover objective
derivatives, grouped exposure, masks, corrupted inputs, split isolation,
reference-free generation messages, authorization and notebook defaults.

To reproduce locally, use the existing isolated tokenizer environment and
exported tokenizer bundle, then build the notebook. Do not use a new output
directory as a way to change the frozen experimental settings silently.

```sh
PYTHONPATH=src HF_HUB_OFFLINE=1 USE_TORCH=0 USE_TF=0 USE_FLAX=0 \
  /tmp/fama-pref-tokenizer-venv/bin/python \
  -m fama_style.research.preference_pilot_packet --output /tmp/new-preference-packet
python3 scripts/build_preference_pair_pilot.py
```

The temporary environment is local convenience, not a distributed dependency.
Its tokenizer stack uses Transformers 5.20.1, huggingface_hub 1.33.0,
tokenizers 0.23.3, Jinja2 3.1.6 and sentencepiece 0.2.2. The private receipt binds
all exported tokenizer files. No model weights are required for this preparation.

## Manual Kaggle use after the execution decision

Create a **new private** notebook and private Dataset containing only
`actual_error_pilot_data.json` and `actual_error_pilot_config.json` from the upload
folder. Upload the prepared notebook through the UI. Its default
builder default `RUN_APPROVED_PILOT=False` performs checks only, needs no
accelerator, installs no packages and does not retrieve Secrets or load a model.
The current notebook was rebuilt with the explicit approval record and has
`RUN_APPROVED_PILOT=True`; it is ready for manual GPU execution.

Approval has been given: attach the original expanded-pilot version-1 outputs,
select **GPU T4 x2**, enable
Internet and enable **HF_TOKEN** in Kaggle Secrets. GPU 0 alone is exposed.
The notebook checks the existing Torch version and installs the recorded
Transformers/PEFT/Accelerate/bitsandbytes/hub versions. It stops if the recorded
Torch `2.11.0+cu128` environment is unavailable. HF_TOKEN is read into memory from
Secrets; it is not printed, hard-coded, stored in the packet or persisted by a
login call. The agent has not uploaded, submitted or run this notebook.

After an approved run, retain the log and entire `actual_error_pilot_v1/` output
directory, including partial outputs if a gate fails. An `evaluation.json` file
can be partial: completion additionally requires `state.json.complete=true`,
44 updates, five completed snapshots and both adapter reload checks.

## Decision remains about generation, not training loss

Use the proposal's unchanged quality gate: at most three material-issue outputs
and at least four locally eligible outputs out of eight, paired generic-expression
wins exceeding losses versus control, and no exact input copies, empty or
token-limited outputs. A DPO-specific benefit also requires an advantage over
matched SFT. No detector scans are part of this run. Only a quality-qualified,
frozen checkpoint is eligible for a later sparse detector decision.

Chosen/input overlap is higher in 16/17 pairs and chosen responses are longer
in 12/17. The tiny, correlated sample and provisional semantic judgments remain
limitations. Better target likelihood or preference margins alone cannot
establish better rewriting, reduced detectability or distinctive Fama style.
