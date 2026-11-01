# Bounded SFT learning-rate check

The manually executed pilot is now complete; see
[results and limitations](SFT_STRENGTH_RESULTS_V1.md). The design and authorization
below record that completed run and do not authorize a repeat.

**Prepared and authorized for manual Kaggle execution; not executed by the agent.**
The user agreed to the stronger SFT-only follow-up after the
[training-side diagnostic](TRAINING_SIDE_RESULTS_V1.md). This is one fixed
intervention, not full training, another DPO experiment or a parameter search.

## Fixed design

| Setting | Completed weak SFT | This pilot |
| --- | --- | --- |
| Starting adapter | Original expanded-pilot epoch 2 | Same exact files |
| Training pairs / paragraph groups | 17 / 11 | Same bytes, masks and weights |
| Epochs / optimizer updates | 2 / 22 | 2 / 22 |
| Initial learning rate | `1e-6` | **`1e-5`** |
| Schedule | Linear decay, no warmup | Same; tenfold at every step |
| Objective | Response-token mean NLL, then sibling mean | Same |
| Dropout / seed / optimizer | Disabled / 1729 / fresh AdamW | Same |
| Model, quantization, precision | Pinned Gemma 3 4B, NF4, FP32 | Same |

The starting adapter is **not** the weak preference-pilot adapter. Use the original
`oliverlds/teacher-pair-expanded-pilot-4b`, version 1, `epoch_2/adapter/`.
Its file and loaded-tensor hashes must match the archived starting policy.
The root Gemma 1B entrypoint and all prior experiments remain unchanged.

The new private payload preserves the old data/config as exact UTF-8 strings;
the frozen prior CPU validator checks those original bytes without modifying
old files or module settings. The new runtime uses only chosen responses for
optimization. Rejected responses remain part of the historical artifact and
are never used for a preference loss. The prior runtime is embedded solely to
reuse its verified validators and pure helpers; its training function is not called.

Before updates, require 25 exact processor/response audits (17 training targets
and eight development references), agreement of manual and built-in losses,
17 reproduced initial chosen likelihoods, and eight exactly reproduced control
generations. Train only the original 136 F32 language q/v LoRA tensors, using
the same group order and normalization. Stop on nonfinite/zero gradients or a
failed checkpoint check. The twenty-two gradients and updates are recorded.

Save epoch 1 for provenance only; perform no epoch-1 generation or checkpoint
selection. Save and reload the fixed final epoch-2 adapter, check bitwise tensor
equality, freeze all parameters, then evaluate it.

## Evaluation fixed before execution

Generate **19 final outputs**: the same eleven seen training prompts and the
same eight milestone development prompts. Also record initial/final target
likelihoods for all seventeen training pairs and eight development cases.
Together with eight startup control checks, this schedules **27 new generations**.
There is no additional initial training-side generation: those controls were
already verified and archived.

The local private comparison key freezes 38 existing responses: original control
and weak SFT for each of the nineteen cases. These baseline responses stay out
of the generation prompts. References are separate training/scoring targets;
the generator receives only the unchanged user-prompt prefix.

Review meaning and expression separately with masked labels and the same local
one-minor/no-material budget. On seen cases, count targeted errors resolved,
persistent and newly introduced errors, locally eligible outputs, exact control
repetitions, source copying and shared spans. On development, retain the existing
quality gate: at most three material-issue outputs, at least four locally eligible
outputs, generic-expression wins exceeding losses versus original control, and
no exact source copies or empty/token-limited outputs. Compare directly with
weak SFT as well. Do not treat loss reduction or training-case copying as success.

The small, error-selected training set is not a representative accuracy benchmark;
the development examples have been reused. This can diagnose behavior under
stronger optimization, not prove generalization, distinctive Fama style, human
authorship or reduced detectability. No reserved-test access or detector scans.
No result is reported before execution.

## Manual Kaggle steps

1. Upload `kaggle/sft_strength_4b.ipynb` as a **new private notebook**. Training is
   enabled only for this bounded pilot.
2. Create/attach a private Dataset containing both files from
   `data/sft_strength_v1/upload/`: `sft_strength_data.json` and
   `sft_strength_config.json`. Do not upload `private_key/`.
3. Attach notebook outputs of **`oliverlds/teacher-pair-expanded-pilot-4b`, version 1**,
   including its original `epoch_2/adapter/`. No weak SFT/DPO adapters are needed.
4. Select **GPU T4 x2**, enable Internet and **HF_TOKEN** in Secrets, then
   **Save & Run All** in a fresh session. GPU 0 alone is exposed.
5. Retain the full `sft_strength_v1/` output directory and notebook log, including
   partial output if a gate fails. Send the notebook address and version.

The total wall-clock cap is **30 minutes**, beginning in the first code cell and
including dependency setup. Recorded Torch `2.11.0+cu128` and the previous pinned
Transformers/PEFT environment are required. The earlier comparable pipeline fits
one T4, but this new run's runtime, memory and quality are unmeasured. Do not
silently change settings or retry beyond the approved budget.

HF_TOKEN is retrieved in memory through Kaggle Secrets, not printed, written to
the packet or persisted through a login call. This notebook is self-contained
and has no saved outputs. No automatic upload, submission or training occurred
during preparation. Authorization is limited to this single run; full training,
DPO continuation and repeated searches are not authorized.

## CPU reproducibility

- Runtime: `src/fama_style/training/sft_strength_pilot.py`.
- Packet builder: `src/fama_style/research/sft_strength_packet.py`.
- Notebook builder: `scripts/build_sft_strength_pilot.py`.
- Private packet: `data/sft_strength_v1/`, including authorization, plan,
  comparison key and hashed receipt/manifest.

All seven packet files reproduce byte-for-byte in a fresh directory. The real
payload and source adapter pass `--check` without ML imports, weights or GPU use.
Tests cover the tenfold schedule, frozen originals and grouping, response-prefix
boundaries, explicit execution flag, absence of calls into old DPO/training code,
embedded source equality and bounded authorization. The original pre-change
suite and root training dry run passed before training logic was added.
