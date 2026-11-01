# Matched DPO at the stronger update setting

The manually executed pilot is now complete; see
[results and limitations](DPO_STRENGTH_RESULTS_V1.md). The preparation and approval
below record that completed run and do not authorize a repeat.

**Prepared and authorized for one manual Kaggle run; not executed by the agent.**
The [stronger SFT result](SFT_STRENGTH_RESULTS_V1.md) changes behavior and repairs
some claims, but still fails the development quality gate. This comparison tests
whether explicit contrast with actual erroneous responses helps at that same
update setting. No result is claimed before execution.

## Fixed comparison

- Same seventeen chosen/actual-rejected pairs, eleven paragraph groups and
  two epochs: **22 grouped optimizer updates**, learning rate **`1e-5`**,
  seed 1729, disabled dropout and the same linear decay and fresh AdamW settings.
- Same original expanded-pilot epoch-2 starting adapter. **Do not initialize
  from stronger SFT or the weak DPO adapter.**
- Standard sigmoid DPO, **beta `0.1`**, using summed response log probabilities,
  then averaging sibling-row losses within each group.
- Reference policy is the exact unchanged original adapter, not raw Gemma or
  a moving copy of the updated policy. Cache 34 reference scores before updates.
- Same pinned Gemma 3 4B revision, NF4 double quantization, FP32 compute,
  language q/v rank-8 adapter, package versions and greedy decoding.

The upload data are byte-identical to the stronger SFT payload; only the new
configuration and runtime differ. Prior data, checkpoints, hashes and results
are preserved. The root Gemma 1B training entrypoint remains unchanged.

The runtime is an isolated counterpart derived from the exact completed SFT
source through `scripts/build_dpo_strength_runtime.py`. It preserves model loading,
causal response scoring, adapter checks, grouped updates, saving/reloading and
evaluation. The derivation refuses a changed source hash and is tested to
reproduce the new runtime byte-for-byte. Neither old training entrypoint is called.

This retains the existing explicit PyTorch/PEFT implementation and tested
sequential DPO derivatives, rather than adding a new TRL release. Equal updates
and learning rate do not equalize gradient scale or compute: DPO processes actual
rejections as well as chosen responses. No SFT retraining or parameter search.

## Gates and memory control

Before optimization require:

1. Forty-two exact response encodings/masks: seventeen chosen, seventeen actual
   rejected and eight development references.
2. Manual versus built-in causal loss agreement on chosen, rejected and development
   completions; exact reproduction of the eight archived control outputs.
3. All 34 reference scores reproduced against the earlier original-policy cache,
   with unchanged adapter tensor fingerprint. Cache is bound to dataset, model,
   revision and source adapter hashes.
4. Policy/reference equality in training mode with dropout disabled, and initial
   DPO loss approximately `log(2)` on every pair.

For each DPO row, obtain current chosen/rejected scores without gradients, then
recompute and backpropagate each response separately using the detached sigmoid
derivative. Parameters stay fixed until the whole paragraph group's gradients
are accumulated. This avoids retaining two response graphs. The actual gradient
and no-gradient scores must agree; stop if they differ or the reference cache
changes. Finite nonzero gradients, exact group/order coverage and all 22 updates
are recorded. Existing CPU finite-difference tests cover these loss derivatives.

Epoch 1 is saved only. Epoch 2 is fixed as primary, saved, reloaded and checked
for bitwise adapter equality before frozen evaluation. Record initial/final
target scores, final chosen/rejected preference scores, forward/token counts,
runtime and peak GPU allocation. No other checkpoint selection.

## Final evaluation and interpretation

Generate the same **nineteen final outputs**: eleven selected seen-training inputs
and eight reused development inputs. With eight startup control checks this is
**27 new generations**. Thirty-eight archived original-control and stronger-SFT
responses are frozen in a separate local private comparison key. Generation
uses only the unchanged user-prompt prefix, never the reference/negative text.

Assess original control, stronger SFT and stronger DPO in masked order. Keep
material errors, minor shifts, readability/expression, copying and truncation
separate. Preserve earlier semantic boundaries, including unsupported exclusive
purpose claims. Count repairs and new errors rather than only net error totals.

The existing development gate remains at most three material-issue outputs,
at least four outputs within the one-minor/no-material budget, generic-expression
wins exceeding losses versus original control, and zero exact copies or empty/
token-limited outputs. A DPO-specific advantage requires a direct benefit over
stronger SFT; improved margins alone are insufficient. No detector scans are
part of this pilot, and no Fama-specific or human-authorship success is assumed.

## Manual Kaggle steps

1. Upload `kaggle/dpo_strength_4b.ipynb` as a **new private notebook**.
2. Create/attach a private Dataset with both files from
   `data/dpo_strength_v1/upload/`: `dpo_strength_data.json` and
   `dpo_strength_config.json`. Do not attach/publish `private_key/`.
3. Attach notebook OUTPUTS of **`oliverlds/teacher-pair-expanded-pilot-4b`, version 1**,
   including the original `epoch_2/adapter/`. Stronger SFT outputs are not needed
   in Kaggle; they are cached locally for review.
4. Select **GPU T4 x2**, enable Internet and **HF_TOKEN** in Secrets, and
   **Save & Run All** in a fresh session. Only GPU 0 is exposed.
5. Preserve the complete `dpo_strength_v1/` directory and notebook log, including
   partial artifacts on failure. Send the notebook address and version afterward.

Training is enabled for this single approved DPO-only pilot, with a **30-minute
total wall cap including setup**. It requires recorded Torch `2.11.0+cu128` and
the previous pinned environment. Earlier related runs fit one T4; this new
run's time, memory and results remain unmeasured. Stop on errors rather than
silently changing settings or retrying beyond the authorized budget.

HF_TOKEN is read into memory from Kaggle Secrets, never hard-coded, printed,
stored in the packet or persisted by a login call. No credentials, copyrighted
reference paragraphs or private model outputs are distributed publicly.

## CPU preparation and limits

Runtime: `src/fama_style/training/dpo_strength_pilot.py`.
Packet builder: `src/fama_style/research/dpo_strength_packet.py`.
Notebook builder: `scripts/build_dpo_strength_pilot.py`.
The private packet contains authorization, plan, frozen comparisons and a hashed
receipt/manifest. All seven files reproduce byte-for-byte in a fresh directory;
the real data and original adapter pass `--check` without ML imports or weights.
New tests cover reference-score validation, matched scheduling, frozen-source
derivation, explicit execution flags and bounded notebook authorization.

This remains one seed on seventeen correlated, error-selected pairs. Chosen
references have greater source overlap in 16/17 pairs and are longer in 12/17.
The eleven seen cases are not an accuracy benchmark; the eight development cases
are reused. Automated judgments are provisional. No reserved examples, detector
calls, automatic upload or GPU job submission occurred during preparation. No
full training or further optimization is authorized by this run.
