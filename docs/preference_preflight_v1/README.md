# Preference token preflight — CPU only

**Current status: [real CPU tokenizer gate passed after local recovery](RUN_V2_RECOVERY.md).**
Version 2 exported the tokenizer and then hit a structured-return bug. The code
is fixed and all seventeen pairs / 34 response sequences now pass locally,
including seventeen exact archived token-ID matches. No Kaggle rerun is needed.

The original packet was prepared without a local tokenizer or HF token; gated
assets were obtained through Kaggle Secrets and are now available locally. The official
[pinned repository](https://huggingface.co/google/gemma-3-4b-it/tree/093f9f388b31de276ce2de164bdc2081324b9767)
requires authenticated acceptance for its files. No alternate tokenizer or model
revision is substituted.

This small private notebook supports the tokenizer gate without GPU time or model
weights. It downloads only six pinned tokenizer/config files, then exports those
assets for subsequent local checks without a local HF credential. No training
pilot is authorized or submitted. The original candidate packet remains frozen.

## Optional future Kaggle replay — not required now

1. Create a private Dataset, for example **Actual Error Token Preflight V1**,
   containing the two files in `data/actual_error_preflight_v2/upload/`:
   `actual_error_preflight_pairs.jsonl` and `actual_error_preflight_config.json`.
2. Upload `kaggle/preference_token_preflight.ipynb` as a **new private notebook**
   and attach that Dataset. No adapter, model weights or prior notebook output
   needs attaching.
3. Select **Accelerator: None / CPU**. Enable Internet and the existing `HF_TOKEN`
   Secret. Save & Run All, then share the actual notebook slug and version.

The notebook hides CUDA and disables Torch/TensorFlow/Flax use before the tokenizer
import. A fresh CPU session is required. The token is read through Secrets,
never printed or saved in the exported files. Only explicit tokenizer/config
filenames are downloaded; no wildcard snapshot or model-loading call is used.
Keep both Dataset and notebook private because inputs contain reference prose.

Retrieve `actual_error_preflight_v1/`, the emitted runtime and notebook log. The
output contains state, summary, 34 token audits, seventeen encoded pairs and the
credential-free `tokenizer_assets/` bundle. If a gate fails, retain the state,
partial audits and assets; do not change length limits or silently drop pairs.

## Checks

- All seventeen chosen/rejected pairs and eleven groups match the frozen packet.
- Each chosen full token sequence must reproduce its **archived exact token-ID
  hash**, prompt length and supervised length. This checks tokenizer-only rendering
  against the original processor rendering, not just approximate word counts.
- Each actual rejected response is newly encoded with the same exact chat
  template. Prefix identity, one BOS, one supervised end-of-turn marker, decoded
  content and the 1,024-token limit must pass without truncation.
- Prompt and left-padding positions are ignored in labels. A real response-ending
  token stays supervised even if its ID equals the padding ID.
- Constant synthetic observed-token log probabilities check causal indexing,
  response sums and token means. A synthetic DPO identity check yields log(2)
  when policy and reference are identical. These are algebra/packing checks,
  **not measured Gemma log probabilities**.

No model is generated, scored or trained. Real reference-score caching, gradient
checks, memory-fit checks and compatibility with the eventual training trainer
remain future gates requiring an explicitly authorized GPU implementation/run.
Passing this CPU notebook does not establish writing-quality gains or make the
candidate packet independently verified semantic gold.

## Local reuse

Once the real exported tokenizer bundle is downloaded, the same runtime supports:

```sh
PYTHONPATH=src python3 -m fama_style.evaluation.preference_token_preflight \
  --data data/actual_error_preflight_v2/upload/actual_error_preflight_pairs.jsonl \
  --config data/actual_error_preflight_v2/upload/actual_error_preflight_config.json \
  --tokenizer-dir PATH_TO_DOWNLOADED_TOKENIZER_ASSETS \
  --run-tokenizer-check --output FRESH_PRIVATE_OUTPUT_FOLDER
```

This requires only the pinned tokenizer dependencies in an isolated local
environment. It reads the exported bundle locally, with no HF login or weight
download. All ML-related imports remain lazy under the repository entrypoint
convention. `--check` remains standard-library-only.

Packet builder: `fama_style.research.preference_preflight_packet`.
Notebook builder: `scripts/build_preference_preflight.py`.
The packet reproduced byte-for-byte; the notebook is self-contained, unexecuted
and below Kaggle's one-megabyte source limit. Public [plan.json](plan.json)
preserves the original prepared state; [plan_v2.json](plan_v2.json) records the
corrected packet and [recovery_results.json](recovery_results.json) the successful
local execution. The
[bounded pilot proposal](../actual_error_preferences_v1/PILOT_PROPOSAL.md) is a
design for review, not an executable or approved GPU job.

Initial validation: 643 CPU tests passed, including exact archived-ID mismatch rejection,
response-boundary/BOS/end-marker checks, PAD=EOS handling, causal-score indexing,
DPO reference identity/sign/stability and the notebook's no-model/no-training
execution contract. The original `train.py --check` also passed. Synthetic checks
did not certify the then-unexecuted real tokenizer run. The new recovery report
records the real result and the structured-return regression fix separately.
