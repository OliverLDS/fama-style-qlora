# Prepared 200-pair pilot and matched held-out inference

The user authorized this bounded step with “do it.” Version 1 has now
[completed and been reviewed](STAGE_0200_RESULTS_V1.md): all 200 pairs and 100
updates, with 32 matched held-out outputs. The preparation instructions below
are retained for provenance, not as a request to rerun it. Submission used the
user's manual Kaggle UI workflow.

## Run in Kaggle

1. Create a **new private Dataset** containing only these two files from
   `data/corpus_200_pilot_v1/upload/`:
   `corpus_200_pairs.jsonl` and `corpus_200_config.json`.
2. Upload `kaggle/corpus_200_pilot_4b.ipynb` as a **new private notebook** and
   attach that Dataset. No prior adapter or old baseline output is required.
3. Enable Internet and the `HF_TOKEN` Kaggle Secret. Choose **GPU T4 x2**;
   the first code cell exposes GPU 0 only. Restart first if torch was already
   imported, then use **Save & Run All**.
4. After completion, retain the notebook log and the whole
   `corpus_200_pilot_v1/` output folder. Report the notebook slug/version for
   retrieval and review. Preserve partial outputs if it fails.

No credentials are in the notebook, Dataset or repository. The token is read
only from Kaggle Secrets, passed directly to Hugging Face and never printed.
The private data and model outputs must remain private.

## Fixed scope

| Component | Setting |
| --- | --- |
| Model | `google/gemma-3-4b-it` |
| Revision | `093f9f388b31de276ce2de164bdc2081324b9767` |
| Initialization | Fresh base, no old adapter continuation |
| Training | All 200 unique parents, 50 per author group, 48 papers |
| Budget | One epoch, batch 1, accumulation 2, **100 optimizer updates** |
| Learning rate | `2e-5`, linear decay, no warmup |
| Adapter | Language attention q/v only; rank 8, alpha 16, dropout 0.05 |
| Precision | NF4 double quantization, FP32 compute; no FP16/BF16 AMP |
| Length | 1,536-token ceiling, **no truncation** |
| Evaluation | 16 examples: two paragraphs from two papers per group |
| Generation | Same prompt and greedy decoding for base and trained model; 512-token output cap |
| Runtime cap | 90 minutes including dependency installation; no automatic retries |

The original Gemma 1B CLI and historical notebooks/results are unchanged.
The dependency pins match the previously successful expanded 4B runtime.
The 1,536-token ceiling gives inline notation and the full prompt additional
room; actual token lengths are checked before loading model weights.

The worker checks all **216 supervised response boundaries**, masks prompt
tokens, checks target round trips and refuses overlength rows. It compares its
target-only likelihood calculation against the model's built-in masked loss.
It saves 16 base generations before training, logs every update and every seen
training ID, checks gradients, saves the adapter, verifies reloaded tensors and
then saves the same 16 generations with the reloaded adapter.

## What to inspect afterward

Successful execution requires `evaluation.json` and `state.json` to report
completion, all 200 examples seen, 100 updates, 216 label checks, changed adapter
weights, an exact reload match and two complete 16-output snapshots.
`label_audit.json`, `gradient_checks.json`, `training_log.json`,
`generation_config.json` and `epoch_1/adapter/` provide the supporting evidence.

Compare base versus trained target NLL, outputs, exact/long-span copying,
readability, generic style change and claim preservation on the same inputs.
Runtime numeric-string flags are triage only; they cannot certify meaning.
Automated source/reference review follows after downloading outputs. Do not
interpret low loss or successful execution as faithful rewriting by itself.

Eight detector cases are frozen in the config before generation: two examples
from one development paper per author group. Once outputs are reviewed, prepare
matched Pangram scans of **base and trained outputs**, calculate credits from
actual full texts, and split scanning over the ordinary daily allowance as
needed. No scans of the training inputs/targets are requested now. No detector
calls or scores are part of training, selection or the notebook.

This is paper-held-out development within known author groups, not an
author-held-out or sealed-test result. Earlier development exposure and automated
adjudication remain limitations. It tests a broader mixed-author direct-SFT
pilot; it cannot alone establish distinctive Fama-style learning. Future stages
must restart from the same base with the same one-epoch schedule, and still
require their own explicit GPU authorization. There is no automatic expansion.

Builders: `scripts/build_corpus_200_pilot.py` and
`src/fama_style/research/corpus_200_packet.py`. Runtime:
`src/fama_style/training/corpus_200_pilot.py`. Private plan, manifest and exact
development lineage: `data/corpus_200_pilot_v1/`.

Validation: all 713 CPU tests pass, including five new pilot contract tests.
The real 216-row packet passes `--check`; packet hashes, notebook syntax and
embedded runtime agree. The original 200-pair cohort hash is unchanged. The
root `train.py --check` passes and its notebook was regenerated without a
historical notebook change. No model weights were downloaded on the Mac.
