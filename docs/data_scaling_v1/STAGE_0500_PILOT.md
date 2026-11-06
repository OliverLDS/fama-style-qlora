# Mixed-teacher 500-pair pilot, before separate teacher tracks

The mixed 500 release will be tested first. Review technical execution, meaning,
copying, readability and detector headroom before completing separate Gemini and
GPT-6 tracks. Gemini counterparts can then be completed automatically first;
the manual GPT-6 track can follow later. Version 1 has now
[completed and been reviewed](STAGE_0500_RESULTS_V1.md). The preparation instructions
below are historical, not a request to rerun. Six matched Pangram scans come next.

## Manual Kaggle upload

1. Create a **new private Dataset**, for example **Corpus 500 Mixed Teacher Pilot
   v1**, with only the two files in `data/corpus_500_pilot_v1/upload/`:
   `corpus_500_pairs.jsonl` and `corpus_500_config.json`.
2. Upload `kaggle/corpus_500_pilot_4b.ipynb` as a **new private notebook**, attach
   that Dataset, enable Internet and the `HF_TOKEN` Secret, and choose **GPU T4
   x2**. The first cell exposes GPU 0 only. Restart the session first if torch
   was already imported. Use **Save & Run All** when ready.
3. Preserve the notebook log and the entire `corpus_500_pilot_v1/` output folder,
   including `evaluation.json`, `state.json` and `epoch_1/adapter/`. Share the
   notebook slug/version for retrieval. Keep partial outputs on failure.

This upload includes all 500 training pairs and the evaluation inputs/references;
do not upload only the training JSONL from the curation release. Both the Dataset
and notebook must remain private. No old adapter or notebook-output input is
needed. The notebook reads HF_TOKEN from Kaggle Secrets without printing or
embedding it. It calls neither Gemini nor Pangram.

## Training and evaluation controls

| Component | Setting |
| --- | --- |
| Base | Fresh `google/gemma-3-4b-it`, revision `093f9f388b31de276ce2de164bdc2081324b9767` |
| Training | Mixed 500 release, 125 pairs per group, 53 papers |
| Budget | One epoch; batch 1, accumulation 2; 250 updates |
| Adapter | Language-attention q/v, rank 8, alpha 16, dropout 0.05 |
| Optimizer | AdamW, learning rate `2e-5` with linear decay; seed 1729 |
| Precision | NF4 double quantization, FP32 compute; no FP16/BF16 AMP |
| Length | 1,536-token ceiling; no truncation |
| Outputs | 24 base + 24 saved/reloaded-adapter outputs |
| Decoding | Greedy, one beam, 512 new tokens, repetition penalty 1.0 |
| Runtime | 90 minutes including dependency installation; no automatic retries or extra epochs |

The training instruction and recipe match the original 200-pair pilot. Inference
uses the agreed simple instruction, exactly matching the 200-pair simpler-prompt
runtime, including formatting:

> Rewrite the following passage in natural, human-written academic economics prose.
>
> Return only the rewritten paragraph.

The new runtime and builder are separate files; the root 1B pipeline, earlier
notebooks, adapters, prompts, datasets and reports remain unchanged. The original
200 rows are still the exact prefix of the curation release. A size increase also
changes teacher composition, exposure and update count; this is a practical
expansion pilot, not an isolated causal estimate of data size.

## Held-out inputs and detector check

The 16 original paper-held-out contents are reused with their existing inputs.
Eight have an additional Gemini-generated neutral input, selected using the
eight detector IDs frozen before the 200-pair run. This gives 24 input cases,
not 24 independent reference paragraphs. Every base/adapter comparison uses
the same input and decoding. Targets are used for masked likelihood, never
inserted into the generation prompt. No training paragraph is a detector case.

The eight Gemini inputs were generated automatically on the confirmed free tier
and checked in a separate meaning-only call. Codex comparison made three logged
minimal corrections; generation, review, original responses and final hashes
are retained privately. This is automated review, not independent human ground
truth. The inputs were not selected using detector scores or made artificially
"AI-like."

After outputs are downloaded and checked, prepare two frozen cases for initial
Pangram scans: **input, base output, trained output** for each, six texts total.
The remaining six cases form an optional extension of the same panel. Use actual
text word counts to budget ordinary daily credits; retain the exported PDFs.
No CSV entry, sentence-level report or detector subscription is required.

If Gemini inputs are already labeled human, that finding limits this detector
endpoint; it is not evidence of detector improvement after fine-tuning. Report
all input/output scores and copying, without selecting only favorable cases.
If inputs have AI-score headroom, compare changes against both the input and
base model while examining meaning and readability. Detector scores are not a
training-pair acceptance criterion or proof of writing quality.

## Review gate before separate tracks

First require complete coverage: 500 examples, 250 updates, 524 label checks,
finite gradients, changed adapter weights, exact adapter reload, and two
24-output snapshots. Failures retain partial outputs and are diagnosed before
another run. No silent data exclusion or truncation is allowed.

Then compare the same 16 existing inputs against the completed 200-pair
simple-prompt outputs, and compare base versus mixed-500 on the eight Gemini
inputs. Assess copying, readability, stylistic change and material claim changes
jointly; small wording/claim shifts are not automatic failures. A detector gain
accompanied by lost numbers, reversed relationships or unreadable text is not
sufficient evidence of useful rewriting. NLL is also reported by input condition;
do not directly compare the old strict-prompt NLL with the new simple-prompt NLL.

Proceed to separate tracks after resolving material pipeline or data problems
and documenting mixed findings. No detectable improvement, or already-human
Gemini inputs, should not silently turn into a claim of success. It may instead
motivate a focused repair or a different evaluation endpoint. Completion of the
Gemini track comes first, with existing eligible inputs reused; GPT-6 manual
completion follows when the necessary batches are ready. Neither expansion nor
additional training starts automatically from this notebook.

Builders: `scripts/build_corpus_500_pilot.py` and
`src/fama_style/research/corpus_500_packet.py`. Self-contained runtime:
`src/fama_style/training/corpus_500_pilot.py`. Private upload packet:
`data/corpus_500_pilot_v1/`. Private teacher-input provenance:
`research_runs/data_scaling_v1/corpus_500_pilot_4b_v1/gemini_development_v1/`.
