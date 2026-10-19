# Recorded results and limits

## Design

The task is neutral plain English → original-author economics prose, preserving substantive claims, actors, quantities, qualifications and causal direction. Earlier 1B diagnostics checked copying, label preparation and two-example overfitting. Current experiments use `google/gemma-3-4b-it`, NF4 double quantization, FP32 compute/unquantized layers and one Tesla T4.

The pilot starts with a fresh adapter: LoRA rank 8, alpha 16, dropout 0.05, language-attention q/v only, 2,228,224 trainable parameters across 68 modules. AdamW learning rate `5e-5`, zero weight decay, clipping at 1.0, microbatch one, accumulation two, seed 42 and non-reentrant gradient checkpointing. Each of 247 training pairs is seen once: 124 updates, the last correctly normalized for one example. No padding/truncation; longest full conversation 751 tokens, permitted limit 1024.

## Verified progression

| Kaggle run | Recorded outcome |
|---|---|
| `oliverlds/corpus-v1-baseline`, v4 | 313 token/boundary audits and 21 valid base generations; no training |
| `oliverlds/corpus-v1-smoke`, v1 | Three pairs, five finite updates, changed adapter and verified reload |
| `oliverlds/corpus-v1-pilot`, v1 | One epoch, 247 pairs, 124 updates, adapter reload and 21 validation generations |

Baseline versions 1–3 exposed code defects in revision lookup, CUDA initialization ordering and optional device-map assumptions. The corrected code resolves an explicit public Hub revision, initializes CUDA before allocator resets and checks actual parameter/buffer devices.

Model revision: `093f9f388b31de276ce2de164bdc2081324b9767`. Recorded versions: PyTorch 2.11.0+cu128, Transformers 5.20.1, PEFT 0.21.2, bitsandbytes 0.50.2, Accelerate 1.15.0 and huggingface-hub 1.33.0.

## One-epoch pilot

| Measurement | Value |
|---|---:|
| Original validation loss | 3.5448506873 |
| Adapted / reloaded loss | 1.1402539135 / 1.1402539135 |
| Relative loss decrease | 67.83% |
| Individual validation losses improved | 21/21 |
| Exact source copies, base / epoch | 0/21 / 6/21 |
| Output words, base / epoch | 2,960 / 1,713 |
| Input / reference words | 1,675 / 2,040 |
| Peak allocated / reserved GPU memory | 6.79 / 11.30 GiB |

Loss weights per-example mean cross-entropy by shifted supervised-token count. Adapted/reloaded per-example losses matched exactly. Local review verified all eight manifest hashes and all 136 saved FP32 adapter tensor hashes. Full artifacts remain local; this repository publishes code and aggregate measurements, not paper text or checkpoints.

## Interpretation

Unblinded qualitative agent review found less generic elaboration and fewer unsupported claim extensions. Several conditions and attributions altered by the base were retained by the pilot. A table-reference error remained: Table 2 was described as single-variable although it belongs with full specifications. These are qualitative observations, not independently calibrated human scores.

Copying increased. Six outputs exactly match their inputs after whitespace normalization; 13/21 reach 0.90 word-sequence similarity. Similarity uses whitespace words, case/punctuation-sensitive, with `difflib.SequenceMatcher(autojunk=False)`; it is not a semantic/style score. Median source similarity rose from 0.268 to 0.934. No adapted output exactly matches its original reference.

The model became less verbose, but **distinctive Fama/coauthor style transfer beyond conservative copying remains unestablished**. Teacher-forced loss does not demonstrate faithful generation or style gains. All 21 validation samples are from one paper and were used during development. No fresh multi-paper benchmark or reserved-test improvement is claimed. Test rows were length-audited in the baseline but not trained, generated or scored.

## Next evaluation

The offline tool compares unchanged input, base and pilot in balanced A/B/C order. Claims and three style dimensions are rated separately; identical texts must tie. It produces descriptive paired wins/ties/losses, not a combined quality score or significance claim. Real-data scores remain blank. Candidate masking is not independently blinded evaluation: earlier familiarity and exact copies can reveal identity.

The decision is whether the adapter improves expression beyond returning the input unchanged, without semantic loss. More epochs, model size or QLoRA itself cannot be declared the solution/cause from these results alone. Rebuilding the real-data review requires local artifacts; public synthetic tests cover the tool's contracts.
