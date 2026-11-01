# Matched prose-only Gemma 4B comparison v1

The preparation specification below is preserved; [version 1 results](GEMMA4B_PROSE_RESULTS.md)
are now available. Nine first outputs compare direct rewriting, detailed
brief reconstruction and compact-claim reconstruction on the same three exposed
training-side cases. The notebook is `kaggle/reconstruction_prose_4b.ipynb`.
Original 4B and local 1B runs remain separate and unchanged. No training,
adapter loading, additional annotation or reserved examples are involved.

## Experimental controls

The generation jobs are copied byte for byte from the executed local 1B
`package_v2`. Thus all nine user prompts, case and claim IDs, repaired confidence
and media-authority bindings, compact transformation and rotated condition order
are unchanged. References are provided only to direct rewriting as its legitimate
input, never to brief reconstruction requests. Each generation uses a fresh
one-turn chat, without demonstrations, other outputs or conversation history.

Use `google/gemma-3-4b-it` at the earlier executed 4B revision
`093f9f388b31de276ce2de164bdc2081324b9767`. All nine jobs share the native processor
and chat template, NF4 double quantization with FP32 compute, one T4, seed 1729,
greedy one-beam decoding, a 512-token output budget and a 4,096-token combined
context budget. Tokenization of all nine prompts is checked before loading model
weights; no input truncation is permitted. A model access request confirms the
pinned revision and does not silently use a moving main branch.

Dependency pins reuse the package versions from the successful generation stage
of the previous 4B run: Transformers 5.20.1, Accelerate 1.15.0, bitsandbytes 0.50.2
and huggingface-hub 1.33.0, with sentencepiece and Pillow. The 1B engine and GGUF
quantization differ, so matched user prompts do not make cross-model effects
causally attributable to parameter count alone. Native chat templates and numerical
implementations can differ. The useful comparison is between the three conditions
inside this new 4B run. The three selected failures are not a fresh validation
sample or evidence of generalization.

## Prose output and completion status

The model is asked for prose only, or `BLOCKED:` with a reason if it cannot
faithfully realize the input. Python attaches immutable job IDs and configuration
metadata; the model need not generate JSON or administrative fields. Raw decoded
text is saved unchanged. No wrapper removal, extraction, regeneration or automatic
repair is applied to prose. The existing pipeline's blank, blocked, formatting-cue
and output-limit issues are retained and counted rather than hidden.

`execution_complete` and `complete` mean that all nine generation calls finished
and their first outputs were saved. `response_contract_complete` separately checks
observable format, empty and output-limit flags; it accepts an explicit blocked
response as compliant with the requested contract, not as a generated paragraph.
These fields never certify claim preservation or style. A formatting deviation
does not convert completed GPU inference into a runtime failure. Numerical, access,
context or model-execution errors still fail the run, preserving available outputs.

Runtime artifacts record the pinned model revision, installed package versions,
GPU name/count, prompt and input-token hashes, complete decoding configuration,
input/output token counts, elapsed times, peak allocated memory and runtime source
hash. Kaggle retrieves `HF_TOKEN` only from its enabled Secret, without writing its
value into code, configuration, environment variables or logs. Upstream exception
messages are omitted because they can contain credential-bearing material.

## Manual upload

The private packet is
`research_runs/neutral_reconstruction_v1/gemma4b_prose_v1/package/`.

1. Import `kaggle/reconstruction_prose_4b.ipynb` as a **new private notebook**.
2. Create a **private Dataset** titled **Reconstruction Prose 4B Comparison** with
   exactly the two files in that packet's `upload/` folder. Attach it to the new
   notebook. Do not upload `private_key/`, old outputs or other packet files.
3. Enable **HF_TOKEN** in Kaggle Secrets, **Internet**, and **GPU T4 x2**. The first
   cell exposes GPU 0 only. Start a fresh session, then **Save & Run All**.
4. Download the entire `reconstruction_prose_4b_v1` output directory, or provide
   the actual notebook URL slug and version for retrieval. Keep outputs private.

This notebook is self-contained and below Kaggle's source-size limit. It has not
been GPU-executed. It is prepared for manual upload only; no push or run is submitted.

CPU preparation and validation:

```sh
python3 scripts/build_reconstruction_prose_4b.py
PYTHONPATH=src python3 -m fama_style.evaluation.reconstruction_prose_4b \
  --jobs research_runs/neutral_reconstruction_v1/gemma4b_prose_v1/package/upload/reconstruction_prose_4b_jobs.jsonl \
  --config research_runs/neutral_reconstruction_v1/gemma4b_prose_v1/package/upload/reconstruction_prose_4b_config.json \
  --check
```

The packet builder `fama_style.research.reconstruction_prose_4b_packet` takes
`--local-packet` and `--output`, verifies the parent manifest and exact three
exposed cases, copies the nine prompts unchanged, records lineage and refuses
overwrite. Reproduction requires the private local packet; public CPU tests use
synthetic text and require no weights, server or GPU.

After execution, assess actor binding, claim coverage, conditions, negation,
modality, time and numeric scope, analogies, attribution and causal direction
against the frozen references. Report commentary, readability, organization and
copying independently. Preserve all nine cases, including abstentions and failures.
The primary question is whether compact input improves fidelity over detailed
briefs without losing supporting content, with direct rewriting as the baseline.
Neither lower overlap nor successful execution establishes useful Fama-style
transfer or author authenticity. Do not scale annotation or begin post-training
based solely on three reused cases; a fresh sample is required later.
