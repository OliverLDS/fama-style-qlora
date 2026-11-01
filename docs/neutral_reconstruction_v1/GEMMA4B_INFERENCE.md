# Gemma 3 4B reconstruction transfer check v1

The preparation specification below is preserved; [version 1 results](GEMMA4B_V1_RESULTS.md)
are now available. Six first-attempt outputs across the same three previously
exposed training-side cases compare direct neutral rewriting with brief-only
neutral reconstruction. One brief incorporates the versioned confidence and
media-authority repair; two remain unchanged. No test or fresh development sample
is selected and no training is performed.

This checks whether the intended student model can use the data-construction
protocol, rather than assuming GPT-6 realization behavior transfers. It does not
measure neutral-to-Fama rewriting, a fine-tuning gain, generalization, human
authorship or detector evasion. Outputs remain private and are not automatically
admitted to training.

## Controls and limitations

Both conditions use `google/gemma-3-4b-it`, a common neutral-prose instruction,
claim-preservation constraints and the revised rule against turning guardrails
into evidence disclaimers. Both request the same JSON response contract; one
receives only its original paragraph and the other only its content brief.
The original 1B training entry point and earlier experiments are unchanged.

The generator receives one request at a time, without conversational history or
few-shot examples. No direct output or original paragraph enters a reconstruction
request. The private upload contains the direct originals as legitimate inputs;
the runtime passes only the current job to the processor. A new cache is created
per generation call. Only the offline evaluation key pairs reference text with
reconstruction outputs afterward.

Opaque claim IDs and all schema fields are retained. Instruction wording is
adapted from GPT-6 batch instructions to individual Gemma requests; download and
chat-account coordination instructions are omitted. The corrected brief and
changed request format mean the old GPT-6 outputs are contextual evidence, not
a precisely controlled cross-model comparison.

Generation is greedy, one beam, at most 768 new tokens, seed 1729 and an 8,192-token
combined input/output budget. No input truncation is allowed. Each first output
is saved, including blocked, malformed or output-limited responses. There are no
format-repair calls or retries. Invalid numerical scores fail the run; negative
infinity used to suppress individual tokens is permitted if finite options remain.

Use one exposed T4 with NF4 double-quantized weights and FP32 compute, following
the repository's earlier precision correction for T4 generation. The runtime uses
`Gemma3ForConditionalGeneration` and `AutoProcessor`, consistent with the
[official Transformers Gemma 3 documentation](https://huggingface.co/docs/transformers/model_doc/gemma3)
and the [Google model card](https://huggingface.co/google/gemma-3-4b-it).
The notebook pins Transformers 5.20.1, Accelerate 1.15.0, bitsandbytes 0.50.2 and
huggingface-hub 1.33.0, reusing the existing project dependency set. GPU compatibility
has not been newly executed locally. The model's immutable revision is resolved
once at runtime and used for processor and weights; package versions, token
counts, prompt/token hashes, decoding settings, timing and peak memory are saved.

## Manual Kaggle run

The notebook is `kaggle/reconstruction_4b.ipynb` (self-contained, below Kaggle's
source-size limit). The private package is
`research_runs/neutral_reconstruction_v1/gemma4b_inference_v1/`.

1. Create a **private** Kaggle Dataset titled **Neutral Reconstruction Gemma 4B Pilot**
   with the two files in its `upload/` folder. Do not upload `private_key/` or prior
   responses. Both upload files contain frozen configuration or generation inputs.
2. Import the notebook as a **new private notebook**, and attach that Dataset.
3. Enable `HF_TOKEN` in Kaggle Secrets, Internet and **GPU T4 x2**. The first code
   cell exposes only GPU 0. Start a fresh session so torch has not been imported.
   The HF account must already have access to the Gemma 3 4B model. The token is
   retrieved only through Kaggle Secrets there; it is never pasted or printed.
4. Save & Run All. Download the whole `reconstruction_4b_v1` output folder or
   provide the notebook slug and version for retrieval.

Model loading may dominate this six-output run. Progress phases and each
completed output are printed; prose and credentials are not printed. Errors
record only phase and exception type, omitting potentially credential-bearing
upstream messages. A run is structurally complete only with all six valid,
nontruncated JSON responses; blocked responses stay in the denominator and do
not count as successful prose generation.

Review outputs against the frozen private references for: missing/added claims,
confidence and categorical scope, attribution, conditions, causal direction and
numeric values; added commentary; readability and organization; word-sequence
overlap and longest shared runs. Report per-case tradeoffs and all failures.
Keep execution success separate from semantic or stylistic success. No detectors
or paid external model calls are needed for this check.

CPU preparation and validation:

```sh
python3 scripts/build_reconstruction_4b.py
PYTHONPATH=src python3 -m fama_style.evaluation.reconstruction_4b \
  --jobs research_runs/neutral_reconstruction_v1/gemma4b_inference_v1/upload/reconstruction_4b_jobs.jsonl \
  --config research_runs/neutral_reconstruction_v1/gemma4b_inference_v1/upload/reconstruction_4b_config.json \
  --check
```

Rebuild the frozen private package into a fresh directory with
`fama_style.research.reconstruction_4b_packet`, passing `--packet` (stage2_package_v2),
`--instruction` (realization_instruction_check_v1), `--repair`
(semantic_binding_repair_v1/results), and `--output`. The builder verifies parent
manifests, exact cases, revised handoff and input lineage, and refuses overwrite.
