# Grounded-operation realization comparison v1

**Prepared, not executed. Inference only.** Compare ordinary rewriting, shallow presentation plans and the new claim-bound rhetorical programs on base Gemma and the saved planned adapter. This tests richer realization guidance before another training run.

## Fixed design

Five previously inspected training paragraphs: four transformation hypotheses and one preservation control from [grounded rhetorical operations v1](GROUNDED_RHETORICAL_OPERATIONS.md). All five were seen by the saved adapter during training. No development-validation or reserved test row is read. These are oracle mechanism diagnostics, not generalization or deployment results.

| Model | A | B | C |
| --- | --- | --- | --- |
| Pinned base Gemma 3 4B | Ordinary + shared source evidence | Shallow plan + shared source evidence | Rhetorical program + shared source evidence |
| Frozen planned adapter from pilot v1 | Same | Same | Same |

**30 first-attempt greedy calls**: two model states × five examples × three conditions. Rotate condition order within examples; run the base before attaching the adapter, avoiding a second base-model allocation. Preserve identical examples, semantic instruction, source evidence, decoding and output cap. The two model states receive byte-identical prompts for a given example/condition.

All three conditions share the same source claim index, source-supported component-count supplement and selected existing relation evidence. A is therefore an ordinary instruction **with this common source scaffold**, not a reproduction of an earlier unscaffolded baseline. B adds the old schema's count/word-band/semicolon controls. C adds the closed rhetorical program and its paragraph/sentence actions. The source scaffold is shared to avoid attributing extra ledger access to the richer program. Its evidence is source-derived, but its selection follows reference-informed cards; this is not a fully deployable source-only pipeline.

C's actions organize supported premises/inferences, criterion exposition, existing decisions and evidence/conclusions. Stages are discourse moves, not compulsory sentence slots. The control already has parallel finding/alternative-explanation blocks; preserving it must not count as a style-transformation success.

## Adapter and data identity

Use only `adapter_planned/` from `oliverlds/plan_realizer_pilot` version 1. CPU preflight requires exact reviewed SHA-256 hashes for `adapter_config.json` and `adapter_model.safetensors`, with the expected base ID/revision. After loading, verify every adapter tensor hash against the reviewed pilot. Both model states have all parameters frozen. No optimizer, backward pass, adapter update, new adapter, retry or full training path exists in this runner.

The planned adapter was trained on **shallow plans**, not the new programs or this shared evidence scaffold. Its response to C is an unfamiliar-format diagnostic, not evaluation of an operation-trained realizer. It also saw all five paragraphs, so a target-like answer cannot be treated as held-out transfer.

The private pack binds the frozen training split, operation-package manifest, selected cases/control, compiled source payloads, reference-derived shallow plans, saved adapter files/tensors, runtime code, model revision, decoding and exact jobs. No reference paragraphs, target annotations, target cue spans or evaluation notes are sent to the generator. Structural side information remains oracle-derived even after reference wording is removed.

## Runtime and failure handling

One visible T4 from the T4 x2 option. Pinned base Gemma 3 4B, NF4 double quantization, FP32 compute, SDPA, deterministic greedy decoding, 512 generated tokens and 4096 total context tokens. Enable `HF_TOKEN` through Kaggle Secrets; values are never printed or saved. Original Gemma 3 1B training remains unchanged.

Audit all fully rendered/tokenized prompts before loading model weights. Over-context prompts stop the run; none are truncated. CPU preflight does not certify token fit or GPU memory. Record prompt/output tokens, generation seconds, total elapsed time and memory rather than promise runtime.

Empty/cutoff generations are retained as failed attempts while remaining calls continue. Multi-sentence or unexpected formatting remains verbatim. No repairs, replacement with source/reference, or selection of a later retry. Runtime/model errors stop the run with sanitized exception types. Unload Gemma before loading independent NLI.

## Separate assessment dimensions

Save all raw outputs, full frozen-verifier verdicts and source-claim-to-output-sentence NLI matrices. These matrices estimate where source claims appear in the output. Localization uses entailment ≥0.75 and a ≥0.10 margin over the next candidate sentence; these are uncalibrated diagnostic rules, not semantic accuracy. Over-context NLI returns unknown rather than truncating. Ambiguous sentence boundaries abstain.

Composite claims can support more than one rhetorical stage. The criterion-introduction/definition distinction cannot be fully assessed from one composite source claim's sentence location; record that limitation. The component-count supplement is separately marked rather than treated as an atomic claim. Location matrices do not prove operation adherence, resolve every subclaim or measure language actions.

Assess after execution:

- **Claim drift:** actor/cohort identity, quantities and their roles, modality, conditions and unsupported mechanisms.
- **Style:** concrete reference-grounded paragraph/sentence/language changes versus generic formalization.
- **Copying:** exact/near source copy and source/reference similarity.
- **Operation realization:** did the requested separation, staging or evidence/inference integration actually occur? Sentence-count match alone is insufficient.
- **Cost:** tokens, runtime and overhead for each condition and model.

Report four transformation cases separately from the one preservation control. Retain failed attempts in denominators. Preserve the existing allowance for minor drift when substantial style improvement justifies it; NLI is not a binary acceptance gate.

A provisional positive signal is substantial style improvement on at least two of the four transformation cases for C over both A and B within a model state, without increasing high-drift cases. Report exact paired evidence and style/drift sensitivity, not statistical significance. The control must retain its hypothetical alternatives rather than inventing an adjudication. A positive result supports expanding evidence across papers, not immediate mapper training. If C still fails, investigate representation or realization capacity instead of adding epochs to the old shallow-plan adapter.

## Manual Kaggle run

1. Upload `kaggle/operation_realization.ipynb` as a **new private notebook**.
2. Attach a **private Dataset containing only** `research_runs/grounded_operation_realization_v1/input_pack/operation_pack.json`.
3. Also attach the **outputs of `oliverlds/plan_realizer_pilot` version 1**, containing `adapter_planned/adapter_config.json` and `adapter_planned/adapter_model.safetensors`. No fresh training is needed. If attaching outputs is unavailable, a private Dataset may carry that unchanged directory and those two files.
4. Enable internet, `HF_TOKEN` and **GPU T4 x2**, then Save & Run All. Only GPU 0 is exposed. Share the notebook slug and version after completion.

The pack and original pilot outputs contain copyrighted/private paragraph data and must remain private. No credentials should be uploaded in any Dataset.

Expected outputs under `/kaggle/working/grounded_operation_realization_v1/`: `evaluation.json`, `generations.private.jsonl`, `jobs.private.jsonl`, `metrics.private.jsonl`, `verifier_results.private.jsonl` (full semantic/localization evidence), `token_audit.json`, `adapter_identity.json`, `runtime.json`, `run_status.json`, `manifest.json`.

Implementation: `src/fama_style/rhetoric/operation_realization.py`. Builder: `scripts/build_operation_realization_notebook.py`. CPU tests: `tests/test_operation_realization.py`.

The builder embeds runtime code and public verifier fixtures, without private paragraph data, adapter weights, credentials or saved outputs. CPU tests exercise matching source scaffolds, generator-reference exclusion, failure denominators, verdict identity, unknown localization, adapter bytes/revision checks, absence of training actions and isolated notebook preflight. No new GPU run has been submitted.

Notebook packaging fix: the embedded runtime is transported as losslessly compressed code and expanded inside the notebook. The rebuilt upload is 243,294 bytes; the builder and regression test enforce a conservative 900,000-byte ceiling for both the notebook file and combined cell source. Runtime source bytes, experiment prompts, adapter identity and private input-pack hash remain unchanged. Replace only the notebook after the original over-limit commit failure; reuse the existing private Dataset and pilot output attachments.
