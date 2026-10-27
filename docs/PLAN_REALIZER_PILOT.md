# Matched supervised-realization pilot v1

**Prepared, not executed. GPU training is disabled by default.** No existing experiment, source/target text, adapter or result is replaced. The original 1B CLI remains unchanged.

The recent groupwise diagnostics achieved sentence-format control and actor resolution, but generated unsupported mechanisms and changed numerical roles without substantial author-style gains. The next question is whether *supervised realization* can learn to use bounded structural controls. This is not yet a structural-mapping experiment.

## Hypotheses and comparison

Train two fresh adapters on the same 12 existing source/reference pairs: direct SFT and plan-conditioned SFT. Both target the unchanged original author paragraphs. Use the same pinned Gemma 3 4B base, adapter initialization, learning rate, shuffled example schedules, target supervision and optimizer-update budget.

Evaluate the base and both adapters under both prompt formats. This yields six model/prompt combinations:

| Model | Ordinary prompt | Oracle-plan prompt |
| --- | --- | --- |
| Base Gemma | Prompt baseline | Oracle prompt effect |
| Fresh direct adapter | Matched direct-SFT control | Direct adapter with plan instructions |
| Fresh planned adapter | Plan-trained adapter without plan | Plan-conditioned realization |

The main practical comparison is planned adapter + plan versus direct adapter + ordinary prompt. Compare both adapters under the *same plan prompt* to isolate the training-format effect more closely. The historical 247-row pilot is background, not a matched causal control for this much smaller training set. The two new arms have equal example exposures and update counts, not equal input-token cost.

## Data and source/plan/reference audit

Start from the 13 existing taxonomy-discovery pairs whose proposed neutral text was marked `retain`, with exact equality to the frozen training source. Verify annotation manifests and target sentence-text reconstruction. Twelve support the fixed 2–6 sentence/word-band plan; the remaining eight-sentence paragraph is excluded for representation feasibility, not model performance. The 12 rows cover six training papers. No proposed source revision is applied.

Inspect all 12 source/reference pairs in-session and retain uncertainty in a private hash-bound audit. Checks include actor/cohort roles, conditions, numerical values, comparisons, causal direction, modality and major propositions. All 12 training and four selected development pairs have equal numeric-token multisets. This **does not prove numeric-role fidelity or semantic equivalence**. Surface negation/modality/causal cue mismatches are reported, not automatically treated as errors.

Some training pairs retain scope/intensity caveats, including stronger target emphasis, explicit versus implicit long-term scope, and differences in independence qualifiers. The previous LLM `retain` decision and this assistant audit are not independent verification. The pilot is an experiment on imperfect existing pairs, not a newly certified gold dataset. No silent repairs or human-review dependency is introduced.

Private audit files live in `research_runs/plan_realizer_pilot_v1/preparation_review/`; automatic per-pair checks also travel in the private pack.

Select four rows from the existing 21-example development split using fixed plan feasibility and lowest combined source/reference word budgets, with ID tie-breaks. This deliberately inexpensive subset is biased toward shorter paragraphs and cannot establish broad transfer. All conditions use exactly those four rows. Two separately labeled training probes check memorization; they are not included in development-quality results. The 45 reserved test examples are never read. A promising pilot would need confirmation across the full development set before mapping-model training or broader claims.

## Plan representation and semantic constraints

`realizer_surface_plan_v1` is a closed, content-free projection:

- Paragraph: exact sentence count, 2–6.
- Sentence: one word band per sentence: short 1–16, medium 17–32, long 33–48, extended 49–72.
- Language: semicolon ceiling, 0–2.

This is a deliberately limited presentation-plan realization test, **not the full standardized rhetorical taxonomy**. It does not predict paragraph roles, claim alignments, syntax trees or rhetorical relations. Existing LLM target annotations validate coverage/count; their free-text descriptions and lexical clues are not included. The same deterministic plan extractor is used on both training and development targets.

The generator always receives the original source. Claim order and substantive fidelity take priority over formatting hints; the plan authorizes no new causal, conditional or quantitative relation. No extracted reference words, numeric claims or target sentence text enter the generation prompt. Development plans are extracted from references and must be reported as **oracle side information**, not deployable predictions. Learning the mapper remains deferred.

## Bounded training and GPU budget

Pinned base: `google/gemma-3-4b-it`, revision `093f9f388b31de276ce2de164bdc2081324b9767`. Original `google/gemma-3-1b-it` workflow is untouched.

Each arm uses fresh language-attention q/v LoRA, rank 8, alpha 16, dropout 0.05; NF4 double quantization with FP32 compute. AdamW learning rate 5e-5, zero weight decay, gradient norm cap 1, microbatch 1, accumulation 2. Four epochs over 12 pairs gives **24 updates and 48 example exposures per arm**. The entire run is **48 optimizer updates**, not a full-corpus run. Both arms must share the exact initial adapter tensor hashes and fixed per-epoch schedules. The second arm starts from the frozen base after unloading the first adapter.

Expose only GPU 0 from Kaggle T4 x2; no second model copy or GPU parallelism. Load the base once, train/save/unload/reload each adapter sequentially, verify trained/reloaded tensor hashes, and unload Gemma before loading the independent verifier. Embedding/vision parameters stay frozen. Tokens come from Kaggle Secrets without printing or saving values.

All targets are fully supervised with an exactly checked chat boundary and prompt masking; no truncation. Training ceiling is 1536 tokens, generation context ceiling 4096, output cap 512, greedy decoding. Actual tokenization and T4 memory remain runtime checks; CPU preparation does not certify GPU fit.

There are **36 generation calls**: six model/prompt combinations × (four development rows + two training probes). Input overhead, loading, training, generation and NLI make wall time uncertain; no runtime or monetary cost is claimed before execution. The fixed update budget limits training; there is no automatic epoch extension, retry or full-run escalation.

## Evidence and decision criteria

Save all raw generations, token/label audits, update losses/gradients and example IDs, development losses, adapter files and hashes, full verifier verdicts, metrics, runtime/memory and file manifests. Verify reloads and both fresh initial states. Empty/cutoff generations stay failed attempts; no repairs or replacement with source/reference text. A technically complete run with generation failures must say so.

Report separately by development versus training probes and by model/prompt combination:

- Claim drift, with actors/cohorts, quantity roles, conditions and mechanisms inspected separately.
- Meaningful style change versus generic formalization, using the existing style/drift rubric.
- Exact/near copying, length and reference/source overlap.
- Sentence-count adherence and surface style measurements.
- Per-target-token weighted development loss and cost.

The frozen independent NLI verifier and finite surface checks remain imperfect diagnostics, not semantic proof, calibrated severity, author-style accuracy or binary selection gates. Lower loss alone does not establish style learning. Minor drift remains eligible when substantial style gains justify it.

A **provisional** positive signal requires substantial style improvement over the matched direct arm on at least two of the four development rows, without increasing the count of high-drift development outputs, with style/drift sensitivity reported. This small threshold is a follow-up trigger, not statistical significance or proof. Compare the same-plan crossover as well before attributing gains to plan-conditioned training. Training-probe gains alone indicate fitting, not transfer. Otherwise report a negative/underpowered result and reconsider supervision or realization capacity before learning a mapper.

## Code and CPU checks

Implementation: `src/fama_style/training/plan_realizer.py`.
Builder: `scripts/build_plan_realizer_notebook.py`.
Tests: `tests/test_plan_realizer.py`.

The private pack binds frozen train/validation hashes, annotation manifests, selected rows and their provenance, plans, automatic pair audits, configuration and runtime source hashes. The notebook embeds runtime code and public verifier fixtures; it contains no private paragraph data or credentials. Tests cover content-free plans/target exclusion, exact label masking and actual target rechecks, complete matched schedules, fresh adapter/reload contracts, weighted losses, failed generations and isolated notebook preflight with training disabled.

## Manual use after reviewing the prepared experiment

1. Upload `kaggle/plan_realizer_pilot.ipynb` as a **new private notebook**.
2. Attach a **private Dataset containing only** `research_runs/plan_realizer_pilot_v1/input_pack/realizer_pack.json`. This file contains copyrighted source/reference text; do not publish it.
3. Default `RUN_TRAINING = False` permits CPU preflight only. It accesses no model weights or Secrets.
4. When choosing to authorize this bounded training experiment, enable internet, GPU T4 x2 and `HF_TOKEN`, change the final cell to `RUN_TRAINING = True`, and Save & Run All. Report the slug and version afterward.

Expected outputs: `/kaggle/working/plan_realizer_pilot_v1/`, including `evaluation.json`, `adapter_direct/`, `adapter_planned/`, `adapter_hashes.json`, `development_losses.json`, `training_log.json`, label/generation token audits, raw generations, full verifier results, metrics, runtime, run status and manifest. None of these future outputs is claimed to exist yet.
