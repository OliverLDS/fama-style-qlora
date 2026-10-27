# Elementary packing realization check v1

**Prepared, not executed. Inference only.** This experiment tests whether the new elementary edits improve organization and expression enough to justify any claim drift, before expanding annotation or training a mapper.

## Matched comparison

Use the frozen base Gemma 3 4B instruction model, without an adapter, on six previously annotated training paragraphs from six papers. Each paragraph has two first-attempt greedy generations:

| Condition | Generator input |
| --- | --- |
| A: ordinary rewriting | Concrete name-free style instruction and common source evidence |
| B: elementary edits | Identical instruction and evidence, plus reference-derived MERGE/SPLIT/KEEP operations |

There are **12 calls**, with condition order alternating across paragraphs. Five paragraphs are new oracle candidates. The sixth is the previously reserved diagnostic probe and remains separate in every aggregate. It is not a new training label or a guarantee of successful preservation. The sample includes the one independently bound split and five merge paragraphs. Some paragraph plans contain multiple merges.

Selection is fixed before generation: force the sole split and reserved probe; for each remaining paper choose the shortest existing compiled prompt, with case-ID ties. This gives one paragraph each from Market Efficiency 1997, Disappearing Dividends 2000 (probe), New Lists 2003, Volatility Lessons 2018, Separation of Ownership and Control 1983 and Agency Problems 1980. Length-based selection is a low-cost convenience sample, not a representative estimate.

Both arms share source text, sentence offsets, source claim offsets, epistemic status, protected semantic evidence and annotated relations with their original certainty. Compact character offsets avoid repeating evidence quotations. B alone receives the finite operation catalogue and whole-claim output grouping. Reference text, target quotes and annotator notes never enter a generation prompt. Reference grouping and the paired claim inventory remain oracle information; neither arm is a deployable source-annotation pipeline. No claim about held-out generalization is possible from these training examples.

## Execution controls

The model/revisions and NF4/FP32 placement reuse the established inference setup. Only GPU 0 is exposed even when Kaggle offers T4 x2. The Gemma output cap is 768 tokens and total context budget is 4096. A tokenizer audit of all prompts runs before loading Gemma weights. Overflow fails explicitly without truncating; approximate character lengths are not reported as measured token counts.

All weights are frozen. There is no optimizer, backward pass, adapter, retry, automatic rewrite repair or reference fallback. Empty/cutoff responses remain recorded failures and the remaining calls continue. Gemma unloads before the separate NLI model loads. Phase messages and 30-second heartbeats show progress. Kaggle Secrets supplies `HF_TOKEN` only at runtime; its value is never printed or written to the input pack. Failure reporting records the exception type and suppresses exception details that could contain credentials or signed URLs.

## Separate outcomes and decision

The runner retains raw generations and records:

- **Organization:** target sentence count and full source-claim/output-sentence NLI matrices, with tentative slot/grouping matches. Unknown or ambiguous localization abstains. Exact sentence count alone is insufficient evidence of correct organization.
- **Copying:** source lexical overlap, with the existing 0.9 near-copy diagnostic; private reference overlap is evaluation-only.
- **Style:** individual surface features and change in distance to the frozen 247-reference training profile. Sentence packing directly affects these features, so an apparent gain can be mechanical. These are neither independent style judgments nor an author classifier.
- **Claim drift:** full bidirectional semantic-verifier evidence, source-coverage/output-support alignments, surface warnings, bounded actor/quantity/causality guards and protected claim annotations. NLI flags are fallible risk indicators, not drift-severity labels or an automatic veto.
- **Cost/failures:** all attempts, token counts, generation time, memory and version information. Cutoffs never disappear from denominators.

The paired evaluation exposes organization matches, style-distance change, copying change and verifier status together. It deliberately leaves drift severity and acceptability unassigned rather than treating a binary verifier verdict as an answer to the tradeoff question. A post-run automated evidence audit must distinguish minor wording/force changes from lost conditions, reversed causality, altered actors or numerical claims. No human-rating step is required.

A provisional positive signal would be worthwhile expression/style improvement **beyond sentence-count changes** in at least two of the five new candidates from different papers, with clearer requested organization and no increase in cases exhibiting substantial semantic drift. Minor drift can be acceptable when the style benefit is substantial; report the concrete changes and sensitivity to that judgment. Report the probe separately. This is a small mechanism check, not significance testing or a training-readiness gate. If only sentence counts improve, or style gains depend on substantive drift, investigate realization/schema limitations before collecting more labels or training.

## Run manually in Kaggle

1. Upload `kaggle/elementary_realization.ipynb` as a **new private notebook**, for example `elementary_realization`.
2. Create/attach a **private Dataset containing only** `research_runs/elementary_packing_realization_v1/input_pack/elementary_pack.json`. No adapter or earlier notebook output is needed.
3. Enable Internet, GPU T4 x2 and the existing **HF_TOKEN** Secret for this notebook. Save & Run All. Keep the notebook, data and outputs private.
4. Share the notebook slug and version after completion.

Expected directory: `/kaggle/working/elementary_packing_realization_v1/`, containing `evaluation.json`, `generations.private.jsonl`, `jobs.private.jsonl`, `metrics.private.jsonl`, `verifier_results.private.jsonl`, `token_audit.json`, `runtime.json`, `run_status.json` and `manifest.json`. Download the complete folder rather than only the summary. References and generated paragraph text must remain private.

The notebook is self-contained and below the 900,000-byte source/file budget. It embeds runtime code, not the private input pack. Uploading it does not change existing notebooks or their Secret settings; this preparation performs no Kaggle upload or submission.

## CPU reproduction

```bash
PYTHONPATH=src python3 -m fama_style.rhetoric.elementary_realization \
  --prepare-pack \
  --compiled-dir research_runs/elementary_packing_edits_v1/compiled_v1 \
  --package research_runs/operation_expansion_v1/package_v1 \
  --output-dir research_runs/elementary_packing_realization_v1/input_pack
python3 scripts/build_elementary_realization_notebook.py
python3 -m unittest discover -s tests -v
python3 train.py --check
```

Preparation requires a new output directory; use a new version suffix for repeated preparation and update the builder's pack path accordingly. Pack and runtime hashes freeze the design. CPU preflight/synthetic runs do not load weights, access Secrets or assess model quality. All **303 CPU tests passed**, and the original CLI check passed. Training logic, the original 1B entry point, existing results and original operation-family gates remain unchanged. No development or reserved test rows were read, and no GPU job was submitted.

Implementation: `src/fama_style/rhetoric/elementary_realization.py`; tests: `tests/test_elementary_realization.py`; builder: `scripts/build_elementary_realization_notebook.py`. [Compiled edit inventory](ELEMENTARY_PACKING_EDITS.md).
