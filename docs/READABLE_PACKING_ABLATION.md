# Readable elementary edits: two-case format ablation

**Prepared, not executed. Inference only.** The previous elementary-plan check left prompt usability unresolved: sentence and claim IDs were largely bound through character offsets, and output slots were expressed as JSON. This check compares that frozen format with readable source bindings and direct sentence instructions.

## Fixed design

Four greedy generations on frozen base Gemma 3 4B, without an adapter:

| Previously inspected training case | A | B |
| --- | --- | --- |
| Fama–Jensen hierarchy merge | Original offset/JSON edit-guided prompt | Readable bindings and direct plan |
| Market-efficiency finding/presumption split | Original offset/JSON edit-guided prompt | Readable bindings and direct plan |

A's two prompts are byte-for-byte identical to the earlier edit-guided prompts. They are rerun alongside B, with condition order reversed on the second example. Source/reference strings, claim identities, epistemic labels, protected evidence, relation annotations/certainty, operation assignments, output slots, concrete style instruction, model/revisions and decoding remain fixed. No source is rewritten and no new annotation is created.

B labels each verbatim source sentence as `S1`, `S2`, etc. Each claim lists its existing source sentence IDs and exact source excerpts, with protected quantities/conditions/uncertainty/attribution evidence retained. The renderer validates every quotation against its character span; repeated evidence retains its one-based occurrence. It adds no semantic relationship. The complete original source remains available.

B translates the existing MERGE/SPLIT/KEEP slots into numbered output-sentence instructions. The explicit sentence count is the count already implied by the old slots, not a new constraint. The same operation descriptions, composite-claim rule, untouched-sentence rule and pronoun/attribution instruction apply. No stricter preservation instruction is introduced.

Readable source binding and plan presentation change **together**. This is a joint format ablation; it cannot distinguish their individual contributions. Readable evidence also changes prompt length and repetition. Any improvement is evidence about the presentation bundle, not proof that numeric offsets alone caused the earlier failures.

Both cases were selected after inspecting the earlier outputs. This is a retrospective diagnostic on training examples, not a new held-out result. There is no reserved probe, development-validation or reserved test data in this check. Reference-derived grouping and paired claim annotations remain oracle information; original reference wording is evaluation-only and never supplied to Gemma.

## Execution and evidence

The inference body matches the established elementary-check runtime. Pinned Gemma/NLI revisions, NF4/FP32, GPU 0 only, greedy decoding, seed 42, 768 output-token cap and 4,096-token total budget are unchanged. All prompts receive an actual tokenizer audit before Gemma weights load. Overlong prompts fail without truncation. CPU checks do not estimate actual Gemma token counts or promise that a prompt fits.

Weights are frozen. There is no optimizer, adapter, training, retry, repair or reference fallback. Empty/cutoff outputs remain failed attempts. Gemma unloads before the smaller NLI model loads. Secrets are retrieved through the existing Kaggle Secrets helper, never printed or embedded; safe phase reporting records exception types rather than potentially sensitive details.

The output includes raw generations, frozen jobs, tokens, complete source/output-bound verifier and claim-localization evidence, surface features, copying, paired metrics, runtime, run status and hashes. Partial failures remain in paired denominators. NLI/localization and surface style distances remain fallible diagnostics, not automatic acceptance criteria or author-style classifiers. Drift severity and acceptability remain unassigned until a post-run evidence audit.

Inspect the actual operations before calling this successful:

- **Hierarchy:** A already expressed the initial merge. B must improve complete organization and retain the other claims without appending unsupported explanations; merely making the same first merge is insufficient.
- **Market efficiency:** B must separately express the cited finding and the author's own earnings presumption, retain attribution and the separate IPO presumption, and preserve the conditions on the correction process.
- Weigh actual expression/style improvement separately from drift. Minor drift may be acceptable when the benefit is worthwhile; invented mechanisms or lost attribution cannot be dismissed as formatting changes. A sentence-count match alone is insufficient.

If the readable version helps, expand the readable-format realization check across papers before training a mapper or adapter. If it fails, apply the same readable tasks to a stronger instruction-following realizer before interpreting the result as a limit of rhetorical planning. Neither outcome has occurred or is claimed here.

## Manual Kaggle run

1. Upload `kaggle/readable_packing.ipynb` as a **new private notebook**, for example `readable_packing`.
2. Attach a **private Dataset containing only** `research_runs/readable_packing_ablation_v1/input_pack_v3/readable_pack.json`. No adapter or previous notebook output is needed.
3. Enable Internet, **GPU T4 x2** and the existing **HF_TOKEN** Secret for this notebook. Save & Run All; only GPU 0 is used.
4. Share the notebook slug/version after completion. Keep the input pack and all paragraph outputs private.

Download `/kaggle/working/readable_packing_ablation_v1/` in full: `evaluation.json`, `generations.private.jsonl`, `jobs.private.jsonl`, `metrics.private.jsonl`, `verifier_results.private.jsonl`, `token_audit.json`, `runtime.json`, `run_status.json`, `manifest.json`.

The active pack is **input_pack_v3**. Earlier local preparation drafts remain preserved and are not the upload inputs. The self-contained notebook is 287,082 bytes, below the conservative 900,000-byte source/file ceiling. It contains runtime code, not the private pack. No upload or submission was performed.

## CPU reproduction

```bash
PYTHONPATH=src python3 -m fama_style.rhetoric.readable_packing \
  --prepare-pack \
  --parent-pack research_runs/elementary_packing_realization_v1/input_pack/elementary_pack.json \
  --output-dir research_runs/readable_packing_ablation_v1/input_pack_v3
python3 scripts/build_readable_packing_notebook.py
python3 -m unittest discover -s tests -v
python3 train.py --check
```

Preparation requires a fresh directory. Repeated preparation must use a new suffix, with the builder updated to point at it. Pack/runtime/job hashes freeze the comparison. All **311 CPU tests passed**, including exact historical-prompt identity, source-evidence binding, plan coverage, failed-output denominators, evidence identity, frozen inference-body equality and an isolated notebook dry run without ML imports. The original CLI check passed. No model weights were downloaded, Secrets accessed or model calls executed during preparation. Existing experiments, training code and training gates remain unchanged.

Implementation: `src/fama_style/rhetoric/readable_packing.py`; tests: `tests/test_readable_packing.py`; builder: `scripts/build_readable_packing_notebook.py`. [Prior run and motivation](ELEMENTARY_PACKING_REALIZATION_V1_RESULTS.md).
