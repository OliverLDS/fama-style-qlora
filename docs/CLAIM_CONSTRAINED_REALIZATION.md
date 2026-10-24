# Claim-constrained realization diagnostic

Status: prepared and CPU-checked; no model inference or GPU job submitted.
This follows the sentence-slot diagnostic. Existing experiments, notebooks,
datasets and results are preserved.

## Run through Kaggle's UI

1. Upload `kaggle/claim_constrained_realization.ipynb` into a **new private**
   notebook. Suggested title: **Claim Constrained Realization**.
2. Attach the existing private dataset containing `concise_plan_pack.json`.
   No new annotation or dataset upload is needed. The local file is
   `research_runs/concise_plan_v1/input_pack/concise_plan_pack.json`.
3. Enable internet, the `HF_TOKEN` Kaggle Secret, and **GPU T4 x2**. The first
   code cell exposes GPU 0 only.
4. Save & Run All, then share the notebook slug and version.

The notebook checks the approved pack hash before Secret access or model loading.
Credentials are retrieved at runtime without printing their values. The pack,
notebook outputs and generated text must remain private because they include
copyrighted reference material. The builder performs no upload or submission.

## Matched comparison

All four conditions use exactly the same four previously inspected training
examples and reference-derived sentence counts. No validation or reserved test
examples are read. This is an oracle-count diagnostic, not a generalization
estimate or a learned structural mapper.

| Condition | Source-only preservation ledger | Output |
|---|---|---|
| A | Absent | Count-controlled paragraph |
| B | Present | Count-controlled paragraph |
| C | Absent | Count-controlled sentence slots |
| D | Present, identical to B | Count-controlled sentence slots |

B versus A and D versus C measure the constraint intervention within each format.
C versus A and D versus B measure the format intervention. There is no new
ordinary-prompt arm: the earlier run supplies context, while all causal contrasts
for this experiment use these newly matched conditions. Extra ledger tokens and
their generation cost are part of the intervention, not a length-matched placebo.

Sixteen first attempts use the same pinned base Gemma 3 4B revision, NF4 with
FP32 compute, greedy decoding, seed 42 and 768-new-token output budget. Condition
order rotates. No adapter, optimizer, QLoRA training, mapper fitting or automatic
repair is used. Empty/cutoff generations stop the run and preserve partial files;
such a run is incomplete. Context overflow abstains or stops rather than truncates.

## Source-only constraints

The existing surface ledger supplies sentence IDs, complete source evidence and
quoted cues for quantities, negation, modality, scope, conditions and causality.
Explicit upper/lower-bound expressions receive a finite direction label. The
prompt requires preserving these relationships and discourages unsupported
explanations, mechanisms, recommendations and conclusions. The full source remains
authoritative when the surface aid is incomplete.

Every ledger evidence span comes from the supplied input. The generator never
receives target wording, target claims or reference-based corrections. The ledger
does not claim atomic proposition parsing, actor/coreference resolution or
independently verified semantic alignment. It copies some source evidence twice;
increased copying is therefore a possible failure mode that must be measured.
Source transcription and existing rewrite quality are not silently repaired.

## Predeclared format handling and semantic evaluation

Slot outputs use a single JSON object with key `sentences`. Exactly one complete
outer fence with opening ` ```json ` and closing ` ``` ` may be removed before
parsing; no other text is removed, repaired or generated. Raw format validity,
fence removal and normalized validity are reported separately. Wrong slot counts,
multiple sentences per slot, invalid keys and other malformed structures remain
failures. Boundary checking is a surface heuristic, not proof of grammatical
completeness. Paragraph outputs do not receive JSON normalization.

An independent pinned NLI model runs on the same GPU after Gemma is unloaded. It
assesses both directions at document and sentence-unit level, including joined
slot text when it can be parsed. A parseable wrong-count array still undergoes
semantic assessment but cannot pass the combined format/count gate. Unparseable
raw output is diagnostic only. Original source/reference NLI checks remain
reported for all four pairs, including the existing three flags.

In addition to the existing three equivalent and seven changed-meaning controls,
four synthetic mutations test upper-bound reversal, quantity changes, strengthened
uncertainty/scope and reversed causality. If any control fails or abstains, the
calibrated NLI pass totals and combined semantic pass count are **unavailable**,
not zero model successes. Uncalibrated consistency counts remain diagnostic.
Even passing all controls does not establish the checker's domain accuracy.

Report semantic consistency first, then the joint format/count/semantic proxy,
copying and style-distance change. Conditional metrics must show the normalized
valid-output denominator; malformed outputs stay in the overall four-example
denominator. Protected surface-cue mismatches are warning flags, not verified
semantic errors. Neither the NLI score nor a matching cue proves preservation of
every claim. Do not infer improved style from count compliance alone.

## Decision and cost

The hypothesis is that explicit source constraints reduce drift without forcing
near-copying or preventing meaningful stylistic change. Improved shape alone is
insufficient. A semantic regression, increased copying, or control-test failure
must be reported rather than hidden by an aggregate style score. Four inspected
training cases can support a subsequent inference check, not justify a mapper,
validate author-style learning, or authorize GPU training.

The prior twelve-generation run took 4.57 minutes inside the runner. This run has
sixteen generations, longer constrained prompts and additional semantic controls;
roughly 6–12 runner minutes is a planning estimate, not a measured result. Queue,
installation and cold-download time are additional. Only one T4 is used.

## Artifacts and reproduction

Kaggle output directory: `/kaggle/working/claim_constrained_realization_v1/`.
Main files: `evaluation.json`, `metrics.private.jsonl`, `generations.private.jsonl`,
`jobs.private.jsonl`, `nli_controls.json`, `token_audit.json`, `run_status.json`
and a file-hash manifest. Prompts retain the exact generated ledger, source hash
and prompt hash. The original pack hash fixes the examples and reference counts.
Tokens, time, model revisions, software versions and peak GPU memory are recorded.

Implementation: `src/fama_style/rhetoric/claim_constrained.py`; tests:
`tests/test_claim_constrained.py`. ML imports remain inside the inference runner
named `run_training` for the repository's lazy-import convention. It calls no
training code. The original Gemma 3 1B training workflow is unchanged.

```sh
python3 scripts/build_claim_constrained_notebook.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m fama_style.rhetoric.claim_constrained \
  --dry-run --output-dir /private/tmp/claim-constrained-fresh-dry-run
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

Synthetic dry runs verify wiring only and are explicitly labeled. The exported
notebook embeds code and the approved input hash, without data paragraphs or
credentials, and is self-contained for manual upload.
