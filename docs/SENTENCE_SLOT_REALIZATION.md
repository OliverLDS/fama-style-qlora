# Sentence-slot realization diagnostic

Status: prepared for manual Kaggle upload; no inference or training submitted.
This follows the failed concise-plan realization check. It preserves the earlier
notebook, input pack, code and results.

## Upload and run

1. Upload `kaggle/sentence_slot_realization.ipynb` into a new **private** Kaggle
   notebook. Suggested title: **Sentence Slot Realization**.
2. Attach the same private dataset containing `concise_plan_pack.json` that was
   used for Concise Plan Realization. No new data upload or annotation is needed.
   Its local location is
   `research_runs/concise_plan_v1/input_pack/concise_plan_pack.json`.
3. Enable the `HF_TOKEN` Kaggle Secret, internet, and **GPU T4 x2**. The first code
   cell exposes only GPU 0. No credentials belong in the notebook or input files.
4. Save & Run All. Share the notebook slug and version after it finishes.

The notebook checks the frozen input hash before Secret access or model loading.
It reads the Secret at runtime without printing it. Keep the notebook, attached
dataset and outputs private because the pack contains copyrighted references.
No CLI upload or run is performed by the builder.

## Comparison

The same four training pairs from four papers are retained, including the failed
six-sentence case and all three source/reference NLI flags. These examples have
already been inspected; this is a controlled diagnostic, not fresh validation.
Development-validation and reserved test examples remain unused.

| Condition | Input beyond the ordinary instruction and source |
|---|---|
| A: ordinary | None; return a paragraph |
| B: count only | Requested sentence count; return a paragraph |
| C: sentence slots | Same count instruction; return a JSON object with one sentence per array element |

B versus A tests count guidance without length, syntax or punctuation controls.
C versus B tests the output-format intervention. Both receive the same count;
only C receives the JSON contract. No reference wording, semantic ledger or
claim-to-slot allocation is added to any condition. Exact target counts are
reference-derived oracle information, not a learned prediction.

Twelve first-attempt generations use pinned base Gemma 3 4B in NF4, FP32 compute,
greedy decoding, seed 42, and the same 768-new-token budget in all conditions.
The larger budget accommodates JSON overhead; this is a new matched comparison,
not an exact replication of the earlier 512-token run. Condition order rotates
across examples. No adapter, optimizer, fine-tuning or structural mapper is used.

## Validation and scoring

C must return only `{"sentences": ["First sentence.", "Second sentence."]}`.
The validator rejects fences, commentary, duplicate/extra keys, wrong slot count,
empty/non-string elements, and slots failing surface sentence-boundary checks.
It joins the strings only for evaluation. It neither repairs the JSON nor splits
a long slot into the requested number of sentences. Surface boundary checks do
not prove grammatical completeness or atomic claim boundaries.

Raw generations are always retained. Invalid formatting is a first-attempt
failure and remains in the four-example denominator. Conditional style, copying
and NLI scores must be presented with the valid-output denominator; invalid
formats receive no style/copy score. NLI on malformed raw output is diagnostic
only and cannot make that output pass the combined format/count/semantic gate.
No repair attempts are implemented or budgeted. Empty or cutoff generations
stop the run with saved partial outputs; an incomplete run is not a success.

Independent bidirectional document/sentence-unit NLI runs after unloading Gemma,
with the existing calibration controls and source/reference checks. Over-context
checks abstain rather than truncate. NLI, copying and the frozen reference-profile
style distance remain imperfect proxies. Passing NLI is not proof that claims
were preserved. All original source/reference flags remain reported.

The main decision measure is first-attempt **valid format + correct count + NLI
proxy pass**, out of four per condition. Inspect style change and copying
separately. A shape gain accompanied by semantic regression does not justify a
mapper. Even a clean four-case result would support a larger inference check,
not establish generalization or authorize GPU training.

## Outputs and local checks

Kaggle writes to `/kaggle/working/sentence_slot_realization_v1/`, including
`evaluation.json`, `metrics.private.jsonl`, `generations.private.jsonl`, frozen
jobs, token audits, NLI controls, run status and a file-hash manifest. Generation
tokens/time, NLI time, runner time and peak GPU memory are recorded.

Public implementation: `src/fama_style/rhetoric/sentence_slots.py`.
Its runtime is isolated from the earlier diagnostic so its frozen sources and
notebook remain unchanged. ML imports occur only inside the inference runner
named `run_training` for the repository's lazy-import convention; it invokes no
training code.

```sh
python3 scripts/build_sentence_slot_notebook.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m fama_style.rhetoric.sentence_slots \
  --dry-run --output-dir /private/tmp/sentence-slot-fresh-dry-run
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

Synthetic dry runs verify wiring only, make no model calls and are explicitly
labeled. The exported notebook is self-contained for manual UI upload. It embeds
runtime code and the approved pack hash, without paragraph data or credentials.
