# Source-sentence-preserving edit diagnostic

Status: prepared and CPU-tested, **not run on Kaggle**. Existing datasets,
notebooks, adapters, verifier calibration and experimental outputs remain intact.

## Upload and run

1. Upload `kaggle/sentence_preserving_edit.ipynb` into a **new private notebook**.
   Suggested title: **Sentence Preserving Edit**.
2. Create a new **private Dataset** containing only
   `research_runs/sentence_preserving_edit_v1/input_pack/sentence_edit_pack.json`
   and attach it. Suggested title: **Sentence Edit Inputs**.
3. Enable internet, **GPU T4 x2** and the notebook's **HF_TOKEN** Secret.
   Only GPU 0 is visible. The token is read at runtime without printing it.
4. Save & Run All and share the notebook slug/version after completion.

Keep the input, notebook and outputs private: they contain source/reference
paragraphs and original prompts. No notebook was uploaded or submitted by the
builder. The approved pack SHA-256 is
`ff8d1e10404a0e6d9497bbf851f1f61ddcae345d32abe1c9e1e6cbe6379661af`.

## Comparison and scope

The same two exposed development failures are retained, with seven and three
source sentences. Neither case is a fresh validation example. The reserved paper
test split is never read. There are **four first-attempt model generations**:

| Condition | Model output |
|---|---|
| A: baseline | Paragraph under the previous sentence-evidence/no-grouping prompt, unchanged byte for byte |
| B: edit script | Same prompt plus a local sentence-replacement protocol |

Both conditions use the same pinned base Gemma 3 4B, NF4/FP32 compute, greedy
decoding, seed 42, 768-new-token budget and context guard. No adapters or training
are used. The JSON protocol has output-token overhead within that same per-call
budget, which must be included in cost comparisons. Order alternates across cases.

This tests an editing protocol, including its format and preservation instructions;
it does not isolate one extra sentence-boundary instruction from all other prompt
changes. It is not a learned rhetorical mapper or a full taxonomy experiment.

## Edit contract

B returns only this form, with source IDs already supplied in the evidence:

```json
{"edits":[{"claim_id":"c1","replacement":"One edited sentence."}]}
```

Each ID refers to a source **sentence**, which may contain multiple substantive
claims. Only replacement at that position is permitted. The code cannot insert,
delete or reorder sentence positions. Omitted IDs retain the source text. The
array can list IDs in any order; application always uses source order.

Each replacement must pass a one-sentence surface contract and end with sentence
punctuation. The assembled paragraph must retain the detected source count.
Unknown/duplicate IDs, duplicate JSON keys, extra fields, empty replacements and
multi-sentence replacements are rejected. The detector is heuristic; abbreviations
and unusual punctuation can cause format rejection. It does not prove grammatical
completeness or preservation of the source's meaning.

One complete outer JSON code fence may be removed under the declared protocol.
Raw versus normalized contract validity is recorded. Other malformed scripts are
not repaired, extracted from commentary or converted into apparently successful
source copies. Their proposed/guarded paragraphs are unavailable and their rows
remain in the two-example denominator. Empty/cutoff generations stop the run with
preserved partial files. No extra generation attempts are made.

An empty edit list is valid when the model finds no safe useful changes. It is
reported as unchanged input, not successful style transformation. The generator
never sees reference wording, reference sentence structure or target claims.

## Proposed and filtered variants

One B generation yields two derived paragraph variants, not two independent
generations:

- **B proposed:** apply every structurally valid replacement.
- **B guarded:** apply only changed replacements that pass the unchanged v3
  source-sentence verifier and a separate full-source-to-replacement NLI support
  check. Risky or uncertain replacements leave the original sentence intact.

No-op edits are recorded but do not count as accepted nontrivial edits. The
support check is one-way from the full source, while sentence verification is
bidirectional; a single edited sentence is not required to entail the entire
paragraph. Both proposed and guarded paragraphs also receive the full v3
paragraph check. Per-edit decisions, accepted counts, risk rejections, abstentions
and no-ops are retained.

This is explicit filtering of a valid proposal, not a hidden rewrite or retry.
Sentence positions are structurally controlled, but replacements can still change
claims. Unit-level proxy consistency does not imply paragraph-level consistency,
especially with cross-sentence references. The verifier remains fallible, and a
synthetic calibration pass does not certify economics prose.

Fallback can increase copying. Report exact/token-normalized copies, similarity,
near-copy threshold 0.90, accepted edit counts and style-distance change alongside
semantic status. The joint proxy requires valid format, verifier consistency,
non-copying and positive style gain; B also requires a nontrivial applied edit.
All-fallback and empty scripts therefore cannot count as style successes.

The baseline's output hash is compared with the preceding run. If replication
fails, report it and inspect the controls before attributing changes to editing.
The format/boundary contract is imposed by application; compliance is not evidence
that the model learned Fama/coauthor style.

## Execution, outputs and verification

The notebook checks input, implementation and job hashes, then repeats the passed
v3 synthetic calibration as a runtime canary before Secret access. NLI is unloaded
before Gemma generation and reloaded after Gemma is unloaded. The two models are
not held on GPU 0 simultaneously. Context overflows abstain or stop without
truncation. The original Gemma 3 1B training CLI is unchanged.

Outputs are under `/kaggle/working/sentence_preserving_edit_v1/`, including
`evaluation.json`, `metrics.private.jsonl`, `edit_assessments.private.json`,
`generations.private.jsonl`, `nli_probability_cache.private.jsonl`, prompt/token
audits, run status, runtime canary and a recursive hash manifest. Metrics contain
three variants per case; cost counts the four actual generations only. Detailed
edits and paragraph text must remain private. Nothing is added to training data.

Implementation: `src/fama_style/rhetoric/sentence_edit.py`. ML imports remain
inside the inference runner named `run_training`; it calls no trainer. CPU tests
cover exact baseline prompts, reference exclusion, immutable sentence positions,
strict script parsing, explicit fallback, no-op/copy accounting, invalid-script
denominators, guarded-text derivation, cutoff detection and standalone export.
Mock predictors test mechanics only, not model accuracy.

```sh
python3 scripts/build_sentence_edit_notebook.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m fama_style.rhetoric.sentence_edit \
  --dry-run --output-dir /private/tmp/sentence-edit-new-dry
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

No quality result is claimed before execution. `training_ready` remains false;
successful execution or source copying does not authorize further fine-tuning.
