# Groupwise realization v2

**Prepared and CPU-tested; not executed.** This fixes the [v1 prompt-scope failure](GROUPWISE_REALIZATION_V1_FAILURE.md). It preserves the failed run and frozen implementation and uses a separate module, notebook and private pack.

## Changes

1. **Local group input.** Each B call receives only its assigned source-group text. The prompt begins with an explicit one-sentence task and contains no whole-passage input, all-group plan or instruction to return a paragraph. It retains the concrete style specification and the same semantic tolerance. References, other groups and evaluation-only notes are excluded.
2. **Recorded generation failures.** An empty or token-cutoff response remains in raw outputs with its failure reason. The remaining calls continue without retrying or increasing a cap. A paragraph depending on a failed response has `available: false`, `paragraph: null` and no semantic/style score. It is never assembled from truncated text or replaced with the source.
3. **Honest completion accounting.** All 18 paragraph attempts remain in the records. Each arm reports available/unavailable paragraphs and failed calls. Style means explicitly use available paragraphs only. The overall status distinguishes `inference_complete` from `execution_complete_with_generation_failures`; `all_candidates_complete` is separate from execution completion. Model/runtime exceptions still stop the run with partial artifacts.

The new group prompts also remove the full source context. Missing antecedents and paragraph coherence are therefore genuine limitations to assess. This is a revised realization procedure, not an isolated test of moving the same prompt text to another position. The model is told to retain supplied referents rather than invent missing context. No earlier generated piece is fed into later calls.

## Unchanged controls

- Same three selected training pairs and oracle source-group assignments.
- Nine whole-paragraph calls with exactly the v1 control prompts and paragraph seeds 41/42/43.
- Thirty-three local-group calls, producing at most nine complete assembled paragraphs: **42 calls / 18 paragraph attempts total**.
- The same deterministic group sampling seeds as v1, despite the new protocol name.
- **768 output tokens per paragraph attempt**, with unchanged proportional group allocations. Failed caps are not expanded after seeing their output.
- Same pinned base Gemma 3 4B, NF4/FP32, temperature 0.6, top-p 0.9, top-k disabled and 4096-token context guard. No adapters, training, forced one-sentence decoding, automatic text repair or coherence rewrite.
- Same graded style/drift objective: minor drift remains eligible, and NLI is diagnostic rather than a binary acceptance gate.

The three examples and reference-derived plans are training/oracle diagnostics, not held-out or learned-mapper performance. Preparation does not read development-validation or reserved-test files.

## Evaluation

Available B paragraphs join all returned pieces unchanged in group order. Multi-sentence, ambiguous or unpunctuated pieces are retained and flagged; they do not by themselves abort the run. Full paragraph and per-group verdicts are saved for available candidates. When a paragraph is unavailable, its raw pieces remain inspectable but it receives no paragraph verdict, inferred group quality or source-copy fallback.

Count/assembly success, semantic drift, stylistic benefit and cross-sentence coherence remain separate. Judge all completed paragraphs and report the missing-candidate counts alongside them. An available-only improvement cannot conceal failed attempts. No best-of-three result or generation-quality claim is made before execution.

## Manual Kaggle rerun

1. Import **`kaggle/groupwise_realization_v2.ipynb`** into a new private notebook, for example `groupwise_realization_v2`.
2. Attach a private Dataset containing **only the new** `research_runs/groupwise_realization_v2/input_pack/groupwise_pack.json`. Its filename is the same as v1 but its contents/hash differ. Do not attach both old and new packs.
3. Enable internet, **GPU T4 x2**, and the existing `HF_TOKEN` Kaggle Secret. Only GPU 0 is exposed. No credential value is in the pack or printed by code.
4. Save & Run All. Retrieve `/kaggle/working/groupwise_realization_v2/`, including `evaluation.json`, `run_status.json`, raw generations, paragraph views, metrics, full verdicts, token audits, runtime report and manifest. Keep full text and evidence private.

A successful Kaggle execution may still contain failed generation samples. Inspect `all_candidates_complete`, `unavailable_paragraph_candidates` and each arm's failure counts before interpreting the quality metrics. No rerun was uploaded or submitted by this preparation.

## Local validation

Implementation is `src/fama_style/rhetoric/groupwise_realization_v2.py`; builder is `scripts/build_groupwise_v2_notebook.py`.

```sh
PYTHONPATH=src python3 -m fama_style.rhetoric.groupwise_realization_v2 --dry-run \
  --output-dir /tmp/groupwise_v2_synthetic
python3 -m unittest discover -s tests -v
python3 train.py --check
```

All **245 CPU tests** passed. The ten v2 tests additionally verify local-only prompts, removal of the paragraph-output instruction, retained seeds/budgets/control prompts, failed-candidate accounting and real-mode completion with recorded failures. The embedded notebook ran an isolated CPU preflight. The original training dry-run passed without model weights or GPU access. These checks do not guarantee the model will follow the revised group instructions.
