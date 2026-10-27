# Separate group realization and paragraph assembly

**Prepared and CPU-tested; no inference has run.** This follows the [whole-paragraph grouping result](BOUND_GROUPING_REALIZATION_V1_RESULTS.md), where the oracle plan matched counts in 2/9 outputs and did not establish substantial style improvement. The new diagnostic moves group execution outside the model's whole-paragraph planning task. It does not train a mapper or modify existing experiments.

## Matched strategies

| Strategy | Calls | Paragraph candidates |
| --- | ---: | ---: |
| A: whole paragraph | Nine calls repeating the previous oracle prompts exactly | Nine |
| B: separate groups | 33 calls, one per planned group and sample | Nine assembled paragraphs |
| Total | **42 generation calls** | **18 paragraph candidates** |

Use the same three training examples and their source-bound oracle groupings: 6→4, 7→5 and 4→2 sentences. Each strategy has three paragraph samples per example, identified by seeds 41/42/43. A retains the prior prompts, sampling seeds and historical output hashes as controls. B derives deterministic per-group sampling seeds from the protocol, source ID, paragraph seed and group index; call order does not select the result.

Every call receives the full source passage and the same grouping plan. A requests the whole paragraph. B appends a scope instruction to realize only the current group as one sentence, using other source material only for context. B never receives earlier generated pieces. Source references and evaluation-only obligation notes are excluded from all prompts.

Both strategies retain the same concrete style specification and allowance for minor semantic drift. This compares decomposition plus scoped instructions with whole-paragraph execution; it does not isolate a single prompt token or constitute deployment/learned-mapper performance. These are selected training cases whose references contribute to the existing style profile, not held-out performance data. Development-validation and reserved-test splits are not read by preparation.

## Output budgets and assembly

Both strategies receive an allowance of **768 generated tokens per paragraph**. A receives that budget in one call. B allocates it across groups in proportion to source-group word counts, using deterministic largest-remainder rounding. Exact per-call caps are saved in jobs and token audits. This matches the output allowance, not input-token or wall-clock costs: B repeats context and makes more calls. Tokens, calls, generation time and memory are reported separately.

The assembler joins returned group strings in the prescribed order with a space. It does not rewrite transitions, fix grammar, add punctuation, replace failed pieces with source text or ask a model for a coherence pass. There are no retries or candidate selection within a paragraph sample.

Multi-sentence, boundary-ambiguous, badly formatted or unpunctuated pieces remain verbatim and are flagged. Their assembled paragraph remains available for assessment. An empty or token-cutoff response leaves the run incomplete with partial generations retained; a truncated piece is never presented as a successful assembly. No decoder constraint forces a sentence count.

## Evaluation

All 18 paragraph views are assessed independently of format success. B's generation-call assignment supplies known group-piece correspondence even when a piece is not one sentence. Each piece is checked against the source text of its assigned group, with full verdicts and text hashes retained. This guarantees which source group the call was asked to realize; it does not guarantee complete, correct claim coverage.

A keeps the previous positional checks, which are available only when sentence boundaries are unambiguous and requested counts match. B and A therefore have different availability of localized diagnostics; compare whole-paragraph semantic judgments and all-example denominators as well. Group-level consistency is neither semantic proof nor an automatic acceptance gate.

Gemma unloads before the pinned verifier loads. Both strategies receive whole-paragraph verification, copying and style proxies. B additionally reports per-piece sentence/format checks and assembly validity. Semantic and style grades remain pending until all outputs are read against sources and references. Minor drift stays eligible. Cross-sentence coherence, repetition and reference resolution need assessment alongside individual-group fidelity; assembly alone cannot establish them.

A useful finding would be more reliable realization of the known groups with reference-grounded style benefit under the existing graded objective, after accounting for coherence and extra cost. A mechanically valid assembly or lower copying alone is insufficient. Repeated outputs and pieces are not independent paragraph-level improvements. Do not infer statistical superiority or proceed to a full mapper/fine-tuning run from nine candidates per strategy.

## Model and reproducibility

Use the established base `google/gemma-3-4b-it`, revision `093f9f388b31de276ce2de164bdc2081324b9767`, with no adapter, NF4/FP32, temperature 0.6, top-p 0.9, top-k disabled and a 4096-token context guard. No prompt is truncated. The original Gemma 3 1B training CLI remains unchanged. Kaggle Secrets supplies `HF_TOKEN` only at execution; its value is absent from all code and packs and is never printed.

The notebook embeds the exact runtime, pins the private input pack hash and runs preflight before model access. Full paragraph and group verdicts, raw 42-call outputs, 18 paragraph views, budgets, sampling seeds, formats, costs and hashes are retained for CPU replay. Historical A matches are reported rather than assumed.

## Manual Kaggle workflow

1. Import `kaggle/groupwise_realization.ipynb` into a **new private notebook**, for example `groupwise_realization`.
2. Attach a private Dataset containing **only** `research_runs/groupwise_realization_v1/input_pack/groupwise_pack.json`.
3. Enable internet, **GPU T4 x2**, and the existing `HF_TOKEN` Kaggle Secret. Only GPU 0 is exposed.
4. Save & Run All. Download `/kaggle/working/groupwise_realization_v1/`, especially `evaluation.json`, `generations.private.jsonl`, `paragraphs.private.jsonl`, `metrics.private.jsonl`, `verifier_results.private.jsonl`, `group_verifier_results.private.jsonl`, `runtime.json`, `token_audit.json`, `run_status.json` and `manifest.json`. Keep raw/evidence files private.

No Kaggle upload, execution or GPU training was submitted by preparation.

## CPU checks

```sh
PYTHONPATH=src python3 -m fama_style.rhetoric.groupwise_realization --dry-run \
  --output-dir /tmp/groupwise_synthetic
python3 -m unittest discover -s tests -v
python3 train.py --check
```

With the verified private prior pack/results available, prepare a fresh pack using `--prepare-pack`; export with `scripts/build_groupwise_notebook.py`. Implementation is separate in `src/fama_style/rhetoric/groupwise_realization.py`.

All **235 CPU tests** passed, including exact control prompts, reference exclusion, deterministic matched budgets, order-independent assembly, retained format failures, source-piece verdict binding/replay, incomplete-output rejection and isolated embedded-runtime dry-run/preflight. The original training CLI check passed without downloading weights or starting training. These checks do not establish GPU generation quality or prove that every group will finish within its allocated cap.
