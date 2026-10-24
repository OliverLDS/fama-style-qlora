# Small candidate comparison under the graded objective

**Status: prepared and CPU-tested; GPU inference has not run. No training is proposed or submitted.**

The question is whether less restrictive prompting can produce a worthwhile reference-style improvement while tolerating minor semantic drift. This is an exploratory follow-up to the [saved-output reassessment](STYLE_DRIFT_TRADEOFF_V1_RESULTS.md), not a fresh held-out evaluation.

## Frozen comparison

- Four examples from the previously exposed 21-example development split. Sort by source token count, breaking ties by ID; select the lower median rank from each of four consecutive length strata. Zero-based selected ranks are 2, 7, 12 and 17. Selection uses no generated-output scores. The 45 reserved test examples are not accessed.
- A: direct rewrite instruction allowing small changes in emphasis and secondary wording while retaining the main argument.
- B: the identical instruction plus applicable training-reference-derived operations. No claims ledger, sentence quota, target paragraph wording, oracle plan or repair loop.
- Three candidates per input and arm, paired seeds 41/42/43: **24 generation jobs total**. Run arm order A/B on alternating cases and B/A on the others.
- Base `google/gemma-3-4b-it`, revision `093f9f388b31de276ce2de164bdc2081324b9767`; no adapter. NF4 with FP32 computation, one visible T4, temperature 0.6, top-p 0.9, top-k disabled, at most 768 new tokens and 4096 total context tokens. Prompts are never truncated. This sampled run is not directly comparable to earlier greedy runs as an isolated prompt effect.

This changes both arms' common instruction relative to earlier strict-preservation experiments. Its primary matched comparison is B versus A within this run. It does not isolate every cause of a difference from a historical output.

## Empirical guidance and its limits

The same finite patterns are counted independently in the 247 training sources and references:

| Pattern | Sources containing it | References containing it |
| --- | ---: | ---: |
| Explicit contrast markers | 86 | 118 |
| We/our research voice | 61 | 83 |

A pattern is eligible if reference prevalence is at least 25% and exceeds source prevalence by at least five examples. Application also requires the corresponding relation or voice to be explicitly present in the input. The operations make an existing contrast explicit at a clause/sentence boundary and retain an existing researcher subject rather than replacing it with impersonal nominal phrasing. They cannot license a new contrast, actor or causal relation.

Two selected cases support only the author-voice operation, one supports only contrast, and one supports neither. The last receives an empty plan and serves as a negative control for the extra prompt wrapper. Results must distinguish these cases; an aggregate B gain would not establish that a structural operation caused it. These shallow operations may offer little beyond an already clear source. This is deliberately a small fixed structural baseline, **not the learned three-level rhetorical mapper** envisioned in the broader research proposal.

## Evaluation and decision

Keep every completed candidate visible, including verifier risk and abstention. Report paragraph format, exact/near copying, style features, continuous warning pressure, four sensitivity-weight utilities and per-case nondominated alternatives alongside the unchanged-source control. Identical sampled outputs are not independent improvements. Malformed format remains flagged; it is not silently repaired. Empty or truncated generations leave the run incomplete with partial artifacts saved.

After Gemma unloads, the pinned v3 semantic verifier runs as a diagnostic. This run does not repeat its calibration suite or gate generation on verifier acceptance. Its probability scores are not true drift severity. **Qualitative reference-grounded style and semantic-drift grades remain pending until outputs are assessed.** The notebook does not claim a semantic-accuracy score, author-style accuracy or successful substantial-style transformation from surface proxies alone.

Assess all 24 outputs, rather than selecting a favorable proxy subset or reporting only a best-of-three result. Compare style and drift separately against the source and paired reference. Apply the existing [graded policy](style_drift_tradeoff_v1/policy.json), with minor drift eligible and high-cost changes visible. Report candidate-level outcomes and descriptive paired differences; this sample is too small and reused for a broad performance claim. Do not initiate fine-tuning solely because a numerical style proxy improves.

## Manual Kaggle workflow

1. Import `kaggle/style_tradeoff.ipynb` into a **new private notebook**.
2. Create a private Kaggle Dataset containing **only** `research_runs/style_tradeoff_candidates_v1/input_pack/style_tradeoff_pack.json`, and attach it.
3. Enable internet, **GPU T4 x2**, and the existing `HF_TOKEN` Kaggle Secret. The first cell exposes GPU 0 only; the token is retrieved at runtime and its value is never printed or stored in the pack.
4. Save & Run All. The notebook checks the uploaded pack and embedded runtime hashes before model access. It does not train or publish anything.
5. Retrieve `/kaggle/working/style_tradeoff_candidates_v1/`, especially `evaluation.json`, `generations.private.jsonl`, `metrics.private.jsonl`, `runtime.json`, `run_status.json` and `manifest.json`. Keep these outputs private. Provide the notebook slug and version for result inspection.

The notebook is self-contained and contains no real paragraphs or credentials. Its fixed pack hash identifies this exact preparation. No CLI upload or Kaggle execution has been performed.

## CPU reproduction

```sh
PYTHONPATH=src python3 -m fama_style.rhetoric.style_tradeoff --dry-run \
  --output-dir /tmp/style_tradeoff_synthetic
python3 -m unittest discover -s tests -v
```

With the frozen private corpus available, prepare a pack in a fresh directory using `--prepare-pack`. `scripts/build_style_tradeoff_notebook.py` exports the prepared pack at the path documented above. Local checks require neither model weights nor GPU access. The full 205-test suite and original `train.py --check` passed; notebook compilation and an isolated embedded-runtime preflight also passed. These are execution-contract checks, not evidence of generation quality or sampled GPU numerical stability.
