# Author-name instruction versus concrete name-free style

**Prepared, not executed.** This inference-only follow-up changes the style instruction while retaining the previous four inputs, model, sampling and semantic tolerance. Historical notebooks, packs, results and verifier implementations remain unchanged.

## Matched comparison

- **A — author name:** exactly repeats the direct prompt from style-tradeoff version 1.
- **B — concrete style:** replaces that prompt's author-style sentence with an explicit, name-free specification. No optional plan or claims ledger is appended.
- Four previously exposed development inputs; three candidates per arm, paired seeds 41/42/43: **24 generations**. Reuse all four inputs without selecting on their latest outcomes. The 45 reserved test examples are not read.
- The common semantic tolerance and output instructions are byte-identical between arms. Small changes in emphasis or secondary wording remain acceptable. There is no hard sentence quota, automatic repair, NLI acceptance gate or best-of-three success claim.
- Same pinned base Gemma 3 4B revision, NF4/FP32, one visible T4, temperature 0.6, top-p 0.9, top-k disabled, 768 new-token limit and 4096 context limit as the previous run. All prompts are checked without truncation. No adapter or training is involved.

The named arm retains the historical prompt and output hashes as controls. Exact output matches are reported, not assumed or required. Hardware and sampling reproducibility can affect historical comparisons; A versus B within the new run is the primary comparison.

## Concrete specification

> Rewrite the passage as direct, analytical economics prose. Open with the substantive claim or issue rather than a broad statement about the literature. Use existing economic actors and variables as subjects and direct verbs for methods and findings. Make the existing comparisons, conditions, exceptions and inferences easy to follow, without imposing a fixed argument order. Keep necessary technical terms and place qualifications close to the claims they qualify. Use economical wording rather than inflated nominal phrases. End with the last substantive claim rather than a generic statement about complexity or a new research agenda. No fixed sentence length or sentence pattern is required.

This is followed in **both arms** by:

> Keep the economic argument and main claims intact. Small changes in emphasis or secondary wording are acceptable when they improve the writing. Do not invent findings, numbers, actors or causal mechanisms. Sentence count is free; longer sentences and inflated vocabulary are not goals. Return only one coherent paragraph without commentary or Markdown.

The specification is a manually designed diagnostic, informed by inspection of seven training-reference paragraphs: one lower-median-length reference per training paper, ordered by reference token count and ID. Their IDs and text hashes are retained privately in the pack. The existing style profile remains fitted on all 247 training references. The design also responds to the observed development failures and is therefore post-hoc; it is not an automatically learned style model or an independent validation result. Reference paragraphs themselves are never supplied to the generator. Author names inside legitimate input citations remain intact; “name-free” describes the style instruction.

**Interpretation:** the comparison changes both author naming and the specificity/length of the style instruction. It tests instruction strategies, not the causal effect of removing the author's name alone. A named-plus-concrete condition would be needed to isolate that narrower effect.

## Evaluation and evidence retention

Retain all 24 candidates, exact/near-copy diagnostics, paragraph format, style features, warning pressures, sensitivity-weight utilities and per-case frontiers. Qualitative style and drift judgments remain pending until results are assessed against the existing rubric; numerical proxy gains alone do not establish success.

The pinned semantic verifier loads after Gemma unloads. This version saves **full verdicts incrementally** in `verifier_results.private.jsonl`, bound to source/output hashes, and includes them in `metrics.private.jsonl`. This permits CPU metric replay from saved verdicts. Repeating the actual NLI inference would still require executing its pinned model. Runtime, token audits, historical-control matches, status and output hashes are retained. Partial failures remain incomplete; exceptions are reported by type without credential values or tracebacks.

## Manual Kaggle workflow

1. Upload `kaggle/name_free_style.ipynb` into a **new private notebook**, for example `name_free_style`.
2. Attach a private Kaggle Dataset containing **only** `research_runs/name_free_style_ablation_v1/input_pack/name_free_style_pack.json`.
3. Enable internet, **GPU T4 x2**, and the existing `HF_TOKEN` Kaggle Secret. Only GPU 0 is exposed by code. The token is retrieved at runtime; it is absent from the pack and notebook.
4. Save & Run All. The preflight checks the fixed pack and embedded code hashes before model access. Download the output directory `/kaggle/working/name_free_style_ablation_v1/` and keep its full text and verdict files private.

No Kaggle upload, run or GPU training was submitted by preparation. The notebook is self-contained and has no saved outputs or source paragraphs.

## CPU reproduction

```sh
PYTHONPATH=src python3 -m fama_style.rhetoric.name_free_ablation --dry-run \
  --output-dir /tmp/name_free_style_synthetic
python3 -m unittest discover -s tests -v
python3 train.py --check
```

Preparation uses `--prepare-pack` with a fresh output directory and requires the frozen private training corpus and verified previous run. The builder is `scripts/build_name_free_style_notebook.py`. The local suite passed **212 tests**, including prompt matching, unchanged decoding, reference exclusion, full-verdict retention/replay, context limits and the isolated embedded-notebook preflight. The original training dry-run passed. These CPU checks are not generation-quality results; model weights were not accessed.
