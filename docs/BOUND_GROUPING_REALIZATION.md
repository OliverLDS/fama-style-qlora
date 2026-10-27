# Source-bound grouping realization

**Prepared and CPU-tested; inference has not run.** This checks three concrete grouping hypotheses from the [training-signal audit](TAXONOMY_TRAINING_SIGNAL_V1_RESULTS.md), rather than fitting a mapper to inconsistent annotation differences. Existing experiments, datasets and results are unchanged.

## Comparison

| Arm | Generator receives | Purpose |
| --- | --- | --- |
| A: concrete only | The existing concrete style instruction and source passage | Plain rewriting anchor |
| B: identity groups | Same prompt plus source sentence units, each in its own group | Evidence-scaffold and source-count control |
| C: oracle groups | Same prompt and source units as B, with the proposed combinations | Test whether those group assignments help realization |

**C versus B is primary.** Only the group assignments change between those two arms. A versus B also changes the evidence scaffold, so it cannot isolate the effect of a grouping plan alone. B/C share the same request to aim for one output sentence per group, in order; no decoding constraint forces the requested count.

There are **27 generations**: three selected training pairs × three arms × three seeds. Seeds 41/42/43 and arm order are rotated across cases and seeds. The common style instruction and allowance for minor drift are identical. No claims ledger, repair loop, automatic semantic acceptance filter or best-of-three selection is added.

| Pair | Source count | Identity-plan count | Oracle-plan count | Source sentence groups in C |
| --- | ---: | ---: | ---: | --- |
| Managerial human capital | 6 | 6 | 4 | `[1,2] [3] [4] [5,6]` |
| Dividend initiation | 7 | 7 | 5 | `[1] [2] [3,4] [5] [6,7]` |
| Premium-distribution horizons | 4 | 4 | 2 | `[1] [2,3,4]` |

Every source sentence appears once in order. The groupings were proposed after reading references and are **oracle diagnostics**, not deployment predictions, independently verified atomic-claim alignments or a learned three-level taxonomy. They cannot license a new causal relationship. Reference wording and evaluation-only obligation notes never enter generator prompts. B/C both duplicate the source as units to match their scaffold.

These are previously inspected **training examples**, selected for clear structural contrast. Their references also contribute to the existing train-only style profile. This measures a narrow realization mechanism, not held-out performance or generalization. Development-validation and reserved test files are not read by preparation.

## Execution controls

Keep the established pinned base `google/gemma-3-4b-it`, revision `093f9f388b31de276ce2de164bdc2081324b9767`, without an adapter. Use one visible T4, NF4 with FP32 computation, temperature 0.6, top-p 0.9, top-k disabled, 768 new tokens and a 4096-token context limit. Tokenize complete prompts without truncation; empty/cutoff results leave the run incomplete with partial artifacts preserved. Original Gemma 3 1B training workflows are unchanged.

Gemma unloads before the pinned semantic verifier loads. Full paragraph verdicts are saved incrementally. If requested and observed counts agree and boundaries are unambiguous, each output sentence is additionally checked against the concatenated source sentences in its assigned group. Full positional verdicts are bound to source-group/output hashes and saved separately and in metrics.

An experiment-only boundary check handles parenthesized sentences while retaining the historical splitter unchanged. Disagreement between the two segmentations or an abbreviation ambiguity leaves positional alignment unresolved. Count mismatches also leave positional alignment unresolved rather than forcing a guessed correspondence. All generated paragraphs remain available for assessment.

## Evaluation and decision

Report all 27 candidates: format, source copying, surface style features, requested count, boundary uncertainty, whole-paragraph warnings, positional group warnings, graded-objective proxies, token use, time and memory. **Count agreement alone is not grouping adherence**, and group entailment is not semantic proof. A source group can contain several atomic claims whose bindings remain imperfectly assessed by NLI.

After execution, assess all candidates against their sources and references using the existing separate style/drift rubric. Minor drift remains eligible; no NLI warning automatically vetoes an output. Compare C with B per pair and across all three seeds, keeping A as an anchor. A useful result requires reference-grounded style improvement as well as meaningful realization of the requested grouping under the graded objective. Higher count adherence or longer sentences alone are insufficient. Three selected examples cannot authorize broad mapping/deployment claims.

No new semantic-accuracy or author-style result is claimed before execution. No mapper or additional QLoRA training is authorized by preparing this notebook.

## Manual Kaggle workflow

1. Import `kaggle/bound_grouping.ipynb` into a **new private notebook**, for example `bound_grouping`.
2. Create a private Dataset containing **only** `research_runs/bound_grouping_realization_v1/input_pack/bound_grouping_pack.json`, and attach it.
3. Enable internet, **GPU T4 x2**, and the existing `HF_TOKEN` Kaggle Secret. The first cell exposes GPU 0 only. Credentials are retrieved at runtime and are absent from the notebook and pack.
4. Save & Run All. Preflight checks exact pack and embedded-runtime hashes. Retain `/kaggle/working/bound_grouping_realization_v1/`, particularly `evaluation.json`, `generations.private.jsonl`, `metrics.private.jsonl`, `verifier_results.private.jsonl`, `group_verifier_results.private.jsonl`, `runtime.json`, `token_audit.json`, `run_status.json` and `manifest.json`. Keep all full-text/evidence outputs private.

The builder did not upload or execute anything. The notebook is self-contained, has blank saved outputs and contains no real paragraphs.

## Local reproduction

```sh
PYTHONPATH=src python3 -m fama_style.rhetoric.bound_grouping --dry-run \
  --output-dir /tmp/bound_grouping_synthetic
python3 -m unittest discover -s tests -v
python3 train.py --check
```

With the frozen private corpus, prior pack and audited grouping cards available, use `--prepare-pack` with a fresh output directory. Export the prepared notebook with `scripts/build_bound_grouping_notebook.py`. Implementation is separate in `src/fama_style/rhetoric/bound_grouping.py`.

All **227 CPU tests** passed, including source/reference exclusion, scaffold matching, source coverage, boundary uncertainty, per-group evidence binding, full-verdict replay, incomplete-output rejection and an isolated embedded-runtime dry-run/preflight. These checks do not execute model weights or establish GPU generation quality.
