# Minimal antecedent expansion v1

Status: prepared, not executed. Inference only; no training, adapter or learned mapping model.

The preceding context-restoration run resolved missing economic subjects, but full source context also imported unrelated compensation claims and changed quantity relationships. This experiment asks whether explicitly resolving actors within the assigned group avoids that leakage while retaining useful style change.

## Design

Three previously selected training groups, three matched seeds, three conditions: **27 generation calls**. No paragraph assembly.

| Condition | Generator input |
| --- | --- |
| A local-only | Exact historical local-only prompt. |
| B full-context | Exact historical read-only full-context prompt. |
| C explicit actors | A's identical instruction, with only the assigned source text expanded to resolve actors and cohort. No surrounding passage. |

The selected groups are managerial contract supervision, average dividend-initiation rates, and the 1998 expected-versus-actual rate comparison. C replaces a possessive with managers and makes the never-paid-dividends cohort and dividend-initiation activity explicit. The numeric group receives cohort expansion too; it is not an unchanged-input control.

Expansions are deterministic, case-specific edits derived exclusively from the existing plain-English source. Each replacement records its before/after strings, source support quotes and exact character spans, original/expanded group hashes and annotation provenance. Numbers and existing comparisons are preserved. This is not a generic or learned coreference resolver, an independently verified annotation, a new plain-English dataset, or an extraction of target wording.

A/C differ only in assigned text. B retains its prior context-use and no-derived-quantity instructions, so B/C is a strategy comparison rather than a context-only causal contrast. All three share original group token caps, deterministic group sampling seeds and decoding; expanded inputs do not receive extra output tokens. Triple order rotates to balance positions. All 18 A/B output hashes are available for reproduction checks before interpreting C.

Pinned base Gemma 3 4B, NF4/FP32, temperature 0.6, top-p 0.9, top-k 0, 4096-token context ceiling, no prompt truncation. One visible T4. No retries, repairs, fallback or output from previous calls. Empty/cutoff samples remain failures; malformed outputs remain verbatim. Costs include prompt/output tokens and generation time; loading/NLI overhead remains even for this small run.

The private pack retains the three historical cases for frozen prompt constructors; only three groups from two cases are generated. Neither development validation nor the 45 reserved test rows is read. Existing experimental code and artifacts remain unchanged.

## Assessment and decision

Save local-group and expanded-group full verifier evidence for every available output, plus full-source-to-output NLI. Both arms and C use the same expanded-group premise for this additional comparison. Bind each verdict to source, expanded group and output hashes. These are imperfect diagnostics, not calibrated correctness grades or binary gates. The previous run demonstrated that high NLI entailment can coexist with obvious subject/relationship errors.

Separately assess:

- Whether actor, historical cohort, expected/actual roles, quantities and comparisons survive.
- Whether compensation, enrollment, a new rate gap or another unrelated mechanism is introduced.
- Whether style improves beyond generic formalization, and whether the result simply copies the expanded input.
- Single-sentence completion and measured cost.

Keep the established allowance for minor drift when substantial style improvement exists. Accurate actors alone are insufficient evidence for proceeding to training. If C still invents mechanisms or alters quantity relationships without substantial style benefit, report that realization bottleneck rather than claiming a successful structural mapper. Positive results on these selected groups would require broader validation, not establish generalization.

## Manual Kaggle run

1. Upload `kaggle/minimal_antecedent.ipynb` as a **new private notebook**.
2. Attach a **private Dataset containing only** `research_runs/minimal_antecedent_v1/input_pack/antecedent_pack.json`.
3. Enable internet, GPU T4 x2 and the existing `HF_TOKEN` Secret. The first cell exposes only GPU 0. No credentials should be included in the Dataset.
4. Save & Run All; share its notebook slug and version when finished.

The pack contains copyrighted source/reference text for evaluation and must remain private. Reference text is never sent to Gemma. The notebook embeds source code and verifier fixtures, without private paragraph data or credentials.

Download `/kaggle/working/minimal_antecedent/`: evaluation, raw generations, metrics, full verifier results, jobs, token audit, runtime, status and artifact manifest. Prepared is not executed; no results are claimed here.

Implementation: `src/fama_style/rhetoric/minimal_antecedent.py`; builder: `scripts/build_minimal_antecedent_notebook.py`; tests: `tests/test_minimal_antecedent.py`. CPU tests cover exact controls, matched budgets/seeds, target exclusion, context exclusion in C, anchored edits and numeric invariance, failed samples, verdict hashes, and notebook preflight without a checkout, Secret, ML dependencies or GPU.
