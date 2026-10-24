# Scoped endpoint and causality guards: calibration v3

Status: implemented, frozen, CPU-tested and replayed against existing saved
probabilities. **No fresh pretrained-model evaluation has run.** Version 2's
code, notebook, cases, results and thresholds remain unchanged.

## Run the fresh calibration

1. Upload `kaggle/verifier_calibration_v3.ipynb` into a **new private notebook**.
2. Enable internet and **GPU T4 x2**. Only GPU 0 is exposed.
3. No dataset attachment, Gemma access or `HF_TOKEN` Secret is required.
4. Save & Run All, then share the notebook slug and version.

This runs the same pinned public NLI model. There is no Gemma generation,
fine-tuning, optimizer or mapper training. Nothing has been uploaded or submitted
by this implementation. The 21-example rewriting comparison remains blocked.

Main output: `/kaggle/working/verifier_calibration_v3/evaluation.json`.
The run also saves full evidence, runtime metadata, completion status and hashes.

## Focused changes

`src/fama_style/research/relation_guards.py` adds two finite, evidence-linked checks:

- **Numeric bound relation:** compare operators only when the literal variable,
  canonical numeric value and unit match uniquely. Strict versus inclusive
  bounds remain distinct. Supported decimal/spelled numbers and percent formats
  normalize without deleting the operator. Different variables, units, ambiguous
  repeated bounds or unsupported detected forms do not become forced matches.
  No integer-domain equivalence is assumed.
- **Association strengthened to causation:** match a literal unordered entity pair
  for association, then check whether the output asserts a directed causal
  relationship not already supplied by the source. A causal statement about an
  unrelated pair cannot authorize that relationship. Source causality counts only
  for the matching direction. Conditional, negated, modal and attributed relation
  statements outside the finite grammar do not authorize unconditional causality.

`claim_verifier_v3.py` wraps the unchanged v2 verifier. A new scoped conflict
produces `risk_signal`; a detected but unsupported/unmatched scope turns an
otherwise consistent result into `abstain`. Existing NLI risk signals are never
erased. Identity text bypasses extra guard ambiguity, but copying remains a
separate concern. Every verdict retains `semantic_proof: false`.

The guards are not a general arithmetic, causal or logical parser. They compare
supported complete clauses and literal entity names, without resolving arbitrary
paraphrases, scientific notation, coreference or nested scopes. Numeric condition
evidence is retained; matching a bound does not establish equivalence of the full
condition. Expressions not recognized by the detectors remain subject to the
fallible NLI checks. `model_consistent` still does not certify meaning.

## Executed development replay

The 120 exposed v2 cases are now development-only. Replaying their saved NLI
probabilities with the final v3 guards required no weights, inference or network:

| Measurement on exposed development cases | v2 | v3 |
|---|---:|---:|
| Changed cases detected | 38/40 | 40/40 |
| Changed cases falsely accepted | 2/40 | 0/40 |
| Faithful cases falsely flagged | 2/80 | 2/80 |
| Nontrivial faithful paraphrases accepted | 38/40 | 38/40 |

Both known misses are now flagged. These are post-hoc regression results, **not
held-out improvement or a passing readiness result**. No threshold was raised
to conceal the association error or lowered to accept the quantifier paraphrase.
The original NLI probabilities and all previous output bytes remain intact.

Replay artifacts are private under
`research_runs/claim_verifier_calibration_v3/cached_development_replay_final/`.
The `replay_v2` helper reconstructs the exact cached requests, checks that the v2
baseline verdict did not change, and relabels every exposed row as development.

## Fresh evaluation and decision

The final code hashes were frozen before authoring the final fresh v3 scenarios.
There are 120 development cases and 48 fresh held-out cases, with sixteen
identities, sixteen nontrivial faithful paraphrases and sixteen changed outputs.
Eight scenario groups focus on endpoint/scoping/causality behavior; eight retain
broader actor, quantity, condition, modality, quantifier, negation, exception and
attribution checks. This is a targeted challenge built with knowledge of the
grammar, not a blinded or independently annotated economics-domain benchmark.

An earlier unexecuted candidate set and draft artifacts are archived locally.
They were replaced after a scope check identified a conditional-scope refusal gap,
now covered by a regression test;
the final controls were authored after that fix and final freeze. No fresh NLI
results were used for rule tuning. Final protocol and cases are in
`docs/claim_verifier_calibration_v3/`.

The gate is unchanged: at least 80% error detection, at most 10% faithful false
alarms, at least 50% acceptance of nontrivial faithful paraphrases, and zero
changed outputs accepted on this small set. Abstentions remain in denominators.
The report includes both v2 and global/sentence NLI baselines on exactly the same
cases, plus counts of v2 acceptances overridden or converted to abstentions.

A fresh pass would warrant evaluating the approach on development economics
paragraphs with explicit uncertainty. It would not establish general semantic
accuracy, authorize training or automatically execute the 42 prepared rewriting
prompts. A failure must be reported without post-hoc threshold relaxation.

## Local verification

All 174 CPU tests passed. New tests cover strict/inclusive operators, numeric and
unit normalization, unrelated variables, ambiguous scopes, association symmetry,
causal direction, unrelated causal evidence, conditional causal permission,
negation/modality/attribution refusal, identity, cached replay, development/test
isolation and standalone notebook preflight. Fake predictors test mechanics only.

```sh
python3 scripts/build_verifier_calibration_v3_notebook.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m fama_style.research.claim_verifier_v3 \
  --output-dir /private/tmp/verifier-v3-new-preflight
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

The notebook is self-contained with code and synthetic cases only. Public-model
downloads use `token=False`; no Secret is read or printed. Model weights remain
off the Mac. FP32 NLI, context limits, pair budgets and model revision remain
unchanged from v2; runtime and memory will be recorded by the actual run.
