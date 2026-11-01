# Twelve training-only semantic anchor candidates

Prepared **12 anchors from ten training papers**, three each for scope, time,
numbers and reasoning. Every anchor has a new neutral full-paragraph input, the
unchanged original human target, a structured claim inventory and one synthetic
unfaithful target created by one exact contiguous edit. There are **56 claim
bindings** across the 12 cases.

Private package: `data/teacher_pair_semantic_anchors_v1/`.
This is a provisional candidate package, not a revised training release.
Original datasets, prompts, notebooks, model checkpoints and results remain
unchanged. No development or reserved-test paragraphs were used, no external
annotation calls were made and no GPU job or training was started.

## Selection and construction

Selection uses the [46-pair training coverage audit](TRAINING_COVERAGE.md).
All anchors come from its aligned-to-archived-text category; the four cases
with explicit boundary/source-wording limitations are excluded. Seven selected
anchors still retain their original upstream needs-review flag. The new audit
does not erase those flags or independently certify PDF/OCR extraction.

| Family | Cases | Counterexample operation |
| --- | ---: | --- |
| Scope | 3 | Reverse near-universal scope; strengthen possibility to impossibility; expand selected coverage to all |
| Time | 3 | Reverse event lead order; change a reporting start into an end; substitute past for future realizations |
| Numbers | 3 | Swap percentages between economic roles; swap study-to-horizon bindings; change a reported magnitude |
| Reasoning | 3 | Substitute an unsupported explanation; replace a limiting contrast with a causal link; change the stated explanatory property |

Faithful variants retain the complete supplied claims, names/citations, economic
actors, conditions, negation, uncertainty, numerical roles, periods and causal
direction. Wording and sentence grouping may change; one numerical input also
changes presentation order while preserving the value-to-role mappings.
They are neutral inputs, not requests for stereotypically “AI-like” prose.

The preference candidates use the **same input prompt for both targets**. The
chosen target is the exact archived human paragraph; the rejected target is
reconstructed from it using the recorded before/after span. A swap is one
relational operation even when it moves two value-role bindings. Errors labeled
unsupported are unlicensed by the input; they need not be factually false in
every possible economic model. Rejections are deliberately synthetic, not actual
Gemma mistakes or additional source paragraphs.

## What the checks establish

Each claim has an identifier, statement, actors, relation and explicit lists for
quantities, time, conditions, negation, modality and causality. Source, target
and variant evidence spans are checked against the exact corresponding text.
Affected-claim IDs bind each negative operation to its semantic focus.

The review records preserve text hashes, a hash of the complete claim ledger
and a hash of the negative-edit contract. Every claim has a second-pass check.
Changed wording, altered ledgers, missing claims, unsupported family labels or
ambiguous replacement spans invalidate the integrity checks.

Construction and the second pass were done by the **same Codex reviewer**.
This is not independent semantic verification, a calibrated benchmark or a
formal proof that the variants preserve meaning. Literal evidence spans,
protected-name checks and matching numerical sets are necessary screens;
matching them does not establish entailment. The ledger is an interpreted
annotation, not an author-verified semantic graph.

The second pass clarified two synthetic inputs: an ambiguous proceeds pronoun
was replaced with explicit aggregate IPO proceeds, and effective sample size
was made explicit. These changes, draft hashes and the initial synthetic draft
are retained. No human target or original teacher input was repaired.

## Observed contrast

| Descriptive measurement | Result |
| --- | ---: |
| Faithful input variants | 12 |
| Single-edit unfaithful targets | 12 |
| Median original input/target word-sequence overlap | 0.766 |
| Median variant/target word-sequence overlap | 0.459 |
| Minimum rejected/chosen word-sequence overlap | 0.921 |
| Rejected/chosen word-length ratio range | 0.989–1.023 |
| Negatives preserving the entire numeric-value multiset | 11/12 |

The new inputs provide more lexical contrast against the fixed targets. That is
not evidence of better style, easier learning or faithful student generations.
The near-identical chosen/rejected texts reduce broad length/style differences
as explanations for a later preference. They do not remove all shortcuts:
scope words, local inconsistencies and synthetic edit patterns could be easy
clues. In particular, this small constructed collection cannot validate a
general semantic verifier.

Both numerical role-swap examples keep all numbers, units and citation years
while changing their bindings. A numeric-presence screen alone would miss
them. The magnitude-change example instead changes the numeric inventory.
This illustrates why set matching is useful but insufficient.

## Files and execution boundary

- `anchors.private.jsonl`: full evidence, claims, variants, edits, judgments and
  retained original curation notes.
- `sft_candidates.private.jsonl`: 12 faithful variant → unchanged-target pairs
  under the existing ordinary rewrite instruction.
- `preference_candidates.private.jsonl`: 12 same-prompt chosen/rejected pairs.
- `construction.private.jsonl`, `review.private.jsonl`, provenance, revision
  notes and initial draft: construction and review history.
- `summary.json`, `receipt.json`, `manifest.json`: aggregates and frozen hashes.

Both candidate formats have `training_ready=false`; all anchors also have
`semantic_gold_verified=false`. These files have no training configuration or
Kaggle notebook and are not merged with the 46 original pairs. Model-tokenizer
length/label preflight has not run. They support comparing stronger faithful
SFT inputs with semantic contrasts without choosing an optimizer prematurely.

The next experimental decision should follow semantic validation of the new
candidates and a bounded inference-only check on Gemma 4B. It must distinguish
ability to rewrite these training-side cases from generalization, and must not
treat successful ranking of synthetic negatives as a new style benchmark.
Additional GPU inference or training requires its own approved scope. No
full-training, distinctive Fama-style, human-authorship or detector claim follows.

Public aggregate results: [semantic_anchors.json](semantic_anchors.json).
Construction inputs and earlier drafts are also retained privately under
`research_runs/teacher_pair_reverse_pilot_v2/semantic_anchors_v1/`.
Copyrighted target paragraphs, variants and counterexamples remain private.

Reproduce into a fresh directory:

```sh
PYTHONPATH=src python3 -m fama_style.research.teacher_pair_anchors \
  --coverage research_runs/teacher_pair_reverse_pilot_v2/training_coverage_v1/results \
  --drafts research_runs/teacher_pair_reverse_pilot_v2/semantic_anchors_v1/construction.private.jsonl \
  --reviews research_runs/teacher_pair_reverse_pilot_v2/semantic_anchors_v1/review.private.jsonl \
  --output /tmp/teacher-pair-semantic-anchors-reproduction
```

The builder verifies the frozen coverage packet, training-only membership,
balanced family counts, target and review hashes, positive evidence, numeric
inventory, protected names and exact negative reconstruction. It refuses
overwrite. CPU tests use synthetic records and deliberately unreadable
development bodies; no model, credentials or GPU are needed.
