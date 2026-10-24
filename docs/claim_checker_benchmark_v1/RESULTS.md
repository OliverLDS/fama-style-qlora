# Frozen claim-checker benchmark

Executed locally without model downloads, inference, training, user review or
GPU jobs. The existing checker was frozen before evaluation and was not changed
after seeing the benchmark results. It **failed the readiness gate**.

## Design

There are 72 authored synthetic cases: 36 development and 36 held-out cases.
Each split has twelve scenario groups, one for each error category. Each group
contains an identity, a faithful nontrivial paraphrase and a changed-meaning
output. Thus each split contains 24 faithful and 12 changed examples. Scenario
IDs, source wording, actors and values differ across splits; variants of the
same source never cross the split boundary.

The categories cover reversed bounds, actor swaps, quantity assignments,
frequency qualifiers, conditions, modality, quantifiers, added assertions,
negation, causal direction, numeric values and exceptions. All text is synthetic;
no paper paragraphs or reserved corpus test examples are used.

These are author-written controls, not a blinded benchmark or independently
annotated gold standard. The author knew the checker design. Held-out cases use
different scenarios but the same twelve categories and often similar logical
constructions. They measure narrow diagnostic behavior, not domain generalization.
There was no rule fitting on either split.

## Frozen protocol and metrics

[frozen_protocol.json](frozen_protocol.json) fixes the checker source hashes,
fixture hashes, prediction mapping and readiness thresholds. Fixtures are in
[development.json](development.json) and [held_out.json](held_out.json).

A rule event is a risk signal; a clean bounded surface replay is a surface
acceptance; everything else abstains. A surface acceptance remains weaker than
general semantic certification. Abstentions remain in every applicable
denominator. Identity and nontrivial paraphrases are reported separately to
prevent an identity-only checker from appearing useful for rewriting.

The operational gate requires at least 80% changed-output detection, at most
10% faithful false alarms, at least 50% faithful surface acceptance overall and
at least 50% acceptance on nontrivial faithful paraphrases. These thresholds were
declared before evaluation; they are planning criteria, not statistically
validated deployment requirements. Reject-everything and abstain-everything
controls both fail the gate.

## Executed results

Development and held-out aggregate outcomes are identical:

| Measurement, per split | Result |
|---|---:|
| Changed outputs producing risk signals | 5/12 (41.7%) |
| Faithful outputs producing false alarms | 2/24 (8.3%) |
| Faithful surface acceptances | 12/24 (50.0%) |
| Nontrivial faithful paraphrases accepted | 0/12 |
| Nontrivial faithful paraphrases receiving false alarms | 2/12 |
| Changed outputs falsely surface-accepted | 0/12 |
| Changed outputs receiving abstentions | 7/12 |
| All outputs receiving abstentions | 17/36 (47.2%) |

The accepted faithful cases are all identities. Zero false acceptances does not
make this a useful rewrite verifier: most changed cases abstain, and no
nontrivial faithful paraphrase is accepted. The two false alarms involve valid
frequency and quantified-property paraphrases.

| Held-out mutation category | Outcome |
|---|---|
| Bound reversal | Risk signal |
| Actor swap | Abstention |
| Quantity assignment swap | Abstention |
| Missing annual qualification | Risk signal |
| Missing condition | Abstention |
| Strengthened certainty | Risk signal |
| Changed quantifier | Risk signal |
| Added assertion | Abstention |
| Removed negation | Abstention |
| Reversed causality | Abstention |
| Changed numeric value | Risk signal |
| Lost exception | Abstention |

Full aggregate metrics are in [results.json](results.json). These evaluate the
CPU rules alone, not the independent NLI model or a fused rule/model verifier.
The previously saved NLI outputs do not cover these new cases; no NLI performance
on this benchmark is claimed.

## Decision and prepared comparison

Do not use the current checker to certify faithfulness or select new training
data. Its warnings remain useful diagnostics, but the benchmark gate failed on
error detection and nontrivial paraphrase acceptance. The next verifier must
address role/quantity assignments, conditions, exceptions, causality and free-form
added assertions, while recognizing valid paraphrases. More surface cue rules
alone are unlikely to resolve that broad task.

The 21-example comparison is prepared as 42 frozen private prompts, but **remains
blocked and unexecuted**. Both arms use the same source-only ledger and choose
their sentence count freely. B additionally receives an optional source-bound
presentation plan whose only empirical style operation is eligible adjacent
sentence grouping, derived from the frozen 247 training references. Paragraph
order is source-derived; there is no invented lexical-style prescription. This
is a narrow fixed soft prior, not the full LLM taxonomy or a learned mapper.
No reference wording, target sentence count or target schema is transmitted to
the generator.

The input package and prompts are under
`research_runs/soft_guidance_validation_v1/preparation_v1/`; these contain private
text and are not distributed. Its manifest records `execution_ready: false`.
No new Kaggle notebook or GPU runner was created for a comparison whose gate
failed. Passing a future synthetic gate would still not establish reliability on
economics prose or make these already-used development examples a fresh benchmark.

## Reproduce

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m fama_style.research.checker_benchmark \
  --output-dir /private/tmp/claim-checker-fresh-evaluation
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

Private execution records are in
`research_runs/claim_checker_benchmark_v1/evaluation_v1/`. The independently
repeated run in `reproduction_v1/` produces identical result and summary bytes.
Source and fixture hashes are checked before and after execution. Existing audits,
datasets, experimental outputs and checker rules remain unchanged. New tests
cover denominator accounting, degenerate controls, split separation, frozen-file
tampering, reproducibility, reference leakage, matched prompts and blocked
execution. CPU execution requires neither private datasets nor model weights.
