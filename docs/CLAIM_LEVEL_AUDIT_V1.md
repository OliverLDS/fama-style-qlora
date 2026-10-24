# Claim-level audit of saved generations

Implemented and executed a CPU-only audit of the sixteen outputs from
`oliverlds/claim_constrained_realization`, version 1. No new text generation,
model download, API request, GPU job, training or user review was required.
Existing experiment files and evaluation scores remain unchanged.

## What changed

`src/fama_style/research/claim_audit.py` adds evidence-linked, finite checks:

- Compare explicit upper/lower-bound cues in overlapping subject contexts. A
  correct statement elsewhere does not cancel a conflicting bound.
- Track quantified subject/property pairs, rather than considering a matching
  word anywhere in the paragraph sufficient. Firm size and market-value share
  remain separate obligations.
- Track annual qualifications within a related source/output context.
- Flag possible strengthening of tentative language to categorical language.
- Compare numeric atoms after normalizing decimal formatting, supported number
  words and explicit abbreviated year ranges. Equal numbers do not certify their
  roles, units or relationships.
- Surface newly mentioned concepts from a small declared catalog, including
  predictive validity and systematic bias. Novel terminology is a candidate
  addition, not proof that an unsupported proposition was added.

Each record retains source-unit IDs, source/output evidence, scoped obligations,
hashes and the already-saved sentence-unit NLI probabilities. No reference text
is used to construct the checks. Source units are sentences, not guaranteed atomic
claims. The detectors do not resolve actors, coreference, negation scope,
quantifier scope or general causality. Modality matching is a limited surface
warning, not a complete modality parser.

## Results

All sixteen saved outputs are included, including the one with an invalid slot
count. The additional audit never changes the original format or NLI scores.

| Condition | Outputs with risk signals | Other outputs, unresolved | NLI-passing outputs with risk signals |
|---|---:|---:|---:|
| A: count paragraph | 2/4 | 2/4 | 0 |
| B: claims + paragraph | 2/4 | 2/4 | 2 |
| C: count slots | 2/4 | 2/4 | 2 |
| D: claims + slots | 2/4 | 2/4 | 0 |

The four NLI-passing rows with risk signals include C's invalid-count row.
These counts are diagnostic coverage, not error rates, precision/recall or a
ranking of generation methods. There is no validated semantic-accuracy score.

The checks surface the previously identified counterexamples:

- D/F1965: opposite-direction bounds appear in overlapping contexts, even though
  the output also includes a correctly directed bound elsewhere.
- D/NL2003: the quantified size claim and annual qualification remain unresolved;
  a small market-value share cannot discharge the separate firm-size obligation.
- B/F1965: newly introduced predictive-validity language is flagged despite the
  saved NLI pass. C/F1965 raises the same concept warning.

The rules also expose their limits. B/NL2003 paraphrases the majority-size claim
using a construction outside the finite copular grammar; it receives an unresolved
warning rather than a confirmed-omission judgment. Other concept warnings may
also be valid paraphrases. The audit does not silently treat lexical mismatch as
a semantic error or treat matching numbers as faithful role assignment.

## Gate semantics

`risk_flagged` means one or more scoped diagnostic signals exist. `unresolved`
means the finite rules did not establish equivalence; absence of events is not a
pass. `surface_replay_only` is reserved for identity or the existing bounded,
replayable transformation grammar. Even that status does not claim general
semantic proof. Every record sets `semantic_proof` to false.

All sixteen generated outputs remain either risk-flagged or unresolved. This
means **none is automatically certified**, not that all sixteen are wrong.
The checker deliberately abstains on unsupported paraphrases. Rules were developed
after inspecting these four cases, so their performance here is a regression
check, not independent validation.

## Verification and reproduction

The audit requires the prior successful verification, checks every recorded
artifact hash and the frozen input-pack hash, confirms the run completed without
training, and replays prompt identity and output normalization. It checks original
artifact hashes again before completion. Fresh output directories are required.

Final private outputs:

- `research_runs/claim_level_audit_v1/audit_v2/claims.private.jsonl`
- `research_runs/claim_level_audit_v1/audit_v2/summary.json`
- `research_runs/claim_level_audit_v1/audit_v2/manifest.json`

The independently repeated CPU audit in `reproducibility_v2/` produces identical
evidence records and summary bytes. Earlier local audit snapshots remain intact.
Private evidence includes copyrighted source/generated paragraphs and must not
be published. This document contains aggregate results only.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m fama_style.research.claim_audit \
  --pack research_runs/concise_plan_v1/input_pack/concise_plan_pack.json \
  --results research_runs/claim_constrained_v1/kaggle_version_1/claim_constrained_realization_v1 \
  --verification research_runs/claim_constrained_v1/version_1_review/verification.json \
  --output-dir research_runs/claim_level_audit_v1/new_reproduction
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

The full CPU suite passed 151 tests. Eight new tests cover bound reversal and
internal inconsistency, unrelated-subject negative controls, property/frequency
scope, number/year normalization, changed quantities, role-swap abstention,
modality, concept warnings, NLI-pass counterexamples, identity, reproducibility,
tamper rejection and refusal to overwrite existing audits. Tests use synthetic
text; the public suite does not require private datasets or model weights.

## Next decision

Do not launch another training run or declare the current semantic gate reliable.
The audit makes specific failure modes visible but cannot establish general
meaning preservation with lexical rules and cached paragraph/sentence NLI.

Before another generation comparison, validate a claim-verification specification
with independent meaning-preserving and meaning-changing controls, including
negation, bound endpoints, actor/quantity assignment, implicit paraphrases and
added assertions. Preserve explicit abstention. A future model-based verifier
would need its own reliability check; the sixteen inspected outputs cannot
provide that validation alone.

If generation is resumed, compare semantic-first rewriting with soft structural
guidance rather than exact sentence quotas, and judge style only alongside
faithfulness and copying. Neither that experiment nor a new verifier model was
executed here.
