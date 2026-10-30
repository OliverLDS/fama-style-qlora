# Semantic binding repair v1

Completed a versioned repair of the two remaining scope issues in the existing
commentators/contagion example. Seven fields in one brief changed. The original
paragraph, annotations, input packets and output evaluations are unchanged.

The first correction restores lack of confidence rather than a comparative
reduction in confidence. It updates the proposition, causal outcome and explicit
negation together. The second restores the categorical barrier to authoritative
media representation rather than a gradual limitation. Its proposition, causal
field, negation and relationship qualification now express the same scope.
The restriction concerns authoritative representation, not all media discussion.
No claim IDs, actors, quantities or relationship endpoints changed.

Each repair records a before value, after value, exact supporting passage from
the reviewed reference and rationale. Source and brief hashes bind the decision;
changed inputs or missing evidence fail preparation. The generated handoff keeps
opaque claim IDs and validates schema and relationship closure. Its schema remains
`economic_content_brief_v1`; this repair has a separate revision identity,
`neutral_reconstruction_binding_repair_v1`.

Private files are under
`research_runs/neutral_reconstruction_v1/semantic_binding_repair_v1/`:

- `repair_decisions.private.json`: source-anchored decisions.
- `results/repair_audit.private.jsonl`: decisions and derived result with reference.
- `results/derived_briefs.private.jsonl`: revised brief and generator-safe handoff.
- `results/receipt.json` and `results/manifest.json`: provenance and frozen hashes.

The reviewed reference is the evidence basis; no new PDF verification or
independent semantic validation was performed. The earlier outputs retain their
original assessments. No corrected outputs have been generated, and this derived
brief is not certified training-ready. No further annotation or training starts
as part of this repair. [Aggregate summary](semantic_binding_repair.json).

## What the GPT-6 tests establish

GPT-6 has two distinct roles in this track. It prepares candidate annotations and
neutral inputs, and it serves as a development writer for testing the brief-to-text
protocol. The instruction experiment helps diagnose whether a representation or
instruction produces added commentary even with this writer. It does not test
Gemma's ability to follow the same representation. The automated Codex review is
also imperfect and is not independent human or semantic gold verification.

These roles must remain separate in reporting:

| Stage | What it checks | What it does not establish |
| --- | --- | --- |
| GPT-6 brief and neutral-input preparation | Candidate training signal, semantic bindings and input/reference contrast | Gemma generation quality |
| GPT-6 realization check | Whether this writer can use the brief without extra commentary | Transfer of that behavior to Gemma 4B |
| Gemma 4B inference-only check | Whether the intended model can use the same prompt and brief | A benefit from post-training |
| Later controlled Gemma post-training comparison | Whether training improves the intended model against matched baselines | Distinctive Fama style without target-specific evidence |

A teacher model can supply candidate text pairs without the student reproducing
its full multi-step process. If the intended training objective is neutral text to
human reference, the input brief is a data-construction tool, not necessarily a
student inference requirement. If the intended objective is plan-conditioned
rewriting, the student must explicitly be tested and eventually trained with that
same input format. These are separate experimental conditions, not interchangeable
claims of success.

The next model-facing gate should therefore be a small Gemma 4B inference-only
check on already exposed training-side cases, holding the brief, instruction and
decoding controls fixed where available and comparing ordinary versus brief-guided
realization. References stay outside generation inputs. Judge claims, commentary,
readability and copying separately; do not treat success on GPT-6 as the baseline
result for Gemma. No Kaggle packet or job has been prepared here. A fresh sample
is needed later to assess generalization beyond repaired development failures.
