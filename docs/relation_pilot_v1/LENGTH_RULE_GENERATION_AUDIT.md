# Additional-paper generation audited; verification ready

Current status: [all verification responses audited; policy results available](LENGTH_RULE_RESULTS.md).
The handoff below records the original generation-audit stage.

All eight response files cover **32/32 pairs**, with one candidate each and no
abstentions. Every record passes the frozen generation schema, fixed-input
checks and exact context-evidence checks. Source paragraphs and pair text are
unchanged; no candidate or raw provenance was repaired.

The user reports **GPT-6 with High effort requested**. All response model/effort
fields are null and remain so. The user's report is pinned separately; effective
effort and chat arrangement are not confirmed. This does not affect or overwrite
earlier effort records. Same-model self-review, possible retrieval and previously
used training-pool sources remain limitations.

| Frozen source stratum | Assigned | Generated | Abstained |
|---|---:|---:|---:|
| At most 30 regex words | 16 | 16 | 0 |
| Over 30 regex words | 16 | 16 | 0 |

These are availability counts, **not fidelity, readability or rule-accuracy
results**. All candidates remain semantically unverified. Surface triage records
one over-60-word warning, one numeric-inventory warning and one guard-inventory
warning. Such flags can reflect harmless reference compression or paraphrase;
none is used to reject or alter a candidate. The cutoff stays at 30, with no
classifier fitting or threshold selection.

## Run verification next

Private entry point:
`research_runs/length_rule_replication_v1/verification_package_v1/START_HERE.md`.

There are **eight batches of four**. For each `batches/batch_01` through
`batch_08`, paste `TASK.md` and attach its `items.jsonl` plus
`common/verification_output.schema.json`. Use GPT-6 and request High effort;
report effective effort as unknown when unclear. Save first complete unedited
`responses_01.jsonl` through `responses_08.jsonl` and coverage reports in this
**verification packet's `responses/`**, leaving generation files unchanged.

Every available candidate from both strata is included. The verifier receives
only opaque ID, full neutral source paragraph, the two source sentences and the
fixed candidate. Rule decisions, sampling stratum, earlier labels/claims,
generation context notes, provenance and surface warnings stay offline.
Phase-specific ordering changes batch assignment without changing candidate or
item identity; it does not establish independence. Original author references
are excluded, and the upstream neutral-source/context cautions remain in force.

The verifier extracts fresh V-units, checks fidelity and separately judges
readability, allowing local shift and unresolved outcomes. It must not repair
the candidate. All local judgments stay separate from permission to chain edits
or approval for fine-tuning. After importing those responses, score the locked
KEEP, MERGE-all-available and fixed-30 policies under the
[replication analysis plan](LENGTH_RULE_REPLICATION.md), with ties/drift reported
separately and no refitting. Unlike the earlier 52-item check, this sample is
not restricted to previously MERGE-preferred candidates.

## Artifacts and checks

Private raw snapshot, user settings, per-item checks and warning categories:
`research_runs/length_rule_replication_v1/generation_audit_v1/`.
Public [generation counts](length_generation_audit.json) and
[verification preparation](length_verification_preparation.json) contain no text.
Both frozen packet and audit manifests are verified before verification inputs
are projected. Invalid generated records would block forwarding; abstentions
remain unavailable rather than receiving invented candidates.

CPU entry point: `fama_style.research.length_rule_generation_review`, using
`--packet`, `--audit-output`, `--session` and a fresh `--verification-output`.
Tests cover raw preservation, null provenance, batch/ID coverage, fixed-text
corruption, unavailable candidates, both-stratum inclusion, context/hint exclusion,
hash tampering and overwrite refusal. Existing sources, experiments, labels and
hashes stay unchanged. No reserved examples, new model calls, classifier fits,
GPU jobs or neural/adapter training were used by this audit and preparation.
