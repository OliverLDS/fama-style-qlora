# Repaired-source bindings v1

Four training-only source repairs now have fresh claim/evidence bindings and
recompiled sentence-packing plans. This is a CPU readiness checkpoint. No new
output quality, semantic-equivalence, or distinctive Fama/coauthor-style result
is established.

| Paper | Claims | Planned sentences | Diagnostic scope |
|---|---:|---:|---|
| NL2003 | 3 | 2 | Context-bound profitability comparison; unspecified decline comparator retained |
| FJ1983 | 4 | 3 | Bounded source-fidelity diagnostic |
| ME1997 | 8 | 8 | Remaining recognition-versus-correction antecedent difference |
| F1980 | 6 | 4 | Bounded source-fidelity diagnostic |

All four merge/split/keep plans were compiled again from the revised inputs.
Twenty-one whole-claim anchors, 27 finer source facets and four within-claim
constraints pass exact-span checks. The two bounded cases have no recorded
remaining alignment caveat in this review; that is not proof of completeness or
independent semantic verification.

## Changes to the evidence interpretation

The ME1997 legacy annotation contained a conditional edge whose evidence belonged
to a different claim. The new release removes that cross-claim edge and represents
the actual conditions within their owning claims. Its attributed elaboration edge
also has corrected endpoints and remains interpretive. Author presumptions retain
attribution and uncertainty rather than being promoted to causal findings.

F1980 preserves the rental asset, its supplier and recipient, the market signal,
and qualified performance dependence. Current-pay uncertainty is distinguished
from the asserted future-wage effect. The supported causal relation uses the
future-wage clause as its endpoint evidence. An information-using institution is
not treated as a quoted speaker.

A stronger check now requires each relation's evidence to be inside its declared
endpoint claim. Mere existence of the quote elsewhere in the paragraph no longer
suffices. Constraint parts must similarly belong to their owning claim. These
mechanical checks do not prove that a semantic relation is correct.

## Remaining limits

NL2003 keeps the original repair's context flag. No missing comparison baseline
is supplied. ME1997's repaired source describes gradual correction while the
unchanged reference describes gradual recognition. This difference remains in
the paired audit only; reference wording is excluded from generation evidence.
A future diagnostic must judge source fidelity separately from reference matching.

Fresh namespaces and source hashes distinguish the release from earlier evidence.
Target texts and target claim spans/unit assignments are retained; source evidence,
protection metadata and relations are reviewed anew. Broad protected-claim spans
are supplemented by specific source facets and constraints in the audit ledger.
Those extra audit fields are not silently added to the existing prompt format.
The legacy `already_structured` status is compiler compatibility, not a finding
that sentence packing is identical. Reference-derived grouping remains an oracle
plan, unavailable as a deployment mapping.

All earlier artifacts, repair-overlay stale-annotation flags, frozen corpus,
experiments and outputs remain unchanged. New bindings apply only to the exact
repaired source hashes. None of the four rows is training-ready or automatically
authorized for generation. No validation or reserved test examples were used.

## Reproduce locally

Private inputs are intentionally excluded from the public repository. With the
existing private repair and parent artifacts present:

```sh
PYTHONPATH=src python3 -m fama_style.research.repaired_source_bindings \
  --specification research_runs/repaired_source_bindings_v1/specification_v1/source_bindings.private.json \
  --output-dir research_runs/repaired_source_bindings_v1/release_v2
python3 -m unittest discover -s tests -v
python3 train.py --check
```

Use a fresh output directory. The builder verifies repair-release hashes, parent
annotation/package hashes and the frozen training hash. The private release holds
paired audit cards, source-only ledgers, per-case source/evidence/plan files,
readiness counts and input/output hashes. Tests cover endpoint ownership,
ambiguous quotes, constraint scope/order, tampering and deterministic private
rebuilds. Private integration tests skip when those artifacts are absent.

The next useful experiment is a small inference-only comparison on the repaired
inputs, with the two qualified cases analyzed separately. It should test a
concrete style intervention against the same ordinary shared-evidence baseline;
binding repair alone does not imply a style gain or justify another training run.
