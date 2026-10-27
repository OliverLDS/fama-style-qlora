# Operation expansion and matched-pilot preparation v1

**Prepared: 30 annotation candidates across all seven training papers, a strict response auditor/compiler, and a gated matched-pilot design. No new eligible training set or GPU job is claimed.**

The preceding external-realizer check showed a provisional benefit for three operations, but all three positive examples came from the 1965 paper. This stage tests whether the frozen corpus supports paper-diverse examples before another realization-training experiment.

## Selected candidates

Read only the frozen 247-row training file. Filter for bounded word budgets and unambiguous heuristic source/target unit partitions, leaving 235 candidates; exclude 11 uncertain partitions and one length outlier. Fixed selection contains:

- Three previously positive anchors, one per successful family.
- Three preservation/no-increment diagnostic probes, excluded from primary training.
- Twenty-four additional cue/diversity-selected candidates, not necessarily previously unseen in the project.

| Paper key | Candidate pairs |
| --- | ---: |
| F1965 | 4 |
| F1980 | 4 |
| FF2000 | 5 |
| FJ1983 | 5 |
| ME1997 | 4 |
| NL2003 | 4 |
| VL2018 | 4 |

There are nine candidate hints per family plus three probes. Hints use target cues and deterministic paper balancing; they are **not labels or proof of support**. Neither the hints, paper identities nor selection roles are shown to the annotator. It may assign another family, preserve an already-matching structure, reject a candidate or flag uncertainty. No performance output enters selection. Sources and original author targets remain byte-identical to the frozen corpus; no GPT rewrite is substituted as a target. Development and reserved test data are not read.

## Exact seeds versus related variants

The [family catalogue](operation_expansion_v1/family_catalog.json) is explicitly versioned as candidate extensions. It preserves links to the validated seed operations:

1. Premise → supported inference → qualified applicability.
2. Existing criterion introduction → definition → source-supported rationale, with the validated seed's advance/retrospective contrast.
3. Existing task/component introduction → staged decisions/components, with the validated seed's scope and recommendation.

A generic definition is not automatically the same operation as the retrospective criterion. A numerical list is not automatically staged policy decisions. Exact seeds must match the catalogue's closed role sequences and satisfy their source/target evidence. Related variants remain a separate output and require their own realization check. The successful three examples do not validate every broader family member.

This separation may reveal insufficient examples. If the corpus does not support roughly 20–30 eligible pairs, report that gap; do not meet a quota by changing the definition or forcing annotations. All six batches are annotation requests, not six completed experiments.

## Annotation task and checks

The private package provides both the unchanged neutral source and original reference. **Reference wording is intentionally visible for annotation**, unlike generation. It supplies fixed source/target units; uncertain segmentation is not silently repaired. The [closed schema](operation_expansion_v1/annotation.schema.json) requests:

- Source-based claim IDs, epistemic status, composite-claim flags and exact source/target evidence.
- Anchored actors, quantities, conditions, uncertainty, negation, scope, comparisons, attribution and causal direction.
- Complete ordered source/target unit-to-claim maps.
- Source-evidenced relations, their endpoints/type/certainty, and no invented causal relation.
- Ordered rhetorical stages, source claim bindings, observed target evidence and finite language controls.
- Supported/already-structured/unsupported/uncertain status, exact-seed/related-variant designation and explicit limitations.

The CPU auditor checks identity/hash/schema, evidence occurrences, unit and claim coverage, relation endpoints, required roles, target stage anchors/order and absence of invented claim/relation IDs. It distinguishes observed packing/order changes from a relabeled existing structure. These are declaration/evidence consistency checks, not proof of atomic completeness, semantic equivalence or author specificity. Notes retain scope/strength uncertainty; this is not a new zero-drift quality gate.

The importer preserves original response files, records all errors/missing IDs and excludes every copy of duplicate cases. It never repairs JSON, edits sources/targets, chooses a preferred retry or promotes related variants into primary training by default. The three probes remain diagnostic even if an annotator calls them supported. No human-review dependency is introduced; returned evidence still receives automated/in-session assistant assessment.

## Matched pilot design

The [machine-readable design](operation_expansion_v1/pilot_design.json) is **design only**. A runnable training notebook is intentionally not released before the data exist and are audited.

Primary annotation gate:

- 20–30 supported exact-seed pairs with observed claim packing/order contrast.
- At least three papers overall and at least four pairs/two papers per family.
- No paper above 40% of the primary set.
- Related variants held separately until a realization check; probes not training examples.

Passing this count gate still does not mark the experiment ready to train. Remaining work includes assessment of the returned claim alignments, token/memory preflight and separate development-oracle plan preparation. If gates fail, the report identifies the shortage rather than silently lowering them.

When justified, train **two fresh matched adapters** from pinned base Gemma 3 4B: direct with shared source evidence versus operations with the same evidence. Use identical original author targets, examples, initialization, shuffled schedules, rank-8 language q/v LoRA, alpha 16, dropout 0.05, LR 5e-5, accumulation 2 and four epochs. At 20–30 pairs, this implies 40–60 updates per arm. One visible T4, NF4/FP32; fresh adapters, not continuation of the old shallow adapter. No GPU fit or runtime is certified yet.

The compiler preserves the validated seed program vocabulary, paragraph/sentence actions and source-evidence scaffold in primary prompts. The same renderer is used at training and evaluation, so an operation-trained model actually learns the richer format. Related variants use an explicitly marked hypothesis format and do not enter the primary arm. Reference words, target evidence, annotation notes and assessment metadata are stripped from generator prompts.

The planned training ceiling is 2048 tokens, generation context ceiling 4096, with **no truncation**. Actual tokenizer fit and memory must be checked before training; reapply data-size/diversity gates after any context exclusions. Both arms must have the same examples and supervised targets, but operation prompting has extra input-token cost. Preflight cannot be replaced with an assumption that a longer prompt will fit.

Evaluation will keep source-grounded style change, claim drift, copying, operation realization and cost separate, with the established allowance for minor drift when substantial style gains warrant it. Original development data remain outside operation discovery; reference-informed development plans are oracle diagnostics. Training probes are not generalization evidence. All 45 reserved test examples remain excluded. A learned mapper still waits for demonstrated realization across papers.

## Files and next action

Active private package: `research_runs/operation_expansion_v1/package_v1/`.

For each XX from 01 to 06, use a separate conversation with the same selected model. Attach only:

- `batches/annotation_batch_XX.jsonl`
- `annotation.schema.json`
- `operation_families.json`

Paste `ANNOTATION_PROMPT.md`. Keep the first returned `annotations_XX.jsonl` unchanged and save it under:

`research_runs/operation_expansion_v1/responses/`

Do not attach `private_key.private.json`, reports, old responses or manifests. Keep paragraph files private and credentials outside them. New conversations do not prove independence or remove memory effects; no memory-clearing procedure is required.

When responses arrive, run:

```sh
PYTHONPATH=src python3 -m fama_style.rhetoric.operation_expansion \
  --audit --package research_runs/operation_expansion_v1/package_v1 \
  --responses-dir research_runs/operation_expansion_v1/responses \
  --output-dir research_runs/operation_expansion_v1/response_audit_v1
```

The audit writes coverage/errors, checked annotations, provisional primary pairs, separately held variants, dispositions and the pilot gate. Use a fresh directory per audit. No model/API request, dependency download, Secret or GPU is accessed by preparation/auditing.

Implementation: `src/fama_style/rhetoric/operation_expansion.py`; tests: `tests/test_operation_expansion.py`. All previous datasets, annotations, results, notebooks and original 1B CLI are preserved.
