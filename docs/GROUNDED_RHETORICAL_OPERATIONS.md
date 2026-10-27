# Reference-grounded rhetorical operations v1

Prepared **four claim-bound transformation hypotheses and one preservation control**, using existing training annotations and unchanged source/reference pairs. No model inference, training, GPU submission or additional ChatGPT annotation request was made. This is the evidence-building step recommended after the unsuccessful style objective in the [supervised-realization pilot](PLAN_REALIZER_PILOT_V1_RESULTS.md).

## What was selected

| Operation | Training evidence | Observed source → reference packing | Status |
| --- | --- | --- | --- |
| Separate premise, supported inference and qualified applicability | Fama 1965, paragraph ending `0012` | 2 → 3 sentences; each move is separated | Contrast hypothesis |
| Separate a retrospective criterion introduction from its definition and rationale | Fama 1965, paragraph ending `0047` | 3 → 4; one composite source claim is exposed in two moves | Contrast hypothesis |
| Stage already-supplied portfolio decisions | Fama 1965, paragraph ending `0049` | 3 → 5; explicit component count precedes staged decisions | Contrast hypothesis |
| Attach an empirical comparison to its already-supported inference | Disappearing Dividends, paragraph `0070` | 7 → 5; comparison/inference claims are packed together | Contrast hypothesis |
| Retain parallel finding/alternative-explanation blocks | Market Efficiency, paragraph `0025` | 5 → 5; claim grouping is already the same | Preservation control |

Sentence counts describe the observed evidence; they are not the new control objective. The new programs specify discourse roles, the claims filling them, existing relationships and expression controls. A stage is a move, not a compulsory one-sentence slot. Programs contain no word-count bands or semicolon budgets.

Selection uses five previously `retain`-annotated training pairs across three papers. It is purposive mechanism discovery, not a prevalence estimate or evaluation on unseen data. Only training files/annotations are read; neither the 21 development rows nor the 45 reserved test rows enters compilation. Five examples, one per operation, are insufficient to train or validate a structural mapper.

## Three levels with semantics kept separate

The machine-readable [catalogue](grounded_rhetorical_operations_v1/catalog.json) supplies finite controls:

- **Paragraph:** order and function of moves, with source claim IDs bound to each move.
- **Sentence organization:** separate a premise from its inference, expose a criterion separately, stage existing decisions, or integrate existing evidence and inference. These are not unrestricted rhetorical-tree generation.
- **Language:** mark an existing contrast/sequence/comparison, echo an existing criterion qualifier, or retain modal, deontic, evidential and attribution scope. There is no target-word glossary or request to invent more technical terminology.

The [closed program schema](grounded_rhetorical_operations_v1/program.schema.json) permits only known operations, stage roles, source claim/fact IDs, source relation IDs and declared language actions. A CPU compiler instantiates these programs on the five evidence cards. It does not generate text or predict which operation a new paragraph needs.

The private source ledger separately preserves the existing LLM claim/entity/relationship annotations, numeric evidence, conditions, negation, modality, attributions and exact source spans. Composite source claims remain composite. Source and target claim IDs are independently annotated; explicit alignment tables handle one-to-many and many-to-one cases instead of assuming matching ID numbers mean matching claims.

The existing three-level annotations were inspected before adding this executable layer. Their move, sentence and language labels are retained on each card. The older consolidation proposal is explicitly marked proposed/not frozen; these operations do not silently replace or standardize the discovered taxonomy. Different label names alone are not treated as evidence of a structural transformation.

## Evidence checks and an annotation gap

Every card records source/reference hashes, upstream annotation hashes, exact claim evidence spans, both sentence-claim partitions, declared claim alignments, target cue spans, observed language labels and uncertainty. Compilation checks:

1. The annotations reconstruct the unchanged paragraphs and cover their declared claims.
2. Every target claim aligns to existing source claim(s), with all source claims represented.
3. The target's annotated claim grouping actually matches the proposed organization; reference cues occur in the expected order and required language labels are present.
4. Referenced semantic edges already exist in the source annotation, with the declared endpoints and direction. No new causal edge or adjudication of rival explanations is introduced.
5. Contrast cards differ in source/target claim packing. The already-parallel pair is explicitly a control.

The portfolio source explicitly introduces three decisions, but its old semantic annotation did not give that introduction a separate claim ID. A **separate source-bound supplemental fact** records the exact source span and count. The original annotation is unchanged. The compiler does not fabricate an extra decision or pretend the old claim inventory was complete.

The dividend card additionally binds **seven numerical roles** to source and reference evidence: predicted rates at two years, the period-average expected/actual rates and gap, and the single-year expected/actual rates. Both the value and claim-local span must match the declared binding. These are inspectable, fixed-case alignments, not an independent economic parser. Existing less-than-one-sixth comparison and inference strength remain in the source evidence; no derived percentage is added.

These checks establish exact evidence coverage, declaration consistency and observed packing. They do **not** prove semantic equivalence, verify the PDF transcription, establish atomic claim completeness, or demonstrate that the operations are uniquely Fama-like. Known uncertainty remains explicit, including target/source differences in probability force and temporal scope.

## Generator boundary

Each private prompt preview is compiled through a separate source-only payload. It contains the original source, anchored source claim evidence, source-supported supplemental evidence, selected existing source relationships and the finite rhetorical program. It excludes the reference paragraph, target evidence cues, target annotations and target claim IDs. Source-derived protected items use exact spans rather than unrestricted generated explanatory descriptions.

The program itself remains **reference-informed oracle side information**. Removing target wording does not remove this structural oracle advantage. These previews are not deployable learned mapping predictions. Tests reject extra target-text fields, unknown claim/relation references, reversed/rebound source edges, changed numerical roles, changed supplemental counts and invalid evidence anchors.

## Files and reproduction

Implementation: `src/fama_style/rhetoric/grounded_operations.py`.
CPU tests: `tests/test_grounded_operations.py`.

Active private package:

`research_runs/grounded_rhetorical_operations_v1/package_v3/`

It contains:

- `operation_cards.private.jsonl`: five source/reference evidence cards and source-bound programs.
- `prompt_preview.private.jsonl`: deterministic future-generation prompts and hashes, with no model outputs.
- `summary.json`: aggregate preparation counts and limits.
- `manifest.json`: immutable upstream input hashes, compiler hash and artifact hashes.

Earlier private package drafts remain preserved and are not the active release. All paragraph-level files must remain private and outside Git publication.

CPU verification:

```sh
PYTHONPATH=src python3 -m fama_style.rhetoric.grounded_operations \
  --check --output-dir research_runs/grounded_rhetorical_operations_v1/package_v3
python3 -m unittest discover -s tests -v
python3 train.py --check
```

For independent rebuilding, omit `--check` and use a fresh output directory. Inputs must match the frozen training and audited-annotation manifests. No ML dependencies, credentials, network access or downloaded model weights are needed. Compilation errors are not silently repaired.

## What this supports next

This package gives a concrete distinction to test: **claim-bound rhetorical reorganization versus shallow formatting controls**, with a preservation control to reveal gains caused merely by extra prompting or copying. It does not yet justify training a mapper or scaling the pilot: the four transformations currently have one supporting paragraph each, and three come from the same paper.

The next bounded realization comparison can use these as training-example oracle diagnostics, retaining ordinary/shallow controls, source/reference identity, cost measurements and separate style/drift assessment. A no-change control must not count as a style-transformation success. Any extension to new paragraphs needs explicit supported claim binding; no automatic transfer of causal relations or evidence strength is allowed.

Before a new training comparison, expand the evidence per operation across papers and keep both ordinary and operation-conditioned targets identical. Do not use generated summaries from the preceding pilot as labels. Retain the existing allowance for minor drift when it brings substantial demonstrated style gain; local structural checks are not a new perfect-equivalence quality gate.

No new Kaggle notebook or GPU job was created at this stage. The full taxonomy, direct-SFT baseline, previous pilot and all prior datasets/results remain unchanged.
