# Teacher-pair curation v2: expanded training data

Completed a **data-only expansion** after the 16-update pilot. Of the 42 training
candidates previously held for review, 30 are provisionally admitted and 12 remain
held. The result is **46 training pairs across 12 papers/author groups**, with the
same eight development pairs from four different papers/authors. No source or
target text changed, no new paraphrases were generated, and no training job or
new training budget is created here.

Private release: `data/teacher_pair_pilot_v2/`.

- `training.private.jsonl`: the existing 16 pairs plus 30 admitted pairs.
- `development.private.jsonl`: exact original bytes for the same eight rows.
- `private_key/training_audit.private.jsonl`: all 58 candidate dispositions,
  original flags, upstream issues, text bindings and new explanations.
- `private_key/expansion_decisions.private.json`: 42 new review decisions.
- `lineage.json`, `summary.json`, `manifest.json`: provenance and frozen hashes.

The decision source is
`research_runs/teacher_pair_reverse_pilot_v2/curation_v1/decisions.private.json`.
All earlier datasets, review decisions, model results and checkpoints remain
unchanged. Old training row bytes and all development row bytes are preserved,
not merely normalized to equivalent text. Paper, author-group, individual-author
and PDF separation across training/development are checked. The training probes
remain recorded in lineage for continuity with the earlier experiment.

## What was checked

Each of the 42 held inputs was compared with its archived human paragraph and
original training-annotation issues. Decisions concern actors, claims, quantities,
comparisons, causal direction, conditions, uncertainty, negation and attribution.
All original review/context flags and uncertain source status remain visible.
New eligibility is a **limited-pilot assessment**, not independent semantic gold,
new PDF/OCR certification or full-scale training approval.

Admitted cases contain enough explicit content for meaning-preserving rewriting
without filling gaps. Some retain an unexpanded discourse link such as a prior
analysis, an unspecified historical sample, or a table citation. The reviewer
accepts the locally stated claims as reported, without inventing the missing
proof, period, cell values or surrounding argument. This differs from an unnamed
core correlation, undefined technical operation or unresolved condition whose
meaning cannot safely be established.

The audit explicitly retains uncertainties rather than silently repairing them:
quoted historical percentages are not recalculated against unseen tables; proper
names are copied as supplied without new spelling certification; unusual target
phrases and question fragments are not corrected; and qualitative life-cycle
stage wording remains a noted paraphrase limitation. Eligibility does not mean
all extraction uncertainty has disappeared. The admitted data should still be
used only in a separately bounded, measured experiment.

## Remaining holds

| Reason | Pairs |
| --- | ---: |
| Unbound core context | 7 |
| Technical meaning not resolved | 2 |
| Transcription or notation uncertainty | 2 |
| Ambiguous contract condition | 1 |

Examples include an undefined correlation, unidentified filter/spectral shape,
context-specific p-value usage, unclear observation-count scopes, obscure
`time-interest` wording, an unspecified confidence construction and an ambiguous
reward/production parenthesis. They are preserved as holds, not automatically
repaired or discarded. The original parent quarantines outside the 58-candidate
training pool also remain excluded.

No provider scores were read or used in selection, and no detector was queried.
Two admitted pairs had earlier training-side detector-control exposure. That
metadata is recorded; they are not fresh detector-evaluation material. Decisions
are based on source alignment, not on whether a provider called a rewrite human.

## Contrast and limits

| Measurement | Original training | Added pairs | Expanded training | Fixed development |
| --- | ---: | ---: | ---: | ---: |
| Pairs | 16 | 30 | 46 | 8 |
| Median input/reference word-sequence overlap | 0.764 | 0.777 | 0.777 | 0.780 |
| Sentence-count proxy changes | 8 | 15 | 23 | 2 |
| Pairs above 0.90 overlap | 0 | 2 | 2 | 0 |

These are descriptive surface proxies, not syntactic judgments or inclusion
thresholds. Some high-overlap examples supply preservation or local sentence-
structure evidence. No wording differences are forced to reach an edit-distance
target. The expansion adds audited coverage and volume; it does **not** establish
a stronger distinctive-style signal. High overlap and a mixture of authors remain
limits. No Fama/coauthor signature, human-authorship outcome or detector gain is
claimed. Public aggregates are in [curation_results.json](curation_results.json).

Development remains the same previously exposed eight examples; no evaluation
paragraph was replaced, newly repaired or promoted into training after seeing
outputs. The separate 45-example Fama and 20-example multi-author reserves are
not deserialized for curation or used for selection. Only training annotation
bodies are read from the upstream audit. Model-generation results are not
selection features.

## Handoff to a future experiment

This release has no new Kaggle notebook, training configuration or execution
authorization. The old v1 runtime is intentionally fixed to exactly 16 training
pairs: **do not substitute v2 into the existing v1 Dataset/notebook**.
The additional rows have not yet undergone model-tokenizer length/label preflight.
That must occur before any new training, with no truncation or silent row dropping.

A follow-up must predeclare its training budget and comparison. Keeping two epochs
would now yield 46 updates rather than 16 with batch one/accumulation two, changing
both data size and optimization budget. A step-matched control would be needed to
attribute a difference specifically to data expansion. A practical improvement
study can change both, but must report the confound rather than claim a causal
data-volume effect. No such choice or GPU job is made by this curation step.

Reproduce into a fresh directory:

```sh
PYTHONPATH=src python3 -m fama_style.research.teacher_pair_expansion \
  --parent research_runs/multi_author_milestone_v1/package_v1 \
  --previous data/teacher_pair_pilot_v1 \
  --upstream research_runs/human_prose_annotation_v1/audit_v2 \
  --decisions research_runs/teacher_pair_reverse_pilot_v2/curation_v1/decisions.private.json \
  --output /tmp/teacher-pair-v2-reproduction
```

The builder verifies parent manifests and text hashes, requires a decision for
every formerly held candidate, copies old rows verbatim, retains uncertainty and
exposure metadata, checks split separation/duplicates, and refuses overwrite.
CPU tests use synthetic examples, including unreadable non-training bodies that
must be skipped before JSON parsing. No weights, server, credentials or GPU are
needed for these checks. Copyrighted paragraphs and raw annotations remain private.
