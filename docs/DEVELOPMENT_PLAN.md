# Development plan: author-specific economics rewriting

Updated after the matched stronger SFT/DPO checkpoint. The
[strategic review](STRATEGIC_REVIEW_2026_11.md) weighs the evidence and options.
The former chronological plan is preserved in
[checkpoint Git history](https://github.com/OliverLDS/fama-style-qlora/blob/b856ea92a282c5a146934af3cb782a8cd1569006/docs/DEVELOPMENT_PLAN.md).
This update changes no dataset, split, annotation, hash or completed result.

## Objective and agreed scope

The main goal remains **Eugene Fama/coauthor-style rewriting**, through direct
LoRA/QLoRA, explicit rhetorical structure or a hybrid. Retain other author groups
for comparison and eventual author-conditioned transfer. The original model ID
`google/gemma-3-1b-it` and root `train.py` remain intact; later 4B studies are
separate pinned experiments.

The operational multi-author milestone remains GPT-6-rewritten prose to human
academic prose, measuring meaning, readability, copying and reduced detectability
separately. Detector improvement is an explicit outcome, not proof of human
authorship, writing quality or Fama alignment. Human-original/GPT-6 pairs remain
the current construction; neutral reconstruction is an investigated alternative,
not an automatically superior replacement. No additional paraphrase-model
families or paid human review are required.

## What the completed work establishes

| Question | Evidence | Consequence |
| --- | --- | --- |
| Is there wording contrast? | [247-pair audit](RESEARCH_EXTENSION.md): yes; alignment and author specificity remain uncertain. | Do not diagnose copying as an identity-only dataset problem. |
| Does target structure improve realization? | [Gemma oracle](RHETORICAL_ORACLE_V1_RESULTS.md) fails its gate; [strong-realizer guidance](PAPER_PLAN_COMPARISON_V1_RESULTS.md) has selective organization benefits. | Keep explicit operations as optional controls; no full mapper yet. |
| Are author proxies validated? | [Topic-matched comparison](AUTHOR_COMPARISON_V1.md) is exploratory; [two-proxy replication](AUTHOR_CONTROL_REPLICATION_V1.md) misses full gates. | Author-style measurement remains open. |
| Do stronger updates change behavior? | [Stronger SFT](actual_error_preferences_v1/SFT_STRENGTH_RESULTS_V1.md): all nineteen outputs change; two selected training cases lose material errors. | Earlier near-stasis is not a capacity verdict. |
| Does DPO add fidelity? | [Matched stronger DPO](actual_error_preferences_v1/DPO_STRENGTH_RESULTS_V1.md): thirteen outputs match SFT, with the same error/budget counts. | No demonstrated advantage in this bounded setting. |
| Is the milestone reached? | Both stronger arms have 4/8 material-error development outputs and 2/8 within the local budget. | No full run or new detector scans from these checkpoints. |

Low loss, plan realization and successful execution are different measurements.
Neither faithful distinctive Fama style nor detector benefit is established.
Automated judgment, reused development cases and same-reviewer limits remain.

## Recommended next work: CPU decision packet

**Priority 1: author-style evaluability and exposure.** Use existing registries,
corpora and experiment manifests to distinguish raw papers, extraction candidates,
aligned pairs and prior exposure. Assess whether nonreserved material supports a
paper-held-out, topic/genre-aware author comparison. Reuse earlier author-control
findings; do not enlarge feature searches on the same holdouts until a proxy looks
successful. Propose and freeze measurement criteria before a new benchmark run.

**Priority 2: a bounded fidelity-signal audit.** Proposed minimal-repair preferences
should isolate actual semantic errors from unrelated wording/style differences.
Use training-side responses only; log exact edits and claim evidence, quarantine
uncertainty, and retain human originals as separate style targets. Machine repairs
are not human writing or author-style gold. This is a supporting hypothesis,
not the repository's new main objective or automatic permission for more DPO.

The [strategic review](STRATEGIC_REVIEW_2026_11.md) specifies the decision questions
and alternatives. Neither packet nor repair labels are implemented by this doc
update. Pause further sweeps, bulk annotation and detector collection until a
bounded research question is selected.

## Data and split policy

The [selected inventory](author_inventory_v2/README.md) has thirteen papers each
for Fama/coauthors, Shiller, Holmström/coauthors and Baker–Wurgler/coauthors.
The [39-paper non-Fama extraction](author_curation_v1/README.md) yields 1,042
annotation candidates; these are not verified training pairs. Identify concrete
quality/coverage gaps before more collection or annotation regeneration.

Keep the original Fama corpus, pooled-author corpus and later teacher-pair and
preference subsets separate and versioned. Retain authors, paper/version, source
hashes, transformations and annotation history. Distinguish:

- Pooled transfer to unseen author groups: preserve existing whole-author/paper
  holdouts and acknowledge earlier project exposure.
- A specified author: train on some papers by that group and evaluate on other
  papers by the same group under a new declared protocol. An author absent from
  training is not a learned conditional label.

The **45 Fama and 20 multi-author reserved examples remain sealed**. Do not
repurpose them for feature selection, repairs, tuning or style discovery. Exposure
audits can use split IDs/manifests without opening reserved text bodies.

## Model directions retained

1. Direct: paired rewriting remains the baseline; later compare pooled,
   author-conditioned and author-specific adaptation if data and evaluation
   support those distinctions.
2. Structural: keep semantics separate from paragraph, sentence and language-level
   structure. Prefer interpretable source-only operations over unrestricted trees.
   Existing oracle plans are diagnostics. Multi-paragraph issue structure remains
   a possible extension, not a validated capability.
3. Hybrid: eventually cross base/adapted models with absent/present structural
   guidance under matched sources, snapshots and decoding. Plan-conditioned
   adapters are distinct experiments.

Do not select a method for complexity or edit magnitude alone. Keep ordinary
rewriting and unchanged input as controls. Separate generic readability from
author-specific choices and claim preservation.

## Execution, evaluation and publication boundaries

Before any new GPU experiment, complete CPU checks and freeze its hypothesis,
matched baseline, eligible data, primary checkpoint, cost cap and stop rule.
Earlier approvals cover completed runs, not additional work. No GPU jobs, model
calls or annotation batches begin from this plan.

Completed experiments retain the versioned
[one-minor/no-material rubric](milestone_checkpoint_4b_v1/quality_rubric.json).
Report repairs, new errors and uncertainty separately. Any different future
tradeoff must be declared in advance; do not loosen old gates retrospectively.

Detectors remain sparse checkpoint measurements under the
[existing policy](multi_author_v1/SPARSE_DETECTOR_V2.md). Preserve native labels,
versions, provenance and missing results. Pangram is the practical later default
under the user's free quota; the [provider comparison](milestone_checkpoint_4b_v1/DETECTOR_RESULTS_V2.md)
does not establish general equivalence. No rescan-until-pass behavior or new paid
services are proposed. BUST, NIST and WATTLE remain possible external measurement
resources, not automatically suitable rewriting targets or training data.

Keep credentials outside Git and never print values. Kaggle retrieves HF_TOKEN
through Secrets; retain manual UI upload so access is enabled before execution.
Publish code, synthetic tests, methods, aggregates and limits; real prose,
datasets, raw logs/responses, tokenizers and checkpoints stay private. No full
training run is authorized.
