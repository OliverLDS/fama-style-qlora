# Strategic review after the matched SFT/DPO checkpoint

This is a reassessment and proposed ordering of work, not a new experiment or
GPU authorization. The research checkpoint is
[`b856ea9`](https://github.com/OliverLDS/fama-style-qlora/commit/b856ea92a282c5a146934af3cb782a8cd1569006).
Original datasets, annotations, hashes, adapters and results remain preserved.
The former chronological plan is available
[in Git history](https://github.com/OliverLDS/fama-style-qlora/blob/b856ea92a282c5a146934af3cb782a8cd1569006/docs/DEVELOPMENT_PLAN.md).

## Recommendation

Recenter the next research decision on **distinctive Fama/coauthor style**, with
claim preservation as a separate constraint. Pause optimization sweeps,
full-taxonomy training and new detector scans. First audit whether existing data
can support a credible author-style comparison. Keep minimal-repair preferences
as a small supporting fidelity study, not the next default training run or a
replacement for human style targets.

Three questions need different evidence: whether an author group has distinctive,
transferable choices beyond topic and generic polish; whether the training pairs
isolate those choices without losing meaning; and whether the student can realize
them. More optimization cannot by itself answer the first two. A classifier or
clean dataset would not by itself prove generation success either.

## What the evidence supports

| Evidence | Supported conclusion | Remaining limit |
| --- | --- | --- |
| [247-pair signal audit](RESEARCH_EXTENSION.md) | Wording contrast exists. | Contrast does not certify alignment or author specificity. |
| [Gemma oracle structure](RHETORICAL_ORACLE_V1_RESULTS.md) | Plans modestly affect adherence/counts. | The predefined mapper gate failed; reliable realization/style gain is unestablished. |
| [Taxonomy consistency](TAXONOMY_TRAINING_SIGNAL_V1_RESULTS.md) | Some inspectable grouping hypotheses exist. | Identical-text annotations vary; the 132-category catalogue is not a reliable paired target. |
| [Six-paper GPT-6 realization](PAPER_PLAN_COMPARISON_V1_RESULTS.md) | Readable plans control organization. | Incremental benefit is selective and reference-informed, not a deployed source-only mapper or Gemma result. |
| [Boundary classifier](human_prose_annotation_v1/BOUNDARY_CLASSIFIER.md) | Partial merge signal exists. | Exact layouts do not beat KEEP; layout prediction is not author-style generation. |
| [Author comparison](AUTHOR_COMPARISON_V1.md) and [control sensitivity](AUTHOR_CONTROL_REPLICATION_V1.md) | Empirical controls and exploratory probes exist. | Rhetorical classification is weak; neither highlighted proxy passes full replication. |
| [Stronger SFT](actual_error_preferences_v1/SFT_STRENGTH_RESULTS_V1.md) | Stronger updates change behavior and repair some claims. | Development fidelity fails; generic expression is not Fama style. |
| [Matched stronger DPO](actual_error_preferences_v1/DPO_STRENGTH_RESULTS_V1.md) | Implementation and preference margins improve as intended. | Thirteen of nineteen outputs match SFT; no added fidelity benefit is demonstrated. |
| [Detector comparison](milestone_checkpoint_4b_v1/DETECTOR_RESULTS_V2.md) | Both providers flag the twelve tested Gemma outputs. | Four cases and detector blind spots do not establish general accuracy or author style. |

Successful execution, surrogate loss, plan adherence and detector labels are
different endpoints. None can substitute for the author-style question.

## Weighing the next options

| Option | Priority now | Reason |
| --- | --- | --- |
| More epochs, rates, betas or model sizes | Defer | Optimization already moves without a reliable fidelity/style payoff; reused cases invite overfitting. |
| Collect many PDFs or annotate everything | Conditional | Raw source supply is substantial; establish usable-pair and evaluation gaps first. |
| Train the full rhetorical mapper | Defer | Annotation consistency, author specificity and student realization remain unresolved. |
| Audit author-style evaluability and exposure | First | Directly addresses the main objective and prevents an indefinite generic-fidelity detour. |
| Minimal semantic-repair preference audit | Small supporting study | Tests a specific training-signal hypothesis, but does not teach author style by itself. |
| More detector reports | Defer | Neither new checkpoint meets the frozen quality gate. |

## First: define the next author-style comparison from existing evidence

The [selected roster](author_inventory_v2/README.md) already has 52 papers:
thirteen each for Fama/coauthors, Shiller, Holmström/coauthors and Baker–Wurgler/
coauthors. The [non-Fama extraction](author_curation_v1/README.md) supplies 1,042
annotation candidates, not 1,042 verified pairs. The original Fama and later
pooled corpora remain separate; do not merge their counts or split roles.

A bounded CPU-only decision packet should establish:

- Which nonreserved papers actually have usable pairs, stable author groups,
  extraction quality and alignment evidence, using existing registries/hashes.
- Every paper's prior training, prompt/feature discovery, annotation and evaluation
  exposure. Renaming a reused split does not create a fresh benchmark. Reserved
  membership can be checked from IDs/manifests without opening text bodies.
- Which proposed distinctions have empirical support across papers and matched
  controls. Do not promote the failed relative-marker/passive proxies into rules,
  or equate semicolons, formality and density with Fama style.
- A prospective whole-paper/topic-held-out evaluation, with feature and threshold
  selection confined to training papers. Include lexical/nuisance comparisons,
  genre/extraction sensitivity and paper-level shuffled-label controls where
  feasible. Correlated paragraphs are not independent author samples.

This is not a request to rerun the old classifier until it succeeds. Its weak
and sensitive results are starting evidence. Freeze the next decision criteria
before execution. If existing data cannot support a stable endpoint, report
specific missing papers, genres or annotations before targeted collection.
An automated style panel can be useful yet imperfect; it must not be called an
objective author benchmark merely because it produces a number.

## Supporting study: isolate semantic correction without losing the style target

The seventeen current preferences contrast full human originals with wrong model
responses. Style, wording and often length differ alongside truth. A separate
training-only audit could retain each exact rejection and make a minimally
corrected candidate, recording exact edits, claim evidence and remaining
uncertainty. Do not force a fixed edit distance when missing claims need larger
repairs. Quarantine ambiguity rather than force labels.

Such positives are **machine repairs**, not human originals or Fama-style gold.
Human paragraphs remain separate style targets. Compare semantic locality and
lexical/length contrast before any new optimization. The hypothesis is that this
better isolates fidelity; it is not a proven cause, implemented solution or a
reason to replace the paired style corpus. No repair labels are created here.

## Choosing a future model experiment

Keep direct, structural and eventual hybrid approaches:

- Direct: source to author-group style; use an explicit label only when the group
  has training papers and different evaluation papers. Pooled rewriting remains
  a baseline, not proof of any particular voice.
- Structural: retain source text, separate semantic content and interpretable
  source-only operations. Compare ordinary instructions, fixed empirical priors
  and simple selectors before unrestricted trees or a full neural mapper.
- Hybrid: cross adaptation with structural guidance after each component has
  independently useful evidence. Reference-informed oracle plans stay diagnostics.

Author-specific evaluation holds out papers from authors represented in training.
The current author-held-out pooled split asks a different question: generic transfer
to unseen groups. Keep exact coauthors and the Fama/coauthor convention; do not
claim to isolate an individual's contribution.

Select one GPU experiment only after specifying the bottleneck, eligible examples,
matched baseline, prospective endpoints, stop rule and bounded cost. More changed
words are not automatically better. Keep meaning, expression, author alignment
and copying separate. The completed one-minor/no-material rubric stays fixed;
any different future drift budget must be prospective and reported alongside it,
not used to turn past failures into successes.

## Keep the agreed operational milestone

Multi-author academic rewriting and reduced detectability remain explicit outcomes.
They do not replace or prove Fama transfer. Detector measurements remain sparse,
fixed and separate from quality; Pangram remains the practical later provider
under the existing free-quota policy. No paid plan or new accounts are proposed.
Human originals and unchanged-input passthrough remain controls.

Keep the 45 Fama and 20 multi-author reserved examples sealed. No paid human
review or extra paraphrase-model families are required. Credentials, copyrighted
reference paragraphs and raw artifacts stay private. Earlier GPU approvals cover
completed runs; this review authorizes no new training, generation, annotation
batches or detector calls.
