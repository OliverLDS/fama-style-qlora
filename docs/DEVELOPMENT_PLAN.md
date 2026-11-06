# Development plan: author-specific economics rewriting

Updated through the 500-pair reconstruction, prompt and Pangram checkpoints. The
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

## Current decision

The [500-pair pilot](data_scaling_v1/GPT6_PURPOSE_500_RESULTS_V1.md),
[paired prompt diagnostic](data_scaling_v1/PURPOSE_500_PROMPT_INPUT_RESULTS_V1.md)
and [Gemini-route Pangram checkpoint](data_scaling_v1/PURPOSE_500_PANGRAM_RESULTS_V1.md)
are complete. Explicit re-expression reduces copying, but neither reliable
author-style transfer nor new human-label conversion is established. The next
proposed intervention is an inference-only fractional contribution comparison
with the same saved adapter, revised prompt and existing inputs. No follow-up
job is submitted by this plan. The data/scaling sections below retain the
history and limits of earlier decisions; they do not authorize new training.

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

## Recommended next work: author-style coverage

The [staged quality/scaling plan](data_scaling_v1/README.md) now sets the operational
milestone at 200, 500, 1,000, 1,500 and 2,000 accepted training pairs, with matched
evaluation. The [updated teacher/evaluation policy](data_scaling_v1/TEACHER_AND_EVALUATION_POLICY_V2.md)
retains GPT-6 data and makes detector checkpoints secondary when inputs already
receive human-written labels. Every inferencer call is free-tier only. The
[source/QA salvage](data_scaling_v1/PAIR_SALVAGE_V1.md) is now executed:
230 usable examples from the 249 collected responses, plus 40 reassessed older
training pairs. A balanced 200-pair cohort across 48 papers is frozen. The initial
41-row gate is superseded for direct text pairs; no new manual QA batches are
requested. Automated adjudication is not independent human ground truth.

The [500-pair expansion](data_scaling_v1/STAGE_0500_RELEASE_V1.md) is now frozen:
125 per group across 48 distinct PDFs (53 stored IDs), with the exact original
200-row prefix retained. The [lineage audit](data_scaling_v1/GPT6_TRACK_0500_PREPARATION_V1.md)
resolves five Fama paper aliases; frozen artifacts are unchanged.
Existing pairs and free-tier Gemini generation supply the additions; minimal
semantic patches, PDF-checked child repairs and exclusions have versioned audits.
CPU checks and an identical rebuild passed. The
[mixed-500 pilot completed](data_scaling_v1/STAGE_0500_RESULTS_V1.md): all 500
examples, 250 updates and 48 matched outputs. Automated review finds fewer
material issues and more copying; distinctive author-style improvement is unproven.

The subsequent [two-teacher plan](data_scaling_v1/TWO_TEACHER_TRACKS_V1.md) adds
separate automated Gemini and manual GPT-6 learning curves over matched human
targets. The [full Gemini input/base/adapter Pangram panel](data_scaling_v1/STAGE_0500_PANGRAM_FULL_V1.md)
shows one improvement, four deteriorations and three ties versus actual inputs.
All eight human references now score 100% Human Written, repeating earlier human
calibration. No further routine reference scans are requested. Weigh the mixed
input/output findings before another training stage. Gemini qualification is paused
at the free quota; while waiting, all eight GPT-6 uploads have returned. The
[quality audit](data_scaling_v1/GPT6_TRACK_0500_QUALITY_V1.md) stages all 500
matched candidates, including 273 unchanged reuses. Of the 227 new inputs, 212
have no identified material issue, ten have local repairs, and five reveal
inherited target-selection/boundary issues. Resolve these shared target issues
consistently before final teacher-specific releases; no batch recollection is needed.
The subsequent [contrast audit](data_scaling_v1/GPT6_CONTRAST_RECONSTRUCTION_V1.md)
separates this semantic assessment from useful rewriting signal. It queues 175
weak-contrast inputs for versioned replacement, retains 320, and defers the same
five target problems. Neutral briefs and paragraph reconstruction use separate
requests so reconstruction does not receive the original prose. Substantial style
contrast takes priority; modest emphasis shifts are allowed, with major errors
assessed separately. This is selective data improvement, not a discard of the
227 returned responses or an authorized training run.
The [175 reconstructions have now been assessed](data_scaling_v1/GPT6_CONTRAST_OUTPUTS_V1.md):
163 are retained as returned, ten have local corrections, and two carry inherited
source concerns. A 500-target candidate version stages 173 replacements while
preserving all prior versions. Resolve the seven source/target follow-ups before
final release; copying diagnostics have improved substantially, while student-model,
author-style and detector gains remain unmeasured.
The next [matched brief/prose routes](data_scaling_v1/BRIEF_AND_PROSE_TRAINING_ROUTES_V1.md)
extend brief-first reconstruction to all 500 GPT-6 targets, reusing the 175 completed
responses. Eleven new Stage 1 jobs cover 320 missing clean targets; five further
missing targets await repair. Compare brief-to-human and fresh-prose-to-human
training only after qualification and explicit GPU authorization. Earlier direct
rewrites remain robustness variants, not additional independent targets.
All [320 expansion briefs are now reviewed](data_scaling_v1/GPT6_FULL_BRIEFS_V1.md)
and included in eleven reconstruction-only jobs. Two briefs have local
clarifications; no Stage 1 recollection is requested. Assess fresh prose after
those responses return; the complete 500-target release remains pending repair.
The [320 reconstructions have now returned](data_scaling_v1/GPT6_FULL_RECONSTRUCTION_OUTPUTS_V1.md):
305 are retained unchanged and fifteen have local repairs. Both brief-to-human and
fresh-prose-to-human candidates now cover the same 493 usable targets. Seven cases
remain explicit follow-ups before a complete release. The median prose/target word
ratio of 0.647 also means expansion is a potential training confound. No new GPU
run or detector evaluation is authorized by this data assessment.
The [seven source/target decisions](data_scaling_v1/SHARED_TARGET_REPAIRS_V2.md)
are now implemented in a versioned shared registry. P134/P139 are recovered,
bringing both GPT-6 candidate routes to 495 targets. One five-record manual job
uses [purpose-aware briefs](data_scaling_v1/BRIEF_PURPOSE_V2.md) for the remaining
repaired targets. Existing 495 briefs are reused, not relabeled or recollected.
Six changed targets also require Gemini source reassessment before final matched
teacher releases. This is data repair/preparation, not new training authorization.
The user subsequently requested [full brief/prose recollection](data_scaling_v1/BRIEF_PURPOSE_V3.md)
for all 500 targets, including the existing 495. First run the new five-case v3
pilot with expanded relation labels and no particular-word example. After brief
review, compare identical content with and without explicit purpose/relations.
Freeze that protocol before preparing the remaining 495 jobs; preserve all older
responses as baselines rather than treating them as final replacements for this round.
The five v3 briefs returned but retained redundant summaries and source-order
organization. The [v4 retry](data_scaling_v1/CONTENT_ABSTRACTION_PILOT_V4.md)
uses a short topic, subject-grouped fragments, non-narrative fact anchors and
alphabetical indexing, while preserving the nuanced content-relation vocabulary.
Retry the same five targets, review actual outputs, then prepare the matched
realization conditions before freezing the full 500-target recollection protocol.
The five compact v4 briefs returned. At the user's direction, the reconstruction
comparison is prepared directly from the unchanged outputs: five content-only
and five guided jobs, with the same base instruction and shared content. No
suggested output-label or wording adjustments are applied; results remain pending.
The [ten reconstruction outputs have returned](data_scaling_v1/CONTENT_ABSTRACTION_RECONSTRUCTION_V4_RESULTS.md).
Added guidance helps selected comparisons/precision but has no consistent advantage
for neutral-pair construction and transfers an extra caveat in P484. Content-only
has lower target sequence similarity in four cases; this is a lexical diagnostic,
not a style win count. Use content-only as the provisional reconstruction default
while retaining structured briefs for brief-to-human research. No outputs were
edited, no full-500 recollection was executed, and no student training was started.
The user selected [enhanced briefs with purpose-only reconstruction for all 500](data_scaling_v1/GPT6_PURPOSE_ONLY_0500_V1.md),
superseding the provisional content-only recommendation. Keep all relation
annotations in storage, but send topic, purpose, points and anchors to realization.
Five pilot briefs are reused; 17 extraction jobs cover the remaining 495. All 500
will be reconstructed again under this policy, including those five pilot targets.
The purpose-only arm is not either prior pilot condition and has no reported
performance yet. Stage 2 packaging awaits the actual brief responses.
The [full 500 brief audit and handoff](data_scaling_v1/GPT6_PURPOSE_ONLY_BRIEFS_V1.md)
now includes all returned records unchanged in seventeen purpose-only realization
jobs. Formal checks pass, while content follow-ups remain explicit. Relation
arrays are empty in 478 records, so the retained annotations must not be called a
complete explicit-graph corpus. No relation data are sent to reconstruction.
All 500 are retained for this generation round; training qualification remains pending.
The [full 500 reconstructions returned](data_scaling_v1/GPT6_PURPOSE_ONLY_OUTPUTS_V1.md).
All pairs are staged unchanged; target overlap is lower in aggregate but mixed by
case. P139 introduces an opposite relative-price claim and P400 changes a current-
return question to forecasting; one two-record upstream recollection is prepared.
P056/P206 model warnings are nonblocking after context review. No blanket rejection,
complete semantic certification, detector improvement or training result is claimed.
Current mixed releases remain historical baselines; do not label
them as pure-teacher curves or request redundant manual annotations.

The [CPU decision packet](AUTHOR_STYLE_EVALUABILITY_V1.md) records the starting
coverage gap. The authorized [pairing expansion](author_pair_expansion_v1/README.md)
prepared 249 paragraph inputs across 54 papers, and the roster reaches twenty
works per group. All 249 responses have explicit salvage decisions; earlier
responses, targets and experiments remain unchanged. The
[first capped 200-pair pilot](data_scaling_v1/STAGE_0200_RESULTS_V1.md) completed
all 100 updates and 32 matched outputs. Automated review finds better claim
retention but much more input copying. The
[completed simpler-prompt comparison](data_scaling_v1/SIMPLE_PROMPT_RESULTS_V1.md)
uses the saved adapter and unchanged development inputs. All 16 adapter outputs
copy less, with more substantive meaning issues identified. The
[32 imported Pangram reports](data_scaling_v1/PANGRAM_PROMPT_COMPARISON_V1.md)
give entirely-human labels to 6/8 strict-adapter outputs and 0/8 simpler-adapter
outputs, with 0/8 for both base conditions. Assess that prompt-dependent detector
benefit alongside input retention, meaning and readability
under the [matched input/output detector protocol](data_scaling_v1/MATCHED_INPUT_OUTPUT_DETECTOR_PROTOCOL_V2.md).
The [eight input controls](data_scaling_v1/INPUT_BASELINE_RESULTS_V1.md) are now
scored: all eight already receive human-written labels. The current panel cannot
measure conversion of detector-positive inputs. A separate local batch runner
now trials Gemini through inferencer using the user-confirmed Free-tier project;
existing GPT-6 pairs remain unchanged. Larger self-hosted teachers remain an
alternative, requiring hardware and alignment checks. Future detector panels must include
input scans and a separate Gemma-generated input condition; no new GPU job is
initiated by this protocol update. Interpret these findings
before selecting another stage. No further annotation is needed now. Consider thirty only for demonstrated gaps;
twenty is source coverage, not a sample-size guarantee or training result.

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
and alternatives. The coverage audit is implemented; minimal-repair labels remain
proposed. The current annotation batches address cross-paper coverage. Further
GPU sweeps and detector collection remain paused pending alignment and style evidence.

## Data and split policy

The [selected inventory](author_inventory_v3/README.md) has twenty works each
for Fama/coauthors, Shiller, Holmström/coauthors and Baker–Wurgler/coauthors.
The [39-paper non-Fama extraction](author_curation_v1/README.md) yields 1,042
annotation candidates, and the separate 28-work expansion adds 569. These are not
verified training pairs. Identify concrete quality/coverage gaps before further
collection or annotation regeneration.

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

## Prepared next checkpoint: GPT-6 purpose-only 500

The user accepted the final dataset adjudications and requested an AI-prose →
human-prose notebook. The [isolated 500-pair pilot](data_scaling_v1/GPT6_PURPOSE_500_PILOT_V1.md)
retains all 500 reconstructions, clarifies only P139's target, and records P400
as user accepted. It uses the simple rewrite instruction for both training and
evaluation, one bounded epoch of fresh Gemma 3 4B QLoRA, and the existing
paper-held-out development panel. Earlier audits, datasets and runs remain
unchanged. This is preparation for manual Kaggle execution, not a submitted
job, a brief-to-prose experiment, or evidence of student improvement.

Version 1 has now [completed](data_scaling_v1/GPT6_PURPOSE_500_RESULTS_V1.md): all
500 pairs and 250 updates verified, but lower validation loss coincides with
more copying and several repetitive or content-losing generations. Treat the
saved adapter as diagnostic, not a successful style checkpoint. The next
recommended comparison is inference-only on a small, fixed held-out panel with
fresh reconstruction inputs alongside the prior direct-paraphrase controls.
The reconstruction-versus-paraphrase input mismatch is measured, not proven to
cause the failures. No new GPU training or detector packet is authorized by
this recommendation; retain the full dataset and existing adapter.

The user authorized the inference diagnostic and added the rewrite prompt as a
possible copying cause. The [prepared factorial diagnostic](data_scaling_v1/PURPOSE_500_PROMPT_INPUT_V1.md)
therefore compares three input routes and two instructions on four fixed
held-out contents with both base and saved adapter (48 outputs, zero updates).
Two free Gemini calls supplied fresh reconstructions automatically. It is ready
for manual Kaggle upload; no inference job has been submitted. The clean paired
prompt intervention and the more confounded teacher/input-route comparison
must be interpreted separately.

The [completed prompt/input diagnostic](data_scaling_v1/PURPOSE_500_PROMPT_INPUT_RESULTS_V1.md)
supports instruction sensitivity: explicit re-expression sharply reduces
adapter copying on this four-content panel. It does not establish a reliable
style advantage; fresh Shiller input still yields a repetition loop under the
simple prompt and a missing-passage request under the revised one. Retain the
re-expression prompt as a diagnostic intervention. The next proposed check is
fractional contribution from this saved adapter under fixed revised prompts,
using existing inputs and no training or new annotation. No such follow-up has
been submitted.

The user requested Pangram evaluation on the Gemini routes first. The
[prepared detector packet](data_scaling_v1/PURPOSE_500_PANGRAM_V1.md) uses the
revised prompt, both Gemini routes and matched input/base/adapter comparisons.
Four exact input scans and existing published-human calibration are reused;
19 new scans cost an estimated 29 credits across two rounds. The short Shiller
missing-passage response remains a task failure without a detector score. No
detector scores or new adapter-strength results have been produced yet.

The [completed Pangram checkpoint](data_scaling_v1/PURPOSE_500_PANGRAM_RESULTS_V1.md)
now includes nineteen new PDFs plus four reused exact input observations. No
scored AI-related input becomes Human Written; the adapter's sole human label
preserves an already-human input. Two scans omit only model-added prefaces or
headings and are marked body-only, with an unchanged conclusion in the strict
full-output subset. The short Shiller response remains a task failure with no
detector score. Retain the data and reports; lower copying alone has not met
the detector milestone. No additional scans or training were started.
