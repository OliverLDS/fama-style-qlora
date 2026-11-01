# Development plan: meaning-preserving academic style transfer

Agreed direction, 30 October 2026. This plan extends the existing experiments;
it does not replace their data, labels, hashes, results or original objectives.

## Current priority — 1 November 2026

The [bounded preference-versus-SFT pilot is complete](actual_error_preferences_v1/PILOT_RESULTS_V1.md):
44 updates, all execution checks passed, 20.3 minutes on one T4. Both final arms
retain four material-issue outputs and three locally eligible outputs out of eight.
Six SFT and five DPO generations are identical to the unchanged control. Neither
passes the quality gate or warrants detector scans. Before further optimization,
the next diagnostic is free generation on eleven fixed training-side
paragraphs using the saved adapters, to separate weak behavioral learning from
poor transfer. That [inference-only diagnostic has now completed](actual_error_preferences_v1/TRAINING_SIDE_RESULTS_V1.md):
22 new generations, zero optimizer steps, eight exact control repetitions per
arm and no resolved material errors across the eleven selected cases. The next
intervention is one bounded SFT-only update-strength check, retaining
the same seventeen pairs and 22 updates but fixing learning rate at `1e-5`.
The user agreed and the [stronger SFT check has completed](actual_error_preferences_v1/SFT_STRENGTH_RESULTS_V1.md):
22 updates, all nineteen outputs changed, and material errors resolved in two
selected training cases. Development target NLL reaches 1.517, but material-error
count stays 4/8 and locally eligible outputs fall to 2/8; a new exclusive-purpose
error offsets a planning repair. No detector scans are warranted. A matched
DPO-only `1e-5` / 22-update comparison will test error contrast
at an effective update strength. The [matched stronger-DPO run has completed](actual_error_preferences_v1/DPO_STRENGTH_RESULTS_V1.md):
22 updates, all reference/execution gates passed, thirteen of nineteen outputs
identical to stronger SFT and unchanged material-error/eligibility counts.
Neither stronger arm passes the development gate. The next proposed work is a
training-only minimal-repair preference audit to isolate local semantic corrections
from full human-reference style and length differences. Preserve human targets as
separate style evidence. Pause further sweeps and detector scans; no new GPU job
or repair-label generation has been initiated. Earlier approvals cover completed
runs, not additional training.

The [real CPU token/mask preflight passed locally](preference_preflight_v1/RUN_V2_RECOVERY.md)
on seventeen pairs and 34 responses. All chosen token sequences exactly reproduce
their archived hashes. Kaggle version 2 exported its tokenizer before an API
return-shape bug; the fix requests flat token IDs explicitly. Original data and
tokenizer assets are unchanged, and no further Kaggle replay is needed.
The [bounded DPO-versus-SFT proposal](actual_error_preferences_v1/PILOT_PROPOSAL.md)
defines independent continuations from the same adapter, twenty-two updates per
arm and the original eight-case milestone evaluation. Its
[implementation and private packet are now prepared](actual_error_preferences_v1/PILOT_IMPLEMENTATION.md),
including a self-contained notebook. The user explicitly approved the bounded
44-update, one-T4, one-hour pilot, which has now completed.
The builder still defaults to check-only unless supplied the bound approval record.
CPU checks and packet
reproduction pass; processor equivalence, reference likelihoods, gradients,
memory fit and adapter reload gates also passed during the completed run.
No job was submitted by the agent, and additional training is not authorized.
Do not substitute a raw-base reference policy or treat synthetic
scoring tests as actual model likelihoods.

The [actual-error preference packet](actual_error_preferences_v1/README.md) is now
prepared: seventeen candidates across eleven training paragraphs and ten papers,
with two uncertain outputs held and five without clear material error excluded.
Chosen references are unchanged; rejections are actual saved adapter outputs.
A matched chosen-only SFT file and paragraph/paper grouping are included. This
is a data-only step, not a training release: chosen responses have higher source
overlap in sixteen pairs and are longer in twelve, so copying and length shortcuts
remain material confounds. The input-token gate is recorded in the separate
recovery audit; trainer packing, actual reference scores, gradient/memory checks
remain GPU execution gates. The bounded pilot is approved; full training is not.

The [completed first detector round](milestone_checkpoint_4b_v1/DETECTOR_RESULTS_V2.md)
now has twenty GPTZero and twenty Pangram reports. No further scans are needed
before addressing the rewriting behavior; no new generation or training is
authorized. Both providers flag all twelve Gemma outputs and label all four
human references human; the frozen GPTZero AI-related endpoint shows no
improvement across the three arms on four cases. The user reports the same
default settings on both Pangram accounts used for the two collection batches;
the account provenance is retained separately.
Use Pangram as the operational default for future sparse checks due to the daily
free allowance, preserving separate native score semantics and the existing
GPTZero checkpoint endpoint. Similar results on twenty shared texts do not
establish statistical equivalence or reliable GPT-6-rewrite detection.

First-round sparse measurement for the [reviewed 4B milestone checkpoint](milestone_checkpoint_4b_v1/RESULTS.md)
is complete; preference candidates are now prepared and a new research decision
is needed before further training. Retain the original frozen eight
development cases across four papers; the second four-case round remains
unscanned. The comparison uses ordinary base rewriting, a source-only structural
prior, and ordinary rewriting with the original expanded-pilot epoch-2 adapter.
Ten compatible archived outputs are reused; all fourteen missing outputs have
completed inference. Provisional
masked review identifies material issues in 7–8 base outputs, five structural
outputs and four adapter outputs. Each alternative has only three outputs within
the local one-minor-shift budget. Generic exposition improves in several cases;
distinctive author style and detector improvements are still unestablished.

Assess meaning, readability, copying and sparse detector outcomes separately.
The versioned rubric permits at most one minor semantic shift for an individually
eligible rewrite and no material claim change; unresolved cases remain unresolved.
The existing detector policy limits project usage to an estimated 8,000 words,
using complete paper-balanced rounds and checking the actual remaining allowance.
No detector improvement or distinctive Fama-style gain is established.

Preference learning remains a candidate diagnostic branch with a prepared packet,
not a proven remedy or a replacement for the milestone. Only the bounded pilot
described above is approved. Historical “next step” notes
below describe earlier decisions; this current-priority section supersedes them.

## Main objective and first milestone

The main objective remains rewriting into **Eugene Fama/coauthor style**, using
direct LoRA/QLoRA adaptation, explicit rhetorical-taxonomy guidance, or a hybrid.
Longer term, retain the option to support a small set of other academic authors.
The repository name and original model ID `google/gemma-3-1b-it` are preserved.

The first operational milestone is **multi-author academic rewriting**:
transform GPT-6-rewritten inputs toward human academic prose while retaining
meaning and readability, and measure whether the results become less detectable
as machine text. Reduced detectability is a useful, explicit research outcome.
It is measured separately from writing quality and does not prove human authorship.
Success here would support the training/data/prompting method; author-specific
Fama transfer still needs its own target-specific evidence.

The existing human-original → GPT-6 rewrite construction is retained; training
direction is GPT-6 rewrite → original human paragraph. We do not require humans
to answer new writing prompts, a paid review panel, or additional paraphrase-model
families now. This specifically studies recovery/transfer from synthetic
rewrites; it does not automatically generalize to arbitrary AI drafts.

The agreed next data-construction comparison adds **neutral reconstruction**:
generate from a detailed, wording-free economic content brief alongside the
existing direct rewrites. Do not deliberately request stereotypical "AI-like"
prose. Preserve provenance and assess semantic alignment and structural contrast
separately from detector scores. The [method design](author_inventory_v1/NEUTRAL_RECONSTRUCTION.md)
has a [completed development pilot](neutral_reconstruction_v1/STAGE2_RESULTS.md);
existing milestone inputs remain frozen.

## Two complementary research tracks

1. **Direct style learning:** pooled multi-author SFT/QLoRA, followed eventually
   by comparisons with author-conditioned adaptation and author-specific adapters.
2. **Structural style learning:** source-only semantic constraints and a common
   three-level representation of paragraph organization, sentence structure and
   language features, with optional learned transformations. Content and style
   remain separate; operations may reorganize supported relations, not invent them.

Ultimately compare base/no structural guidance, base/structural guidance,
adapter/no guidance and adapter/structural guidance under matched model snapshots,
inputs, decoding and measurement. A plan-conditioned adapter would be a distinct
experiment; an ordinary-prompt adapter's behavior under plans is not assumed.
The first structured baseline is explicitly a small source-only structural prior,
not a validated learned taxonomy mapper or an oracle target plan.

Recent local-merge work is frozen as a mixed finding: the 30-word rule has a
small additional-paper gain and still selects bad edits. It may serve as an
optional proposal filter, with at most one local suggestion and abstention.
It is not the repository's main objective or a semantic guarantee.

## Preserve authors rather than erase them

Every pair retains paper/version, author list, stable author-group identity,
source provenance, original/input hashes and annotation history. A pooled model
may learn generic patterns but may also average away distinctive choices. Its
benefit to author-specific rewriting must be tested.

There are two different evaluation questions:

- **Generic transfer to unseen author groups:** retain the existing whole-author/
  whole-paper training, development and reserved splits. Current multi-author
  sources support this design, with earlier project exposure stated explicitly.
- **A specified author's style:** the author must have multiple papers available,
  allowing training on some and evaluation on different papers by that author.
  An author absent from training cannot be treated as a learned conditional label.
  Collect additional papers or create a separate versioned corpus for this axis;
  never repurpose existing reserved sources for development.

Coauthor groups remain identifiable. The existing Fama/coauthor grouping is an
explicit project convention; pooling all authors is not a claim of one universal
human writing style.

The [author inventory v1](author_inventory_v1/README.md) records the existing
37 registered papers and nine completed historical-paper acquisitions. The
initial proposed labels are Fama/coauthors, Shiller, Holmström and Baker–Wurgler
(joint authors), with four available works for each added group. Extraction and
curation are still pending for the additions. Availability is not training
readiness; the original author-held-out corpus and reserved splits stay intact.

The subsequent [inventory v2](author_inventory_v2/README.md) adds 27 more historical
papers, bringing each of these four labels to 13 distinct works. Baker–Wurgler
includes additional coauthors as explicitly approved by the user; Holmström's
coauthored works are also identified. V1 is preserved. Newly downloaded sources
were subsequently processed by [source curation v1](author_curation_v1/README.md):
all 1,250 pages across the 39 non-Fama papers were extracted, producing 1,042
mechanically screened source paragraphs and explicit hold/repair records. These
are annotation candidates, not meaning-verified rewrites or training-ready pairs.

The [12-example method pilot](neutral_reconstruction_v1/README.md) prepares two
content-brief batches, followed by direct versus brief-only generation after
brief assessment. Its [six-paper role commitments](neutral_reconstruction_v1/paper_role_commitments.json)
are training-side only and must be respected by future split builders. Its
[Stage 1 review](neutral_reconstruction_v1/STAGE1_REVIEW.md) received all 12 briefs,
prepared eight provisional paired generation cases and held four. Raw responses
are preserved. The [Stage 2 review](neutral_reconstruction_v1/STAGE2_RESULTS.md)
received all 16 outputs: reconstruction has lower overlap in all eight pairs,
but three outputs add epistemic commentary absent from their sources. A
[targeted instruction check](neutral_reconstruction_v1/INSTRUCTION_CHECK_RESULTS.md)
removed that commentary in all three reruns. A [versioned semantic binding repair](neutral_reconstruction_v1/SEMANTIC_BINDING_REPAIR.md)
then corrected two remaining scope issues in one brief. The derived brief has no
new GPT-6 generation results. GPT-6 is the data preparer and development writer here.
The subsequent [Gemma 4B check](neutral_reconstruction_v1/GEMMA4B_V1_RESULTS.md)
completed six outputs but failed formatting validation. Offline recovery found
claim issues in all three brief-only reconstructions, so transfer is not established
and the detailed-brief format should not yet be scaled or used for training.
The subsequent [matched prose-only 4B comparison](neutral_reconstruction_v1/GEMMA4B_PROSE_RESULTS.md)
completed all nine outputs, without a reliable compact-format fidelity advantage.
Do not make student brief-to-neutral realization a prerequisite for reverse-direction
text-pair SFT. The next direction is a small source-checked teacher-pair pilot,
with separate base/adapter and claim/style/copying comparisons; no training is
authorized by these generation results.
That [small teacher-pair pilot](teacher_pair_pilot_v1/README.md) is now prepared:
16 unchanged training pairs from 10 papers and eight development pairs from four
separate papers/authors, with per-pair source-alignment decisions and original
uncertainty flags retained. The base comparison comes first. The fresh two-epoch
QLoRA path is capped at 16 updates. The user authorized that limited pilot after
baseline review on 31 October 2026; its notebook is enabled for
[manual execution](teacher_pair_pilot_v1/PILOT_RUN.md), with no job submitted.
The [base run is now reviewed](teacher_pair_pilot_v1/BASELINE_RESULTS.md): eight
development generations and two training probes completed without updates, with
finite losses and valid target masks. Claim drift, rather than exact copying,
is the dominant baseline concern. The [authorized pilot is now complete](teacher_pair_pilot_v1/PILOT_RESULTS.md):
16 finite updates, changed/reloaded adapters, lower loss and modest expression
gains, but six of eight final development outputs retain material claim issues.
Preserve the checkpoint and curate more existing teacher pairs before proposing
another bounded experiment; do not scale to full training on these results.
The [v2 data curation](teacher_pair_pilot_v2/README.md) is complete: 30 of the
42 held training candidates are provisionally admitted, making 46 training pairs
with the same eight development rows. Text and original flags are preserved;
12 remain held. Additional data has not been tokenized or trained, and a new
optimization budget/comparison must be specified separately.
The [expanded follow-up plan](teacher_pair_pilot_v2/EXPERIMENT.md) now specifies
two epochs / 46 updates, preserving the same eight development cases and two
training probes. This changes data and optimization budget together. The new
baseline notebook checks all 54 rows before weights load. Following review and
explicit user approval, the pilot is now
[enabled for manual execution](teacher_pair_pilot_v2/PILOT_RUN.md). No new GPU
job is submitted by the enabling tools.
The [expanded baseline review](teacher_pair_pilot_v2/BASELINE_RESULTS.md) now
confirms all 54 target masks and context budgets, ten identical generations and
eight identical loss rows relative to v1. The next step is manual execution of
that authorized, bounded pilot; technical preflight does not demonstrate better
claim retention or style. No full training is authorized.
The [expanded pilot is now complete](teacher_pair_pilot_v2/PILOT_RESULTS.md):
46 finite updates and successful reload, lower loss and six generic expression
wins, but six of eight development outputs still contain material claim issues.
Its predeclared claim-improvement gate fails. Preserve both epochs; next audit
training coverage of temporal distinctions, quantifiers, full numerical examples
and unsupported explanations before proposing another fidelity experiment.
No extra training or reserved-test evaluation is authorized.
The [46-pair coverage audit](teacher_pair_pilot_v2/TRAINING_COVERAGE.md) is now
complete: scope/comparison appears in 42 pairs, temporal distinctions in 30,
numeric content in 13 and bounded reasoning in 44. Exact investment-timing and
numeric probability/outcome contrasts are absent, while all pairs preserve
major claim order. Next prepare 12 training-only anchors with checked faithful
variants and minimal semantic contrasts; this is a data-preparation recommendation,
not authorization for another training run.
Those [12 anchors are now prepared](teacher_pair_pilot_v2/SEMANTIC_ANCHORS.md),
with 56 claim bindings, faithful neutral input variants and one-edit synthetic
rejections. Both SFT and preference formats remain provisional, reviewed by the
same Codex reviewer and not integrated into training. Next validate the candidates
semantically and specify a small inference-only Gemma check before choosing an
objective or authorizing new training.
The [anchor inference notebook](teacher_pair_pilot_v2/ANCHOR_INFERENCE.md) is now
prepared: 48 first rewrites and 96 completion scores across twelve training-side
anchors, original/variant inputs and base/expanded-epoch-2 Gemma. Target and
counterexample text never enters generation prompts. The exact existing adapter
is frozen, there are no optimizer updates, and no GPU job has been submitted.
That [inference check is now complete](teacher_pair_pilot_v2/ANCHOR_INFERENCE_RESULTS.md).
Both models prefer the supplied faithful target by mean NLL on all original
inputs, yet free generation still drifts. The adapter's variant-input improvement
is small, training-side and severity-sensitive. Next propose a fresh controlled
SFT run replacing twelve inputs one-for-one, preserving all 46 targets and the
46-update budget; do not extend the existing adapter or assume synthetic
preference accuracy validates faithful realization. No new training is authorized.
That [controlled input-replacement run](input_replacement_v1/README.md) is now
prepared and explicitly approved for manual Kaggle execution: twelve source
changes, all 46 targets and the 46-update budget fixed, fresh adapter from the
same base. The existing reviewed baseline is reused with a pre-gradient exact
generation/loss check. No GPU job, extra run or full training is submitted by
the preparation tools.
The [matched input-replacement review](input_replacement_v1/RESULTS.md) now
records no measured gain: final loss 1.90146 versus 1.89590, six claim-issue
outputs in each final checkpoint, and one direct expression win versus two losses.
Preserve this negative/mixed result. Next consider data-only preference candidates
from verified actual-error generations on training anchors, with faithful controls
and shortcut checks; do not use development outputs or start another SFT run by
default. No new training or reserved-test evaluation is authorized.

## Sequence of work and decision gates

**1. Freeze the milestone specification and provisional data.** Reuse only audited
nonreserved pairs, quarantine known alignment problems and identity pairs, retain
uncertainty flags, and write an explicit selection audit. Existing 45-example Fama
and 20-example multi-author reserved evaluations remain sealed. Candidate status
does not certify semantic equivalence or authorize training.

**2. Check measurement feasibility before spending GPU time.** Prepare a small
matched set of historical human paragraphs and their GPT-6 rewrites as detector
controls. Record provider/date/version, coverage, false positives on human
controls and flags on AI rewrites. If these inputs already lie at a detector
floor, the detector endpoint is uninformative; do not cherry-pick examples or
pretend a null result demonstrates transfer. GPTZero is the initial external
measurement candidate; broader claims later require another frozen detector.

The [sparse detector policy v2](multi_author_v1/SPARSE_DETECTOR_V2.md) incorporates
the user's free-quota constraint: use matched controls and a frozen small
development subset at checkpoints, with local checks for frequent evaluation.
All twelve controls are not an upfront prerequisite. A raw GPT-6 rewrite's
unflagged result does not establish how Gemma outputs will score; the original
six-pair screen remains an incomplete diagnostic and its files are preserved.

**3. Establish prompted-base baselines.** Prepare ordinary and source-only
structural-prior conditions on development examples, with unchanged input and
original human paragraph controls. Keep references and author identities out of
the pooled generator payload. Save first outputs, exact prompts, model commit,
token lengths, package versions, decoding, runtime and truncation. No output is
reported until execution actually occurs.

**4. Run a bounded multi-author QLoRA pilot after explicit authorization.** Use
the same model revision, quantization, prompt format and development inputs as
the base comparison. Tokenizer checks must establish complete target supervision
and no truncation before training. A 3-row/5-step smoke run verifies plumbing;
it is not evidence of learning. Freeze outputs before external detector testing.

**5. Compare quality and detectability separately.** Blinded automated judgments
cover readability and claim preservation; copying, exact reconstruction and
protected quantities/qualifiers are audited. Local shifts are reported under an
explicit drift budget, not silently equated with material errors or automatically
accepted. Detector flags/scores are a separate outcome, with fixed settings,
human controls and missing/too-short results included in coverage.

Evidence of pilot progress requires an improvement over the prompted base on
the declared detectability endpoint, no increase in material claim drift,
readability wins exceeding losses, and no increase in exact input copying.
Small development samples support a pilot finding, not a significance or broad
deployment claim. Target loss alone is insufficient. Do not tune repeatedly to
one detector or regenerate outputs until they pass.

**6. Return to distinctive author transfer.** Compare pooled versus author-specific/
conditioned adaptation, then structural and hybrid approaches. Use topic-matched
author controls and paper-held-out tests to distinguish style from vocabulary,
topic, citations and memorization. Target-specific automated measures remain
proxies; generic detectability gains do not establish Fama/coauthor alignment.

## Benchmarks, cost and operational boundaries

BUST, NIST GenAI and WATTLE support detector evaluation and domain/rewriting
robustness research; they are not automatic gold rewrites for our task. Check
dataset scope, access and licensing before incorporating any content. Keep
external benchmark evaluation distinct from training. Automated style/fidelity
judges have same-model and retrieval biases; separate chats do not certify
independence. No paid human review is required by this plan.

Keep credentials outside the repository and never print them. Kaggle uses
`HF_TOKEN` through Secrets. Notebooks remain private when they contain private
inputs/outputs; publish code, prompts, metadata and aggregate results only.
Prepare and test on CPU first. Do not submit GPU jobs or start training without
explicit user authorization, and retain manual notebook upload so Secret access
can be enabled before execution. No full training run is authorized by this plan.

The current implementation package is described in
[the multi-author milestone protocol](multi_author_v1/PROTOCOL.md).
