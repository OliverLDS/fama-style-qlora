# Development plan: meaning-preserving academic style transfer

Agreed direction, 30 October 2026. This plan extends the existing experiments;
it does not replace their data, labels, hashes, results or original objectives.

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
new generation results. GPT-6 is the data preparer and development writer here;
the next model-facing gate is Gemma 4B inference-only testing, not an assumption
that the same protocol transfers or an authorization to train.

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
