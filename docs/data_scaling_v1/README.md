# Data quality repair and staged learning curve v1

Latest results: the [purpose-only GPT-6 500-pair run](GPT6_PURPOSE_500_RESULTS_V1.md)
completed successfully, with lower loss but copying and generation failures.
The [instruction diagnostic](PURPOSE_500_PROMPT_INPUT_RESULTS_V1.md) reduced
copying through explicit re-expression; the subsequent
[Pangram checkpoint](PURPOSE_500_PANGRAM_RESULTS_V1.md) found no new human-label
conversions among the six scored AI-related inputs. No new training is underway.
The proposed next comparison varies the saved adapter's contribution at
inference, using existing data. Earlier stage descriptions below are historical
snapshots, not current GPU or annotation authorizations.

Agreed 3 November 2026: **200 → 500 → 1,000 → 1,500 → 2,000 accepted training
pairs**, measuring whether reduced detector scores accompany preserved meaning
and readability. Two thousand is a research target, not a universal threshold
below which fine-tuning is meaningless or a guarantee of success above it.
Updated agreement: [preserve the GPT-6 track and use multidimensional evaluation](TEACHER_AND_EVALUATION_POLICY_V2.md).
Pangram is optional when inputs already receive human-written labels. All
inferencer calls are free-tier only; a separate eight-paragraph Gemini trial
automates pair preparation without replacing existing data or starting training.
The [500-pair release](STAGE_0500_RELEASE_V1.md) is now frozen: 125 pairs per
group across 53 papers, preserving the original 200 as an exact byte prefix.
Existing pairs were reused first; missing inputs were generated on the confirmed
free Gemini tier. Hash-bound minimal repairs and exclusions are retained in the
private audit. CPU checks and an identical rebuild passed. The
[capacity audit](STAGE_0500_EXISTING_MATERIAL_PLAN_V1.md) remains
the historical plan.
The [two-teacher plan](TWO_TEACHER_TRACKS_V1.md) now defines separate automated
Gemini and manual GPT-6 curves at the same five sizes, sharing human targets and
evaluation controls. Existing mixed-provenance releases remain unchanged; they
are not relabeled as pure teacher cohorts.
The [mixed-500 pilot](STAGE_0500_RESULTS_V1.md) completed one epoch, 250 updates
and 48 matched outputs. Automated review finds fewer material errors but more
copying. The [full 24-report Pangram panel](STAGE_0500_PANGRAM_FULL_V1.md) is
imported: one case improves, four deteriorate and three tie versus actual inputs.
Only one of seven AI-related inputs becomes entirely human-labeled. All eight
[reference controls](STAGE_0500_REFERENCE_CONTROLS_V1.md) are now 100% Human Written,
repeating earlier published-human findings. Reuse that calibration and avoid
routine repeat scans. No new training has started.

The newly authorized [Gemini-only 500 counterpart](GEMINI_TRACK_0500_V1.md) now
has all inputs, reusing 214 accepted sources and eight historical drafts while
generating only 278 missing counterparts. Qualification is paused at the free
quota/rate limit: 144 new/recovered sources adjudicated, 142 pending. No partial
cohort is labeled training-ready and no training has started.
The [first 200-pair pilot](STAGE_0200_RESULTS_V1.md) completed one epoch, 100
updates and sixteen matched development comparisons. Automated review finds
better claim retention and substantially more input copying. The
[simpler-prompt comparison](SIMPLE_PROMPT_RESULTS_V1.md) completed all 32 outputs:
less adapter copying with more meaning issues identified. Both prompt packets
have now been [scanned and imported](PANGRAM_PROMPT_COMPARISON_V1.md): the strict
adapter receives 6/8 entirely-human labels, while the simpler adapter and both
base conditions receive 0/8. Later training stages remain prospective
and unauthorized.

## Correcting the interpretation of the last audit

The 41/249 figure counts rows passing a conservative provisional gate. It does
**not** establish that 208 pairs are invalid or that 41 are independently verified.
The gate combined text sufficiency with an aggregate review flag. Of the 208 held
rows, 153 have no additional flagged claim/relation alignment, mechanical,
numeric-scope or absent-rewrite issue; 107 of those are training-side. That makes
them a review queue, not 153 automatically recoverable pairs. Real extraction
errors also occur among these flag-only cases.

The current diagnosis distinguishes four problems:

- **Our extraction code:** sorting by exact vertical coordinates could reverse
  adjacent OCR fragments on the same printed line. A selected Fama paragraph was
  demonstrably scrambled before GPT-6 received it. A PDF page check confirms the
  correct left-to-right order.
- **Source transcription:** a second checked paragraph has a comma misread as a
  period. Its versioned recovery target corrects that punctuation without adding
  words or changing economic claims.
- **Overly broad holds:** a third checked paragraph was held solely because a
  technical term lacked a definition, although the term and its role are retained
  in both texts. A missing textbook definition is not automatically missing
  information for meaning-preserving rewriting.
- **Actual semantic or structural issues:** uncertain claims, corrupted symbols,
  unsupported relations and wrong structural bindings remain genuine checks to
  resolve. Neither a blanket rejection nor a blanket acceptance is justified.

The three PDF checks are a purposive diagnosis, not a representative estimate of
GPT-6's quality. Scientific truth checking and replication of reported estimates
are outside this rewriting task. A citation or table need not be supplied merely
to preserve the paragraph's explicit attribution; a missing actor, quantity,
condition or necessary relation can still be a real blocker.

## Implemented corrections

The curation module now offers an **explicit new-version option**:
`--repair-ocr-fragment-order`. It orders nearby, nonoverlapping fragments on one
baseline by horizontal position, retaining every character and original line ID.
It avoids large column gaps and separate regions. Historical defaults remain
unchanged, and the option cannot be used under an older protocol name.

New private snapshots are `data/author_style_corpus_v3_existing/` for the existing
39-paper collection and `data/author_style_corpus_v3/` for the 28 additions. They
record layout operations and exact line accounting, retaining all earlier
snapshots. The updated pools contain **1,017 and 574 annotation candidates**,
respectively. These are not new valid pairs. Candidate counts can fall when the
new supplementary-glyph check correctly holds earlier corrupted text.

There are 652 recorded layout-order events in the existing collection and 2,306
in the additions. These are geometric operations, not counts of visually verified
paragraph repairs. Residual recognition errors, punctuation and native-text word
fusions still require checks; the algorithm does not guess their replacements.

The prepared twelve-case/two-batch manual QA request is **withdrawn**. Its files
remain historical preparation artifacts, not work for the user to complete.
Instead, the [executed salvage](PAIR_SALVAGE_V1.md) assesses all 249 collected
pairs, checks targeted PDF crops and records child-version repairs. It yields
230 examples: 171 training and 59 development, with 20 original units excluded.
Thirteen absent neutral inputs were supplied by Codex from PDF-checked targets
and are explicitly distinguished from the user's GPT-6 responses.

## Common admission standard before scaling

Apply the same standard to new responses and older pair releases. Earlier
Fama/teacher results remain valid historical experiments; their rows are neither
automatically discarded nor grandfathered into a new accepted cohort.

For direct SFT, require a reliable target, meaning-preserving input, explicit
resolution of material uncertainty, and complete source/version provenance.
Separate source verification, pair alignment and structural-label eligibility.
An optional taxonomy or transformation error must not alone invalidate an
otherwise verified text pair. Conversely, a structural plan cannot justify a
bad input/target pair. Retain unresolved numeric and causal changes.

The aggregate `needs_review` flag is a triage signal. Resolve its specific cause;
do not simply erase it or treat it as proof that the text pair is bad. Check
claims in both directions, retaining scope, modality, actors, quantities and
negation. Report review limitations; automated acceptance is not semantic proof.
Keep corrections in child versions with links to originals and old annotations.

The first balanced 200-pair cohort is now frozen at
`data/data_scaling_v1/stage_0200_salvage_v1/training.private.jsonl`: 50 pairs per
group across 48 papers. Forty older training pairs were reassessed; one repeated
legacy finding was excluded. Ten additional approved unique training parents
remain available, and an extra paragraph child is retained outside cohort-size
accounting. No earlier dataset was overwritten. Do not redo the 54 batches.

## Learning-curve design

| Stage | Unique training targets | Default target per group |
| --- | ---: | ---: |
| 1 | 200 | 50 |
| 2 | 500 | 125 |
| 3 | 1,000 | 250 |
| 4 | 1,500 | 375 |
| 5 | 2,000 | 500 |

These are nested cohorts of approved pairs, not counts of all JSONL rows.
Multiple paraphrases of one target do not increase the unique-target count.
Deduplicate drafts and overlapping passages; retain all versions of a work in
one role. Existing development and reserved examples never count toward N.
Balance groups and spread selection across papers before inspecting model outputs.
If a group cannot meet its quota, pause; do not silently change composition or
lower quality standards. Any alternate mixture needs its own declared protocol.

The CPU implementation `src/fama_style/research/data_scaling.py` constructs nested,
paper-interleaved fixed-hash cohorts only from rows explicitly approved for
training. It rejects provisional, development, reserved, duplicate/conflicting
target versions and insufficient groups. Its frozen-prefix option preserves the
earlier cohort exactly when new pairs arrive; reranking an enlarged pool must not
silently replace the first 200 examples. The first 200-pair data cohort is built;
later stage cohorts are not.

Use **direct SFT first**, the existing Gemma 4B study as the model family, one
frozen base revision, the same instruction format, quantization/LoRA settings,
length policy, epochs and decoding. Every size starts from the same base;
continuing the previous stage's adapter would mix dataset size with training
history. Keep the root Gemma 1B pipeline unchanged. Structural guidance, DPO,
author conditioning and the eventual Fama-specific comparisons remain separate
experiments rather than extra variables in this curve.

With fixed epochs, larger sets also receive more updates and tokens. Report
this honestly as a **practical data-plus-compute curve**, not a pure data-only
causal comparison. Record updates, target tokens, truncations and GPU time. A
fixed-token comparison can be a later sensitivity check if improvement appears.
At similar lengths, all five stages sum to 5,200 example passes per epoch,
roughly 2.6 times the largest stage alone; this is arithmetic, not a GPU-time quote.
Freeze actual training settings, tokenizer lengths, cost cap and stop rule before
requesting the next GPU run. No new GPU jobs are submitted by this plan.

## Fixed evaluation and sparse detector checkpoints

The prepared development panel has sixteen examples: two papers per group and
two examples per paper, selected by fixed hash from the adjudicated development
release. Eight are frozen for detector evaluation, one paper/two examples per
group, before generation. It is documented in the private pilot config and
lineage. Do not repurpose the sealed 45/20
reserved examples or present earlier reused development data as fresh validation.

Compare ordinary base output with each size on identical inputs and decoding;
retain raw GPT-6 inputs, human references and unchanged-input copying as controls.
Use the existing one-minor/no-material rubric and report unresolved judgments,
readability, exact/long-span copying, style descriptors and all failed outputs.
Do not use detector improvement to compensate for material loss of meaning.
Detector success is not evidence of Fama-specific style learning.

Detectors are sparse measurements, not training rewards or prompt-tuning loops.
Record native labels and scales, visible provider version/date and complete text
hashes. Freeze outputs before scanning; do not regenerate, trim or select examples
based on detector feedback. A low-scoring baseline may leave no informative
reduction to measure; report that floor instead of searching for favorable cases.
Report paired effects and paper-level uncertainty; a tiny panel cannot establish
general detector performance or require a monotonic curve.

Pangram remains the practical option under the user's reported ordinary daily
allowance. Retain the PDF-import workflow; no manual field entry or extra accounts
are assumed. Under the user's rule, compute `ceil(words/100)` per text and sum
before scanning. Eight 150-word outputs would cost about sixteen reported credits
per checkpoint; actual text lengths, current balance and rules must be checked.
Eight inputs, eight references and six model conditions would be 64 scans, about
9,600 words at 150 words/text. This may exceed the remaining GPTZero budget, so
do not assume the whole curve fits one allowance. Reuse unchanged results only
where provider/version/date comparability permits it; otherwise log the mismatch.

After each stage, check quality first. Pause when material drift, copying or
readability worsens, or when the detector endpoint is uninformative. Record the
failed result and missing detector measurements rather than hiding them. Proceed
to the next size only after reviewing the matched evidence and obtaining explicit
authorization for the next GPU job. No training or detector scans are run now.

## Capacity of the current collection

The new screened source pools total 1,591 candidates. Even adding the older 247
Fama training pairs gives at most 1,838 rows **before** duplicate, split and quality
exclusions. This is not an accepted training count. It excludes additional prose
that might be recovered from existing PDFs and is not a limit on the papers' total
content. Under the default balanced design, Holmström and Fama in particular need
more usable coverage before a 500-per-group final stage is possible.

First measure recovery yield and fix extraction/QA. Then recover more eligible
prose from existing works or acquire targeted additional papers. We may need more
than twenty works per group, but should not extrapolate the overly conservative
41/249 pass rate as a true usable-data yield or multiply it into a collection quota.
Reach 200, evaluate, and expand toward 500 only with that evidence.

Protocol: [protocol.json](protocol.json). Diagnosis aggregates:
[quality_diagnosis_summary.json](quality_diagnosis_summary.json). All data, PDFs,
source images, responses and per-record repair evidence remain private.
