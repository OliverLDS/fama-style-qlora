# Matched Gemini 500: inputs complete, qualification paused at the free-tier limit

The user authorized completing the Gemini-only counterpart of the mixed 500
release before further training. The exact same 500 human targets and order are
retained: 125 per author group across 48 distinct PDFs (53 stored paper IDs).
The [later lineage audit](GPT6_TRACK_0500_PREPARATION_V1.md) resolves five Fama
aliases; earlier hash-bound counts remain historical. Neither the mixed datasets nor
their experiments are overwritten. No new authors, PDFs, manual GPT-6 batches,
detector scans or GPU jobs are needed for this data-completion step.

## Training-data diagnosis

We previously found OCR/order/notation errors, attached headings, non-narrative
records and meaning shifts. Their versioned repairs/replacements precede the
frozen mixed release. The [500 release audit](STAGE_0500_RELEASE_V1.md) documents
27 further child target repairs, 15 repaired new inputs and three replaced
non-prose records; the earlier 200-pair salvage also retains its own repairs.
No claim is made that every paragraph was independently visually verified.

The student model's remaining errors and mixed Pangram outcomes do not by
themselves establish defective training pairs. The known five-years versus
t−1-through-t−6 wording discrepancy is in development, not a new training-target
diagnosis. Existing teacher drafts can still omit definitions or conditions;
those require repair before admission rather than wholesale rejection or an
assumption of perfect quality.

The later [GPT-6 response audit](GPT6_TRACK_0500_QUALITY_V1.md) identified five
inherited target issues in the shared mixed-500 cohort: P250, P264, P291, P458,
and P484. They concern a contradictory statistic, a page boundary, publisher
boilerplate, and attached headings. Resolve them consistently in both teacher
tracks before final release; the frozen checkpoints and mixed release are unchanged.

## Executed progress

- **214 accepted Gemini inputs reused unchanged** from the mixed release.
- **Eight historical Gemini trial drafts recovered**, all matching these targets.
  They require qualification; their original compression/alignment notes remain.
- **278 missing inputs generated**, through the confirmed free Gemini project.
- All 500 inputs now exist. **240** of the 286 new/recovered inputs have received
  separate meaning-only reviews; **46** still await review.
- All **240** downloaded reviews have now completed Codex adjudication. The
  latest offline pass qualifies the previously pending 96 inputs, with 28 local
  patches across 16 of them. Together with seven earlier repaired inputs, 23
  reviewed inputs have minimal repairs. Combined with 214 prior accepted reuses,
  **454** inputs have completed their input-qualification path; **46** still need
  separate review. The five inherited target follow-ups remain separate release
  requirements, so this is not a claim that 454 pairs are ready for training.

The teacher is `gemini-3-flash-preview`. New generation uses bounded low thinking
and the existing full-paragraph neutral-rewrite specification. Historical sources
retain their actual prompt/settings provenance; the preview name is not an
immutable weight revision. A single malformed reviewer ID was repaired in a
derived record, retaining the provider's original response and an explicit audit.
Full-paragraph target-wording restoration is not the repair policy.

A temporary HTTP 503 was resolved after preserved failures and bounded retry.
Later, the fourth review request returned the free quota/rate-limit category.
**API calls stopped**, with no paid or alternate-model fallback. The saved
evidence does not distinguish every rate-limit dimension or promise a reset time.
On the user's requested retry, review jobs 4 and 5 succeeded, returning 96 valid
records. Job 6 again returned the free quota/rate-limit category, and calls stopped.
This establishes some renewed capacity, not the remaining allowance or a daily
quota reset. The same configured free-tier project and model were used throughout.
The next requested retry of job 6 again returned the quota/rate-limit category;
only one API attempt was made in that retry. Offline adjudication continued using
the already downloaded results. It restored omitted PCA methodology, quantitative
comparisons, distributional changes, decision-control distinctions and other
specific meaning details without regenerating paragraphs. Reviewer suggestions
were checked rather than applied blindly, including a grammatically invalid patch.
All generation results, completed reviews and repairs are retained. No final
Gemini-only training-ready release has been published from partial checks.

## Resume and release

When free capacity is available, preserve the failed sixth review response and
retry only that last job (46 pairs). The versioned response index identifies the
successful retry for job 4; its older failure remains untouched. Adjudicate proposed changes,
including any remaining omissions and numeric/citation differences. Carry forward
the existing 240 qualified decisions and merge all 286 decisions before release.
Do not regenerate completed
inputs or request manual ChatGPT uploads.

After qualification, freeze nested Gemini-only 200/500 cohorts under
`data/data_scaling_v1/gemini_track_v1/`. The first 200 rows will be an exact byte
prefix of 500, balanced at 50 and 125 per group, with human targets identical to
the matched mixed cohort. These are **planned output paths**, not completed
training-ready exports. The current mixed release remains usable and unchanged.

Private checkpoint: `research_runs/data_scaling_v1/gemini_track_0500_v1/`.
The original `state.json` and checkpoint remain historical. Current progress and
response selection are in `resume_20261105_v2/state.json` and the earlier
`resume_20261105_v1/review_response_index.json`. The v2 manifest binds the 240
qualified decisions, the latest failure, previous checkpoint, and review evidence.
`drafts.private.jsonl` has all inputs. The original 144 decisions are preserved as
an unchanged prefix of `resume_20261105_v2/qualified_completed_reviews_v2.private.jsonl`.
Current public aggregate: `gemini_track_0500_progress_v3.json`; v1/v2 are preserved.

Runner: `scripts/data/complete_gemini_track_500.py`. CPU-only final auditor:
`scripts/data/audit_gemini_track_500.py`. Two new lineage/reuse tests and ten
existing coverage/patch tests pass. No training logic changed and no new
fine-tuning was launched.

Run `python3 scripts/data/audit_gemini_track_checkpoint.py` to check the partial
checkpoint, including patch replay, target/source hashes, unchanged prior decisions,
qualification counts and coverage of all 500 inputs. Four current checkpoint and
matched-cohort CPU tests pass. No provider calls are made by this auditor.
