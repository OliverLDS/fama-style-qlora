# Two teacher tracks with matched human targets

Agreed direction: maintain an automated Gemini 3 Flash Preview track and a
manual GPT-6 track. Each follows 200 → 500 → 1,000 → 1,500 → 2,000 accepted
pairs. This document defines the comparison; it does not launch generation,
request manual uploads, or authorize training.

Sequencing update: first run and review the [mixed-500 pilot](STAGE_0500_PILOT.md),
including a matched Gemini-input detector check. Resolve material pipeline/data
problems before completing the separate tracks. Complete Gemini automatically
first; the manual GPT-6 track can follow later. Both retain the full five-stage
target path. The matched comparison need not advance both teachers simultaneously.

The [mixed-500 run has completed](STAGE_0500_RESULTS_V1.md). Its automated review
finds better retention with more copying and some remaining economic errors.
The [complete 24-report Pangram panel](STAGE_0500_PANGRAM_FULL_V1.md) is imported:
one case improves, four deteriorate and three tie against actual inputs.
The eight human-reference controls are also imported, all 100% Human Written.
Separate-track completion was initially deferred while interpreting those findings;
reuse the established calibration rather than requesting routine reference scans.

The user subsequently authorized [Gemini-only completion](GEMINI_TRACK_0500_V1.md)
on the existing 500 targets. All inputs are present (214 accepted reuses, eight
recovered drafts, 278 new generations); 142 separate reviews remain after the
free quota/rate-limit response. The mixed release remains unchanged. No training
or new manual GPT-6 batch is part of this completion step.

## Evidence for free automation

The saved 500-expansion logs contain 19 provider attempts: 17 completed responses,
one incomplete response and one generic provider failure. Complete records from
the incomplete response were recovered, and subsequent requests completed. No
saved error was positively classified as quota exhaustion. All requests used
`gemini-3-flash-preview` on the user-confirmed free-tier project; 214 retained
inputs in the mixed 500 release came from that model. This establishes practical
feasibility for that run, not long-term availability or a guaranteed daily yield.

Google lists standard input/output for this model as free, but active limits are
project-specific and capacity is not guaranteed. Check the project's active
limits when scheduling further work. Use ordinary bounded requests containing
multiple paragraphs; Google's separate Batch API is not the free-tier mechanism.
Persist each complete response, resume from saved IDs, distinguish response-budget
failure from provider failure and HTTP 429, and pause when free quota is exhausted.
No paid fallback or automatic teacher/model substitution is permitted.

Sources checked for this plan: [pricing](https://aistudio.google.com/docs/pricing#gemini-3-flash-preview)
and [rate limits](https://aistudio.google.com/docs/rate-limits).

## Comparison design

Use one versioned inventory of human targets and two teacher-specific neutral
inputs per target. Each direction remains generated neutral prose → human author
prose. Keep the same four author groups and target counts in each teacher track:

| Stage | Pairs per track | Pairs per author group per track |
| --- | ---: | ---: |
| 200 | 200 | 50 |
| 500 | 500 | 125 |
| 1,000 | 1,000 | 250 |
| 1,500 | 1,500 | 375 |
| 2,000 | 2,000 | 500 |

Stages are nested within each track. Generate only the additions: 200, then
300, then 500, 500 and 500. At the endpoint this is 4,000 teacher-specific pairs
sharing 2,000 human targets, not five independently generated datasets per teacher.
Do not duplicate human targets within a stage to meet a count.

Reuse every eligible existing annotation before requesting new work. Retain
model identity, observed/reported effort, prompt version, original response,
patch provenance and final source hash. Bounded meaning repairs are allowed but
must be recorded. A wholly Codex-written replacement is not a GPT-6-generated
input. Unknown historical teacher provenance is not silently assigned to either
track. One teacher's source text is never silently substituted for the other's.

Use a common generation specification, acceptance criteria and human-target
repairs. For the matched comparison, a target enters the shared accepted order
only once both teacher inputs qualify. Keep valid unpaired inputs in a separate
queue; do not discard them because the counterpart is absent. Preserve the paper
split, and select additions across papers before exhausting a few papers. Only
expand authors or paper coverage when the qualified target inventory requires it.

Evaluate each teacher-trained adapter on the same held-out content, including
both Gemini-input and GPT-6-input conditions. Report meaning, readability,
copying and style/reference alignment separately. Scan input as well as output
when using a detector; low input detectability limits that endpoint's headroom.
Do not select training pairs merely for high detector scores.

For a later learning curve, hold the base model, inference instruction, decoding,
adapter recipe and evaluation panel constant. Start each independent size stage
from the same base rather than continuing a smaller-stage adapter. Record teacher
versions, token exposure, update counts and training cost; sample size alone does
not hold computation constant. Any teacher version change needs a separately
identified cohort rather than being hidden inside a size increase.

## Existing releases and immediate next work

The frozen 200 and 500 releases remain historical **mixed-provenance** cohorts.
The 500 contains 214 Gemini inputs, one newly recovered historical GPT-6 input,
plus 285 reused rows with their original GPT-6/Codex/legacy provenance. It is
neither a pure Gemini-500 nor a pure GPT-6-500 cohort. Do not overwrite, relabel
or invalidate those releases or their experiments.

After the mixed-500 review, inventory existing inputs by target and teacher, including recorded repairs;
freeze the shared target order; then identify precisely which counterparts are
missing at 200 and 500. Fill Gemini gaps automatically within free quota and
prepare only necessary GPT-6 manual batches, with intended use and coverage
explicit before asking the user to upload anything. No further manual work is
requested by this plan. Qualify the paired teacher cohorts before moving to
larger stages or submitting training.
