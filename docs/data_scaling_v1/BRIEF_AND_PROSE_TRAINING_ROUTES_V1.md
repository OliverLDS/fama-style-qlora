# Matched brief and prose training routes

The goal is fresh brief reconstruction for **all 500 GPT-6 training targets**,
not only the 175 selected by the initial contrast screen. Retain earlier direct
rewrites as alternative inputs and preserve every historical experiment. A new
input version is not another independent human example.

## Data completion

`scripts/data/prepare_gpt6_full_reconstruction.py` prepares a versioned packet at
`research_runs/data_scaling_v1/gpt6_track_0500_v1/full_reconstruction_v1/`.
It reuses the existing 175 brief/reconstruction responses, of which 173 currently
support usable candidates. It queues 320 other targets in eleven Stage 1 jobs.
Five missing targets await source repair; two previously collected targets also
retain source concerns. Therefore 500 fresh candidates are the completion goal,
not a claim about the current release. All 500 identities have a coverage entry.

Review returned briefs before creating Stage 2 packets. Stage 2 receives only
briefs, IDs and target hashes. It must not receive original wording or earlier
rewrites. Reuse the existing prompts: remove authorial framing and sentence
templates while keeping the main economic account. Accept modest emphasis shifts;
repair major factual errors without restoring the author's distinctive wording.
Do not impose an arbitrary percentage of changed words. Every returned item gets
a reuse/repair/hold decision rather than selecting a tiny successful subset.

Target repairs must be versioned and reconciled with Gemini before claiming
matched human targets in final teacher releases. A replacement for publisher
boilerplate changes cohort membership and must be recorded as such. Existing
target hashes and releases remain intact.

The [320 missing-target briefs have returned and been assessed](GPT6_FULL_BRIEFS_V1.md).
All proceed to eleven brief-only reconstruction jobs, with two limited local
clarifications and raw responses preserved. This brings collected briefs to 495;
the five missing targets and earlier two source concerns remain explicit.
The [Stage 2 output assessment](GPT6_FULL_RECONSTRUCTION_OUTPUTS_V1.md) now provides
matched candidates for 493 targets in both routes, including the 173 prior usable
reconstructions. Five derived briefs are clarified using issues discovered during
realization; frozen original annotation packets remain unchanged. The full release
still awaits seven source/target follow-ups, and training remains disabled.

## Route A: brief to human prose

Training: reviewed content brief → human paragraph. Serialize the brief's task,
key points and anchors consistently; do not include review notes, author wording
or metadata that reveals the answer. An explicit author/style condition may be
added identically across both routes in a separately specified training experiment.

Inference: AI input → GPT-6 content brief → fine-tuned Gemma → rewritten prose.
The inference brief comes only from the actual input. Training briefs derived
from human references may differ from briefs extracted from AI prose: measure
that distribution gap on training-side and development examples before choosing
this route. Keep the original input for downstream meaning assessment. A brief
can lose information; do not assume this route preserves meaning better.

The current 173 usable brief/target records are saved as a private preview, with
training disabled. GPT-6 at inference adds a dependency, latency and possibly
manual work. A ChatGPT subscription does not authorize paid API calls; automation
requires a separately available authorized interface. No such calls are made here.

## Route B: AI prose to human prose

Training: fresh brief reconstruction → the same human paragraph.
Inference: AI input → fine-tuned Gemma → rewritten prose.
Earlier direct GPT-6 rewrites that pass assessment are alternative robustness
inputs. Keep their provenance and pair them with the same target identity;
do not inflate dataset size or leak sibling versions across splits.

## Controlled comparison and gates

The existing research student is **Gemma 3 4B**, not a newly selected Gemma 4
model. Neither route is trained by this data-preparation change. Keep the
historical root Gemma 3 1B CLI unchanged.

Compare routes on identical development content, author conditions, student
checkpoint, decoding settings and an explicitly matched training budget. Report
updates and tokens, because brief and prose input lengths differ. Keep whole-paper
splits and the reserved test isolated. Do not regenerate held-out material from
reference paragraphs as deployable input. Oracle construction is labeled separately.

Assess style transformation, readability, input copying and material claim drift
separately. Some drift is acceptable; a reversal of the economic explanation is
not equivalent to a harmless change in emphasis. Detector scores are sparse
auxiliary evidence, with both actual inputs and outputs scanned where useful.
Gemini-generated held-out inputs can test cross-teacher robustness and detector
headroom; low detectability of GPT-6 inputs is not evidence of student improvement.

After full data qualification, authorize a small matched pilot before expanding
the learning curves. No GPU jobs, teacher calls or detector scans are submitted
by this preparation step.
