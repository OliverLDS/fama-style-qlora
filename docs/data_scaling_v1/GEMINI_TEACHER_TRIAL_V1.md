# Automated free-tier Gemini trial: eight responses saved

The user confirmed that the configured Gemini project is Free tier and that
**all inferencer use must remain free-tier only**. The local runner executed
eight requests through `inferencer::query_gemini()`, explicitly selecting
`gemini-3-flash-preview`, with no provider/model fallback. All eight completed
with finish reason `STOP`. No manual ChatGPT uploads, GPU training or detector
calls were involved.

The panel contains eight existing training targets, two per author group from
eight different papers. Selection was deterministic, before seeing generated
text or detector scores. No development or reserved-test paragraph was used.
The original human target remains unchanged in every draft pair; the alternative
Gemini source is stored separately. The frozen GPT-6-containing training cohort
still matches its recorded SHA256.

The responses report 14,955 total tokens, including thinking, and about 62 seconds
of total request time. Deliberate pacing between requests adds wall time. These
are observed trial measurements, not a guarantee of future quota or throughput.
The returned model-version string equals the preview model name and does not
pin immutable weights.

## Pair quality and preservation

The generation workflow works, but the initial prompt tends toward compression:
the median generated-to-human word ratio is 0.670. Provisional automated reading
identifies alignment notes in six responses, including omission of a statistical
definition and an order-of-magnitude comparison, lost conditions, and stronger
normative wording. These are concrete prompt/quality issues to address before
bulk adoption, not grounds for discarding the GPT-6 corpus.

All eight raw responses and draft pairs are retained, with zero rejected rows.
No pair is automatically admitted to training. Exact source/target-bound notes
are private; this is not independent human verification or a comparison proving
one teacher superior. No Pangram score exists for these Gemini sources.

Private run root: `research_runs/gemini_teacher_trial_v1/` contains the frozen
inputs/config, raw responses, usage/model metadata, draft pairs, alignment notes
and hashed manifest. Public aggregates are in `gemini_teacher_trial_v1.json`.

The CLI is `scripts/data/gemini_teacher_trial.py`, with its R adapter alongside.
`--check` performs CPU validation without calls. The generator skips already
completed responses and stops on provider failures, preserving evidence without
paid fallback or automatic retry. API errors are suppressed because the existing
inferencer wrapper can embed credentials in request URLs. No credential values
were written to the repository or printed.

Two focused CPU tests cover stable paper-balanced selection, unchanged inputs,
exclusion of held-out/unapproved rows, and absence of previous teacher prose in
generation prompts. R syntax and the frozen packet pass local checks.

The [teacher/evaluation policy](TEACHER_AND_EVALUATION_POLICY_V2.md) retains the
GPT-6 track and the staged learning curve. Before expanding the Gemini branch,
adjust its summarization tendency and reassess alignment. No later student
training stage is authorized by this trial.
