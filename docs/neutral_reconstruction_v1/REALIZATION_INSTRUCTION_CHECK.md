# Reconstruction realization instruction check v1

The preparation specification below is preserved; the check now has
[reviewed results](INSTRUCTION_CHECK_RESULTS.md). The preceding paired study found lower lexical overlap
for reconstruction in all eight cases, with added epistemic commentary in three.
This follow-up selects all three recorded commentary cases. It is a targeted
post-hoc development repair, not an independent test of generalization.

The generator receives exactly the same three content-brief objects as before.
One added instruction asks it to express modality and attribution faithfully,
without converting annotation guardrails into extra assertions about evidence,
empirical status or measurement. No uncertainty fields are removed. A separate
coordination header confirms that review and freezing are already complete;
it does not change the requested prose. The original prompt and full diff are
retained. References and first outputs remain private and outside generator inputs.

Request GPT-6 with High thinking effort, matching the reported earlier setting.
UI sampling controls are unavailable, so identical decoding and deterministic
causal attribution cannot be guaranteed. Account memory/retrieval is uncontrolled.
Retain the first response, including blocked cases, without outcome-driven retries.

After generation, compare each output with its first reconstruction and reviewed
reference. Assess added commentary, economic claims, actor binding, uncertainty,
attribution, conditions, quantities and causal direction separately from readability,
organization and word-overlap diagnostics. Removing commentary counts as useful
only if economic qualifications and claims remain intact. Report per-case copying
changes rather than selecting a new overlap threshold after seeing outputs.
Automated judgments remain imperfect; this check cannot establish Fama style,
human authorship or detector evasion. No training is authorized or submitted.

The private packet is
`research_runs/neutral_reconstruction_v1/realization_instruction_check_v1/`.
Supply only its current prompt, `inputs_01.jsonl` and output schema. Save the
three-record response as `responses/reconstruction_01.jsonl`. Session notes and
responses are mutable; all other packet files are hash-frozen.

Reproduce preparation with:

```sh
PYTHONPATH=src python3 -m fama_style.research.neutral_reconstruction_instruction \
  --packet research_runs/neutral_reconstruction_v1/stage2_package_v2 \
  --results research_runs/neutral_reconstruction_v1/stage2_review_v1/results \
  --output /tmp/reconstruction-instruction-check-reproduction
```

The builder refuses overwrite, verifies both input manifests and asserts the exact
three previously recorded failures. Source paragraphs, briefs and response texts
are not distributed with the public code or documentation.
