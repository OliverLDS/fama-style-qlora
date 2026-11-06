# Preserve GPT-6 data; automate a separate free-tier teacher trial

User agreement: retain the GPT-6-generated pairs and continue the intended
**200 → 500 → 1,000 → 1,500 → 2,000** accepted-pair learning curve. The eight
GPT-6-derived inputs all received human-written labels from Pangram. This makes
AI-to-human detector conversion unmeasurable on that panel; it does not invalidate
the pairs or demonstrate that every GPT-6 paragraph defeats Pangram. Subscription
access and model capability are useful, but each pair still needs alignment and
source-quality checks.

## Binding cost policy

**All inferencer use in this project is free-tier only.** No paid inference,
paid fallback, paid-model substitution, automatic account upgrade or reliance on
expiring promotional credits as a continuing free service. On quota exhaustion,
stop and preserve progress for resumption. A configured key or listed model does
not establish a project's billing tier. Credentials remain outside the repository
and their values must never be printed. Provider failures must not log request
URLs containing API keys.

The user explicitly confirmed that the configured Gemini project is Free tier.
The trial uses `inferencer::query_gemini()` with the explicitly selected
`gemini-3-flash-preview`, an official Google endpoint and no fallback. The
package's old hard-coded free-model list is not authoritative pricing evidence.
Any future model/provider change needs a current free-tier check.

## Two data tracks

1. **Existing GPT-6 track.** Keep original responses, versioned repairs, frozen
   cohorts and results unchanged. Grow nested cohorts of unique human targets
   while preserving paper/author balance and provenance. A detector-human label
   is neither a pair-admission criterion nor a reason to reject a pair.
2. **Gemini trial.** Eight targets selected deterministically from the existing
   training cohort: two different papers per author group, eight papers total.
   Generate alternative neutral inputs using one ordinary rewrite instruction;
   retain the original human targets exactly. No held-out/reference text is used
   as a demonstration. Zero-shot for this first trial. All results remain draft
   pairs pending alignment review. Do not replace existing GPT-6 inputs or mix
   teacher changes into a claimed pure data-size effect.

The trial packet and responses are private under
`research_runs/gemini_teacher_trial_v1/`. The local CLI is
`scripts/data/gemini_teacher_trial.py`, with a single-request R adapter beside it.
It saves every successful response before proceeding, records model version,
usage and finish reason, and stops on failure. Successful rows resume without
another API call. Failed responses remain retained and require diagnosis or an
explicit new attempt version. No automatic retries or provider/model fallback.

The [executed eight-request trial](GEMINI_TEACHER_TRIAL_V1.md) completed all
responses. Draft alignment notes are retained; none is automatically admitted
to training and no existing GPT-6 row was replaced.

## Learning-curve evaluation without relying on Pangram

Freeze the development panel, instruction, decoding, checkpoints and assessment
procedures before comparing stages. Continue reporting the following separately:

| Dimension | Evidence | Interpretation limit |
| --- | --- | --- |
| Meaning | Added/lost claims; altered actors, numbers, conditions, uncertainty, negation and causality, with severity and concrete notes | Automated judgments remain provisional; minor shifts can be acceptable |
| Rewriting | Source overlap, longest copied span, exact copies and organization changes | Less copying alone is not better writing |
| Target alignment | Held-out reference NLL and reference overlap under identical scoring conditions | Neither establishes semantic fidelity or distinctive author style alone |
| Style and readability | Fixed style-feature profiles, paired organization/readability assessment, author-specific expression where supported | Generic academic fluency is not verified Fama-style learning; earlier author proxies are not validated benchmarks |
| Robustness | Results by paper and author group, with output failures and uncertainty retained | Small reused development panels support limited conclusions |
| Detector, optional | Exact matched input/output scores, provider version/date and native labels/shares | Use as a secondary outcome only when the input baseline provides headroom |

Use the user's meaning-versus-style trade-off: report severity and paired gains
rather than applying the historical zero-material-drift rule as an automatic
veto on every output. Major reversals and factual invention must remain visible;
their acceptance is not implied by a detector or style score. Do not collapse
these dimensions into an unvalidated single quality score. If a free LLM judge
is introduced, freeze its rubric and model, mask model conditions and record
dependence on the teacher; a separate prompt is not an independent model.

Keep the reserved tests out of model selection. Fixed epochs with increasing
dataset size form a practical data-plus-compute curve, not a pure data-size
causal experiment. Do not silently change prompts, teacher mixture, decoding or
training strength between stages. A changed design needs a labeled branch or a
matched control at an existing size. No later GPU training stage is automatically
authorized by this agreement.

This policy supersedes the historical promotion criteria where they imply a
mandatory Pangram gain or an absolute no-drift veto. Frozen experiment manifests
and previous outcomes remain historical records and are not rewritten.
