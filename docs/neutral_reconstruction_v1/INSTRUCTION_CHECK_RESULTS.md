# Reconstruction instruction check: results

All three outputs were received, generated and reviewed. The user reports GPT-6
with High thinking effort. The actual file is `neutral_reconstruction_01.jsonl`;
it was imported in place without renaming or modifying its contents.

The revised instruction removed the three previously observed evidence disclaimers
in this small targeted check. Automated, unblinded comparison with the reviewed
human paragraphs identified no material economic claim errors. This is a useful
prompt repair, not independent proof of semantic fidelity or style learning.

| Case | Prior reconstruction overlap | New overlap | Original direct rewrite overlap | Remaining issue |
| --- | ---: | ---: | ---: | --- |
| Liquidity phrase usage | 0.367 | 0.437 | 0.711 | Awkward “narrator” framing |
| Commentators and contagion | 0.356 | 0.361 | 0.644 | Confidence and categorical scope weaken |
| Sentiment and returns | 0.428 | 0.444 | 0.695 | Repeated assumptions add length |

Overlap is the same word-sequence metric used in the earlier study. Copying rose
slightly in every case relative to the previous reconstruction, but stayed below
the corresponding original direct rewrite in all three. Longest shared word runs
were 9, 6 and 5 words. No new or missing numeric values were detected. Lower overlap
alone does not establish useful style or human authorship.

The outputs no longer assert an absence of liquidity measurement, that attributed
commentator explanations are unestablished conclusions, or that the hypothetical
return prediction is not an empirical finding. Attributions and hypothetical
conditions remain. The sentiment output now says returns accompany an increase
rather than that the increase raises returns, avoiding the earlier causal
strengthening while retaining the framework's prediction.

One output still has two local scope shifts. “Not confident” remains “less
confident”; the frozen brief already encoded that change. “Too poorly understood
… to be represented as an authoritative view” becomes a limitation on the model's
appearance, weakening the categorical barrier. These are recorded as local,
nonmaterial issues in this automated review, not silently accepted as exact
meaning equivalence. The current brief contains both categorical and softer cues
for the second relationship. A generation instruction cannot reliably resolve
contradictory or inaccurate semantic annotations.

Recommendation: keep the revised realization rule as the current development
candidate. Before expanding reconstruction annotation or training, make a
versioned, source-checked repair to the remaining confidence and media-authority
bindings; clarify how narrator roles should be realized without adding content.
Then test the corrected protocol on fresh training-side examples. Preserve the
first briefs and outputs, and keep this selected three-case repair separate from
any later generalization evaluation. No further generation packet or training job
has been prepared or submitted here.

The private review, raw response copy, paired metrics and SHA receipts are under
`research_runs/neutral_reconstruction_v1/realization_instruction_review_v1/`.
The packet's frozen files and previous results were verified unchanged. Session
notes record reported settings; conversation arrangement is not established.
UI decoding controls and account memory/retrieval remain uncontrolled. This review
was unblinded, automated and not independent or calibrated. It evaluates neither
Fama-specific style nor detector evasion and certifies no training-ready pairs.

Public aggregate counts are in [instruction_check_results.json](instruction_check_results.json).
The CPU importer is `fama_style.research.neutral_reconstruction_instruction_results`.
It requires exact coverage and response-hash-bound assessments, retains blocked
cases and refuses to overwrite a results directory.
