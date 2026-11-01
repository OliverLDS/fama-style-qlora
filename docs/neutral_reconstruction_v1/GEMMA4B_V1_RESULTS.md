# Gemma 4B version 1: output-format failure and recovered comparison

The notebook title is `reconstruction_4b`; the actual URL slug is
`oliverlds/reconstruction-4b`. Version 1 completed all six first generations on
one Tesla T4 and saved them. It then failed the final output-contract check.
This was not a Secret, model-access, dependency, GPU-memory or context-budget
failure. Inference including loading took 199 seconds, with peak allocated GPU
memory of 5,523,325,440 bytes. The model revision was
`093f9f388b31de276ce2de164bdc2081324b9767`.

## Cause and local correction

All six responses wrapped their JSON object in a Markdown code fence. The
original strict JSON parser rejected every response, yielding zero accepted
generations and a failed status after all six jobs. Two reconstruction responses
also omitted both `status` and `reason`, independent of the wrapping issue.

The parser now accepts exactly one complete JSON Markdown fence as transport
normalization. It still rejects surrounding commentary, multiple JSON objects,
wrong identities and missing fields. No field values are inferred. Tests cover
these distinctions and the notebook has been regenerated with the updated parser.

Offline recovery of the existing outputs produces:

| Measure | Count |
| --- | ---: |
| GPU generations completed and saved | 6/6 |
| JSON objects recovered by fence-only normalization | 6/6 |
| Responses meeting the full schema after normalization | 4/6 |
| Responses still missing required fields | 2/6 |
| Output truncations | 0/6 |
| Additional model calls or GPU runs | 0 |

All six paragraph strings can be reviewed as saved text. The two incomplete
objects remain schema-nonconforming; they are not silently relabelled as valid
responses. The original failed state, parser results, raw strings and logs are
preserved. The local notebook correction is not reported as an executed rerun.
There is no need to spend another GPU run merely to recover these paragraphs.

## Substantive findings

Automated, unblinded review against the frozen human references found material
claim issues in all three brief-only reconstructions and in one direct rewrite.
These small, failure-selected development cases do not estimate a general error
rate. The classifications are provisional judgments, not independent semantic gold.

| Case | Direct rewrite | Brief-only reconstruction |
| --- | --- | --- |
| Liquidity phrase usage | Search actor and dates retained; local recency/word-choice shifts | Search attributed to newspapers instead of the narrator; unnecessary concluding search-pattern commentary |
| Commentators and contagion | Main content present; confidence, frequency and authority qualifiers shift | Contagion-awareness claim omitted; confidence qualifier remains weakened despite repaired brief |
| Sentiment and returns | Near-definitional relationship becomes an almost-invariable tendency | Lottery analogy and explicit two-component count omitted; concluding identification of prediction components is unsupported |

Fluency and successful model execution therefore do not show adequate claim
preservation. The repaired confidence binding did not prevent weakening in this
Gemma output. Its media-authority barrier was retained. The model also struggled
with actor binding and coverage of the longer briefs. This supplies precisely the
student-model evidence that the GPT-6 protocol checks could not establish.

Recommendation: stop treating the current detailed-brief format as ready for
Gemma-based training-data construction. Use these saved outputs to localize actor,
coverage and qualifier failures. A next experiment, if authorized, should test a
simpler claim representation and prose-only response contract against the same
baseline rather than repeat the same GPU run for formatting. Preserve the detailed
briefs and current results. No new prompts, selection or GPU job have been submitted
as a result of this diagnosis. No Fama-specific style improvement, human-authorship
or detector result is established.

Private artifacts are in
`research_runs/neutral_reconstruction_v1/gemma4b_kaggle_v1/` and
`research_runs/neutral_reconstruction_v1/gemma4b_review_v1/`.
The latter binds each assessment to the exact paragraph hash and pins all downloaded
files and the frozen source packet. The downloaded runtime matches that packet's
original runtime hash; the repaired local parser is recorded separately.
Public aggregate counts are in [gemma4b_v1_results.json](gemma4b_v1_results.json).
