# Rhetorical oracle: Kaggle version 1

Source: `oliverlds/rhetorical-oracle`, explicitly downloaded version 1.
This is inference on the 21 development-validation examples with base Gemma 3
4B, not the QLoRA adapter. No training or reserved-test evaluation occurred.

## Execution and integrity

The run completed **63 unique generations**, 21 per condition, without empty or
cutoff outputs. The model revision, prompts, manifests and embedded source hashes
were verified; all saved metrics were reproduced locally from the downloaded
outputs and saved NLI scores, without another model call. The log has no traceback.

The progress field `completed_jobs=62` is a stale count recorded before the last
generation. The 63 saved generations, 63 metrics and 63 NLI assessments establish
completion. This logging detail does not indicate a missing example.

Raw logs, paragraphs and outputs remain private under
`research_runs/rhetorical_v1/kaggle_version_1/`.
Verification hashes are in `research_runs/rhetorical_v1/version_1_review/`.
Public aggregate evidence: [rhetorical_oracle_v1_results.json](rhetorical_oracle_v1_results.json).

## Comparison

| Measure | A: ordinary prompt | B: source structure | C: oracle target structure |
|---|---:|---:|---:|
| Independent NLI proxy passes | 13/21 | 13/21 | 15/21 |
| Combined oracle adherence | 50.1% | 50.4% | 51.5% |
| Exact copies / similarity ≥0.90 | 0 / 0 | 0 / 0 | 0 / 0 |
| Training-style distance gain; higher is better | −0.050 | −0.634 | −0.602 |
| Exact reference sentence-count matches | 4/21 | 5/21 | 7/21 |
| Mean generated sentence count | 3.14 | 2.86 | 2.90 |

The references average 4.29 sentences. The style-gain values are standardized
surface-feature distances, not percentages or author-identification scores.
Negative gain means movement farther from the training-reference feature profile.

### What changed with the oracle

C versus B, the primary comparison, improves combined adherence by **0.0114**,
below the predefined 0.05 gate: 6 examples improve, 5 worsen and 10 tie.
NLI switches from flagged to consistent on 4 examples and the opposite direction
on 2, yielding a net gain of 2. This small development sample does not establish
a reliable semantic improvement.

The NLI model passed all three equivalent controls and seven semantic mutation
controls. Its rates remain imperfect proxies, not verified claim-preservation
rates. None of the 63 assessments overflowed context. Protected surface markers
differ on 21 A outputs and 20 outputs in each of B/C; these are risk flags, not
adjudicated semantic errors.

No condition copied exactly or nearly, so this base-model experiment provides no
evidence that plans reduce copying relative to ordinary prompting. It also does
not resolve copying in the earlier QLoRA adapter, which was not loaded here.

### Realization behavior

The oracle helps exact sentence-count matching somewhat, but it is still achieved
on only 7 of 21 outputs. Mean heuristic words per sentence are 47.7 in A, 58.3
in B and 57.5 in C, versus 24.6 in validation references. Semicolons per 100
tokens are 0.264, 0.725 and 0.744 respectively, versus 0.067 in references.
Thus structural prompting coincides with excessive sentence density and semicolon
use. It does not reliably realize the requested reference structures.

These observations suggest a representation/realization problem worth isolating.
They do not prove that the three-level research direction cannot work: the
annotations are heuristic, B/C include repeated source evidence, and the generator
may respond differently to concise operational instructions.

## Decision and next check

**The predefined mapping gate did not pass.** C misses the adherence threshold,
the ≥90% semantic-proxy criterion, and the requirement to improve style over A.
Do not train a learned rhetorical mapper or another QLoRA adapter based on this run.

The next useful experiment is a small **realization-format ablation**: hold the
source, source-evidence information, decoding and desired structure fixed, then
compare the current JSON template with concise instructions controlling one
operation at a time. Start with sentence count and clause organization. Target
sentence counts remain oracle information; reference wording remains excluded.
Retain independent semantic checks so better adherence cannot compensate for
changed claims. No new notebook or GPU job was submitted during this review.

## Execution cost

Total wall time was **48.9 minutes**: approximately 14.6 minutes of Gemma generation,
32.8 minutes loading/running CPU NLI and 1.5 minutes of other setup. Peak GPU
allocation was approximately 4.78 GiB (4.96 GiB reserved). The NLI stage dominates
wall time; these are measured runtime figures, not monetary prices.
