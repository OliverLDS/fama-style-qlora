# Source-context restoration: version 1 results

The run completed successfully, but restoring full source context is only a partial fix. It resolved the missing economic subjects in the six outputs for the two ambiguous groups. It also allowed other passage claims to leak into the assignment, and did not produce substantial author-style improvement. Do not use these outputs as training labels.

Kaggle: `oliverlds/group-context-restoration`, owner-reported version 1. All 18 calls finished in **189.6 seconds (3.2 minutes)**. All 18 were complete single sentences, with no empty/cutoff samples. All nine local-only controls reproduced the prior v2 output hashes exactly.

## Verification

Downloaded the logs and output artifacts read-only. Verified the prepared pack hash, runtime source hashes, saved job prompts/seeds/budgets, artifact manifest, model revisions, decoding and completion status. Recomputed all metrics and `evaluation.json` from raw generations and saved full verifier verdicts; replay matched within floating-point tolerance. No model calls or GPU submissions were made during this review.

The CLI's session-output endpoint does not guarantee immutable historical version selection; version 1 is the owner's version designation. Pack/runtime/job identity and exact control reproduction provide the execution identity checks.

## What changed

| Diagnostic | Local-only | Read-only source context |
| --- | ---: | ---: |
| Complete, single-sentence outputs | 9/9 | 9/9 |
| Substantial style improvement, exploratory assessment | 0/9 | 0/9 |
| Minor / moderate / high drift, exploratory assessment | 0 / 3 / 6 | 1 / 2 / 6 |
| Input tokens | 2,193 | 4,299 |
| Output tokens | 420 | 370 |
| Generation seconds, excluding loading/NLI | 38.8 | 38.9 |

These drift grades concern individual assigned groups, not assembled paragraphs, so they should not be compared directly with previous paragraph-level grade distributions. Grades are an unblinded assistant assessment, not calibrated semantic or author-style accuracy. Source overlap increased from 0.195 to 0.310; lower copying by itself was not evidence of better style.

**Managerial supervision:** Context restored managers as the subject in all three samples. One remained close to the assigned contract-supervision/viability claim. The other two imported performance and compensation from elsewhere in the source. One made compensation adjustments a mechanism for maintaining firm viability; the other connected the managerial function to future compensation. Correct actors did not ensure correct relationships.

**Average dividend-initiation rates:** Context replaced the invented enrollment/student setting with firms and dividends in all three samples. Two retained the expected and actual rates, although the historical never-payer cohort was weakened or omitted. The third invented widening to a **6.8-percentage-point gap in 1998**. Here 6.8 was the period-average expected rate, not a 1998 gap; the source's 1998 rates imply a 4.4-point gap.

**1998 numeric control:** Local-only generation reproduced the erroneous 92 percent shortfall in one sample; 5.2 expected versus 0.8 actual implies about 84.6 percent. The context arm avoided that particular calculation, but all three samples reframed the actual-versus-expected discrepancy as **declining predicted initiation**. The broader source separately discusses declining predictions over time, but importing that framing does not preserve the assigned evidence about actual initiation and propensity. Thus context did not reliably improve even the group whose quantities were already available locally.

The context intervention added both source text and context-use instructions, including a prohibition on derived quantities. These effects cannot be separated by this experiment. Avoiding the 92 percent error is therefore not evidence that context alone fixes arithmetic.

## Why the automatic verifier does not settle this

Local-group verifier verdicts were 7 risk / 2 consistent for A and 8 risk / 1 consistent for B. Valid antecedent resolution can change a context-free premise enough to trigger a warning.

Full-context forward NLI assigned entailment above 0.9 to 8/9 A outputs and 7/9 B outputs, including plainly problematic outputs. Enrollment/intentions substitutions and imported relations show why this probability must not be interpreted as verified claim preservation. Conversely, the invented 1998 gap did receive low entailment. Keep the full verdicts as diagnostics, without treating either score as a semantic grade or rejection gate.

## Decision

The result supports a narrow conclusion: missing referents contributed to the failure, but full-context access introduced another failure mode. It does not show that structural style transfer is ineffective, nor that further QLoRA training would fix realization.

The next focused check should use **minimal, source-supported antecedent expansion**: make actors and the never-payer cohort explicit inside the assigned group, while withholding unrelated paragraph claims. Compare that input against the reproduced local-only and full-context controls. Keep quantity roles and style assessment separate; resolving referents is not permission to alter predictions into outcomes or invent causal mechanisms. A positive result would still need a broader realization check before structural-model training.

No follow-up notebook or GPU job was created by this review. No existing experiment, dataset, hash or model was modified; the 45 reserved test examples remain unused.

Private artifacts: `research_runs/group_context_restoration_v1/kaggle_version_1/`; replay verification, hash-bound assessments and review manifest: `research_runs/group_context_restoration_v1/version_1_review/`. Public aggregates: [group_context_restoration_v1_results.json](group_context_restoration_v1_results.json).
