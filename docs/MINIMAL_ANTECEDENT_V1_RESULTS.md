# Minimal antecedent expansion: version 1 results

All 27 inference calls completed, but explicit actors did not produce reliable, distinctive author-style rewriting. The new inputs kept the output in the intended economic domain; they did not stop invented mechanisms, expected/actual rate errors or unsupported percentages. Further prompt-only variants on these three groups have limited value.

Kaggle: `oliverlds/minimal-antecedent`, owner-reported version 1. Runtime **219.5 seconds (3.7 minutes)**. All 27 outputs were complete single sentences. All **18/18** historical A/B control output hashes reproduced exactly. No retries, repaired outputs or assembled paragraphs.

## Identity and evidence

Downloaded output artifacts read-only. Verified the prepared pack, runtime source hashes, original artifact manifest, saved prompts/seeds/token caps, pinned model/NLI revisions, decoding and completion status. Recomputed metrics and evaluation from raw outputs and saved verifier evidence; both replayed within floating-point tolerance. Kaggle's session-output endpoint does not guarantee immutable historical version selection; the owner-specified version is supported here by pack/runtime/job identity and exact control reproduction.

The review uses three selected training groups and three seeds, not held-out examples. A/B qualitative judgments are reused only after exact source/reference/output identity checks; they are not independent new quality evidence. C was assessed by the same assistant that prepared the experiment, with conditions visible. Grades are exploratory, unblinded and uncalibrated.

## Comparison

| Observation | A local-only | B full context | C explicit actors |
| --- | ---: | ---: | ---: |
| Complete single sentences | 9/9 | 9/9 | 9/9 |
| Minor / moderate / high drift, exploratory | 0 / 3 / 6 | 1 / 2 / 6 | 0 / 5 / 4 |
| Substantial author-style improvement, exploratory | 0/9 | 0/9 | 0/9 |
| Input tokens | 2,193 | 4,299 | 2,268 |
| Output tokens | 420 | 370 | 388 |
| Generation seconds, excluding load/NLI | 39.1 | 39.3 | 36.2 |

C used only 3.4% more input tokens than A and approximately half B's input tokens. Its lower high-drift count is descriptive, not statistically established improvement: there are only three selected groups, and judgments are uncertain. No C output was assessed as combining substantial style change with only minor drift.

C's mean similarity to the original local group was 0.143, versus A 0.195 and B 0.310. This metric compares against the original group, not C's expanded wording. Low similarity did not establish useful style transfer: generic expansion and changed claims also reduce overlap.

## What the nine new outputs did

**Managerial supervision:** All three C outputs retained managers, contracts and viability. They nevertheless added operational continuity, resource allocation, interdependent-input management or risk mitigation. The compensation leakage seen in B disappeared from C, but other unsupported mechanisms remained. Style changes were generic managerial formalization, not substantial evidence of author-specific style.

**Period-average dividend rates:** All three stayed on firms and dividends, eliminating the enrollment/student domain substitutions in A. One retained the expected/actual rate ordering but weakened the never-payer cohort and made firms the apparent forecasters. Another treated the expected 6.8 rate as actual and assigned the 3.0 actual rate to firms with prior dividend history. The third made 6.8 actual and invented an expected rate of **9.8**, while omitting 3.0 actual. Explicit cohort wording did not guarantee preservation of quantity roles or cohort identity.

**1998 rate comparison:** One C output omitted the year, both numeric rates and the explicit less-than-one-sixth comparison, retaining only a qualitative shortfall. Another invented a **96% shortfall**. Expected 5.2 versus actual 0.8 implies about **84.6%**, not 96%. The third retained the two rate roles and propensity direction, but dropped the year and explicit comparison. None showed a substantial style gain that justified those losses.

These observations separate subject resolution from semantic fidelity: the model can remain on the right topic while changing the evidence or adding a mechanism.

## Automatic verifier limitations

C had three expanded-group `model_consistent` verdicts and six risk signals, B had one consistent and eight risk signals; A had eight risk signals and one abstention under the same expanded-group premise. These are diagnostic categories, not measured semantic accuracy.

Full-context forward NLI exceeded 0.9 for 7/9 C outputs, including flawed candidates. The independent economic-role errors above remain despite high entailment scores. Preserve full verdicts and explicit error evidence; do not promote high NLI confidence into a correctness label or severity grade.

## Recommendation

Pause further prompt-only refinements on these repeatedly selected examples. The useful finding is that format and referent control are insufficient; source-grounded realization remains the bottleneck. This does not disprove rhetorical planning or establish that a larger model or new training will succeed.

A more informative next step is to prepare a **small plan-conditioned supervised-realization pilot** using existing author references as targets, rather than these generated outputs. First audit a limited source/plan/reference slice for claim coverage, cohort conditions and numeric roles, preserving provenance and existing datasets. Then prepare a separate adapter experiment against the existing direct-SFT baseline, with held-out style, copying and claim-drift measurements. Do not train a structural mapper before testing whether a supervised realizer can use its plan. Keep the 45 reserved test examples out of selection.

This is a recommendation, not an executed experiment or a claim of expected improvement. No new training pack, notebook or GPU job was created during this review. GPU training requires explicit authorization. The existing allowance for minor drift when substantial style improvement is achieved remains unchanged.

Private raw artifacts: `research_runs/minimal_antecedent_v1/kaggle_version_1/minimal_antecedent/`. Hash-bound assessments, replay checks and review manifest: `research_runs/minimal_antecedent_v1/version_1_review/`. [Public aggregate results](minimal_antecedent_v1_results.json).
