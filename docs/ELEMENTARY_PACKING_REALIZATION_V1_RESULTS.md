# Elementary packing realization: completed review

The run completed, but **elementary edit guidance did not establish a worthwhile style-versus-drift improvement** on the five new oracle candidates. There is a limited organization/concision signal in one paragraph; it is insufficient for mapper training or operation-conditioned fine-tuning.

The user reported `oliverlds/elementary_realization/1`; artifacts were retrieved from Kaggle's canonical `oliverlds/elementary-realization` slug. All **12/12 generations completed**, without empty responses or output cutoffs, in **324.8 seconds (5.4 minutes)**. No training occurred. These are six training paragraphs from six papers, with five new candidates and one separately reserved probe, using frozen base Gemma 3 4B and reference-derived oracle plans. No development or reserved test examples were used.

## Verified artifacts

Verified the artifact manifest, prepared pack hash, frozen runtime identity, all saved prompts, model/NLI revisions, decoding settings, complete output/evidence identities and token budget. Recomputed metrics and `evaluation.json` from raw generations and saved full diagnostic evidence; both matched within floating-point tolerance. Actual input lengths were 794–2,223 tokens, comfortably within the 4,096-token total budget with 768 tokens reserved for output. Prompt truncation does not explain the failures.

The installed CLI parses a version suffix but its output request does not transmit that suffix. The separate version-pinned source/metadata request returned 403, and the session-output API supplied no standalone execution log. Thus “version 1” remains user-reported; the retrieved artifacts do independently match the exact prepared experiment. Completion is supported by the hashed run status and complete artifact matrix. No external model calls or GPU jobs were made during review.

## Five new candidates

| Measurement | A: ordinary + shared evidence | B: elementary edits + shared evidence |
| --- | ---: | ---: |
| Completed calls | 5/5 | 5/5 |
| Reference-derived sentence-count matches | 0/5 | 0/5 |
| Near-source copies, overlap ≥0.90 | 0/5 | 0/5 |
| Mean source overlap | 0.369 | 0.428 |
| Mean surface-style distance improvement over source | 0.279 | 0.229 |
| Exploratory moderate / high drift | 2 / 3 | 2 / 3 |

B had a lower surface-style gain than A in all five pairs. This proxy is sensitive to sentence length/packing and is not an independent measure of target-author style. Low copying also does not imply success: these outputs often replace source content with generic explanation and unsupported claims.

The drift/style assessments below were made by the same in-session assistant with the conditions and references visible. They are **unblinded, exploratory, uncalibrated and not independent verification**. Moderate denotes a material scope, qualification or unsupported-conclusion change; high denotes central attribution/condition corruption, invented quantities/mechanisms or multiple substantive alterations. They are not NLI labels or calibrated error rates. Exact source/output evidence spans and hashes are retained privately for every assessment.

| Paper / intended edit | Desired sentences | A | B | Paired finding |
| --- | ---: | ---: | ---: | --- |
| New Lists 2003: merge difference and inference | 2 | 4 | 4 | Neither merges the final two claims. Both change the second-through-fifth-year scope; A also adds an unsupported profitability explanation. |
| Volatility Lessons 2018: merge volatility and horizon risks | 2 | 4 | 5 | B invents a 30% probability and adds performance consequences. A also invents portfolio consequences and strengthens the long-horizon risk. |
| Fama–Jensen 1983: merge hierarchy and decision process | 3 | 6 | 4 | B follows the intended first-three-sentence organization more closely, but A already makes the initial merge. B adds an unsupported uncertainty/coordination conclusion; this is partial organization/concision, not a faithful complete program or substantial author-style gain. |
| Market Efficiency 1997: separate finding from author presumption | 8 | 6 | 6 | B keeps the finding and presumption together and attributes the latter to Mitchell and Stafford. Both lose the condition governing gradual correction and add or distort claims. |
| Agency Problems 1980: two local merges, retain management role | 4 | 6 | 5 | B omits managers' contracting/viability function and invents a network/wage-negotiation mechanism. Its first merge also occurs in A; the final requested merge remains unrealized. |

The hierarchy paragraph provides a small descriptive signal: B removes much of A's invented discussion and comes closer to the intended grouping. It still has material drift and an additional unsupported conclusion. This does not meet the predeclared requirement for worthwhile expression improvement beyond count changes on at least two new candidates from different papers. No substantial incremental target-style gain was demonstrated under this assessment.

## Reserved probe: matching counts concealed a reversal

Both probe outputs have the desired five sentences. B nevertheless reverses the source comparison between predicted and actual dividend initiation. It also drops the 6.8%-versus-3.0% evidence and misstates the one-sixth comparison. Both arms introduce statistical significance unsupported by the supplied source. The final statistic and interpretation are not combined as planned.

This probe is excluded from the five-candidate table and success criterion. It is a direct example of why matching sentence counts, low copying or fluent formal wording cannot establish successful restructuring.

## What the automatic diagnostics can and cannot say

The full verifier flagged all five A outputs and four B outputs. The remaining B output, New Lists, was labeled `model_consistent` despite changing the second-through-fifth-year scope to the second and fifth years. This illustrates a missed semantic distinction, not independent evidence of acceptability. Claim-localization matrices also mostly abstained; they do not certify realized slots.

The user's tolerance for minor drift remains appropriate. The negative decision here is not based on demanding perfect paraphrase: the guided outputs contain substantive problems, and there is no demonstrated substantial style benefit to justify them. The runner's unassigned drift severity/acceptability fields remain unchanged; this separate review supplies explicitly exploratory judgments rather than overwriting original results.

## Prompt-format limitation and next step

I found a weakness in the prompt representation used for this experiment. To reduce repetition and context usage, the shared source scaffold encoded sentence IDs, claim evidence and protected bindings largely as **numeric character offsets**. The complete source remained visible, but the model had to associate operation IDs with passages through those offsets. CPU checks prove valid offsets and complete coverage, not that Gemma can use that representation effectively. The plan also conveyed output sentence structure through a machine-readable slot list.

This is a plausible usability problem, **not an experimentally established cause**. This run therefore does not cleanly identify whether Gemma's realization capacity or the presentation of the edits is the main bottleneck. It does not refute explicit rhetorical planning as a general approach.

The next diagnostic should be a **two-case prompt-format ablation** on the hierarchy merge and market-efficiency split. Render source sentences as readable `S1`, `S2`, etc., bind each claim to its exact source excerpt, and express the same finite plan in direct sentence instructions. Compare against the frozen offset-based representation using the same model, source facts, operation assignments and decoding. Supply no target wording or added semantic relations. Do not add stricter preservation constraints, collect more annotations or start training to resolve a possible prompt-transport failure.

If readable bindings still fail, use those same tasks with a stronger instruction-following realizer to separate actionable-plan quality from this model's capacity. Earlier stronger-realizer evidence concerned different narrow programs; it does not establish success on these elementary edits. Neither follow-up has been implemented or executed during this review.

Private artifacts: `research_runs/elementary_packing_realization_v1/kaggle_version_1/`. Hash-bound assessments, replay script and verification: `research_runs/elementary_packing_realization_v1/version_1_review/`. Existing data, annotations, model results and training gates remain unchanged.

[Aggregate results](elementary_packing_realization_v1_results.json). [Frozen design and manual run instructions](ELEMENTARY_PACKING_REALIZATION.md).
