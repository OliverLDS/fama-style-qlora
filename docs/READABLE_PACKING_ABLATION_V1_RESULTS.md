# Readable packing ablation: completed review

**The readable format improved sentence-count compliance, but did not demonstrate faithful execution of the claim-bound edits.** There is modest concision progress in the hierarchy example; the market-efficiency example still fails the requested split and corrupts attribution. This is useful evidence about prompt usability, not readiness for training a mapper or adapter.

All **4/4 calls completed**, with no empty/cutoff responses, in **178.0 seconds (3.0 minutes)**. The user reported `oliverlds/readable_packing/1`; artifacts and the completion log were retrieved from canonical slug `oliverlds/readable-packing`. Frozen base Gemma 3 4B was used without an adapter or training. The cases are two retrospectively selected training paragraphs, with oracle reference-derived grouping. No development or reserved test data was read.

## Verified execution

Verified the prepared pack, frozen runtime, artifact hashes, all prompts, model/NLI revisions, decoding and complete source/output-bound diagnostic evidence. Replayed metrics and `evaluation.json` from raw generations and saved probabilities; they matched within floating-point tolerance. The execution log ends in `inference_complete`.

Both offset/JSON control prompts **and their actual outputs** reproduce the previous elementary-plan run exactly. Thus this comparison has a directly checked historical control, rather than relying on a similarly worded baseline. Input lengths were 987–2,583 tokens; all prompts fit the unchanged 4,096-token total context budget with 768 tokens reserved for output. Truncation does not explain the result.

The installed CLI's output request ignores the supplied version suffix. Version 1 remains user-reported rather than independently pinned, while the retrieved artifacts independently match the exact prepared experiment. No additional model or GPU calls were made during review.

## What changed

| Case | Desired sentences | Original offset/JSON | Readable format | Actual operation finding |
| --- | ---: | ---: | ---: | --- |
| Hierarchy merge | 3 | 4 | 3 | Both make the initial merge. Readable output then merges two claims that should occupy separate KEEP slots and fills the third sentence with a generic conclusion. |
| Finding versus author presumption | 8 | 6 | 8 | Readable output still puts the finding and presumption in the same sentence; extra interpretation and repeated listing motivation supply the additional sentences. |

Count matches increase from **0/2 to 2/2**, while complete claim-to-slot programs remain unrealized in **both readable outputs** under the exploratory text audit. No output is a near-source copy at the frozen overlap threshold. This failure is largely one of reorganization/content control and generic expansion, rather than excessive copying in these two cases.

The source-to-reference surface-style distance proxy improves more under readable prompts: mean gain **0.685 versus 0.497**. Both paired proxy differences favor the readable format. These features are affected by sentence count, length and generic phrasing; they do not establish author-style alignment. The eight-sentence output is an especially clear case where a favorable number coexists with incorrect claim realization.

## Hierarchy: modest concision, wrong final slots

The readable output's first sentence successfully combines formal hierarchy and the lower-to-higher decision process. The offset output already did this. The readable second sentence combines organizational examples with informal monitoring; those were supposed to remain distinct source claims in separate output slots. Its third sentence generalizes about consistency and a fundamental decision-making mechanism instead of expressing the intended final monitoring claim there.

The main source claims remain largely present in the first two sentences. The generic final interpretation is a **minor or uncertain extrapolation**, rather than an obvious reversal of a central claim. Compared with the offset output's added uncertainty-reduction/coordination effects, the readable output is shorter and arguably less speculative. This is a modest positive signal worth retaining; it is not a substantial demonstrated Fama/coauthor-style improvement or faithful plan execution.

## Market efficiency: the critical split still fails

The readable output assigns the strong-earnings presumption to Mitchell and Stafford, although the supplied source presents it as the author's own presumption. It keeps that presumption together with their stock-return finding in sentence 2. The separate IPO-earnings presumption is omitted rather than placed in its intended sentence.

It also turns the conditional gradual-correction argument into an asserted gradual correction, recasts the author's over-reaction interpretation as a consistent cited finding, and adds listing-timing interpretation. Listing motivation is subsequently repeated. These additions achieve eight sentences without executing the specified finding/presumption split. This is **substantial claim drift**, not a minor wording change justified by a clear style benefit.

The corresponding offset output also has high drift, including the same attribution/conditional problems. The readable format therefore does not fix the central failure in this case.

## Assessment limits

All four full-paragraph verifier results are `risk_signal`; this neither proves that all changes are equally severe nor cancels the hierarchy's modest concision improvement. Claim-location matrices mostly abstain or flag slot mismatches. The original machine metrics remain unchanged, including their unassigned drift-severity/acceptability fields.

The text-level assessments were made separately by the same in-session assistant with source, reference and conditions visible. They are unblinded, uncalibrated and not independent verification. Exact source/output evidence spans and hashes are saved privately. Minor or uncertain hierarchy drift and high market-efficiency drift are deliberately distinguished; the user’s allowance for a worthwhile style-versus-drift tradeoff remains in force.

Source binding display and plan wording changed together, so this does not isolate their individual contributions. Two reused training cases also cannot establish generalization, statistical significance or that numeric offsets were the sole earlier bottleneck. The observed distinction is narrower: readable guidance changes layout/count behavior, while correct semantic claim placement remains unreliable.

## Next decision

Do not begin mapper or adapter training from this result. The next discriminating check is to run the **exact same four frozen prompts with a stronger instruction-following realizer**, supplying no reference wording. This tests whether the existing plans and source evidence can be realized by another model before making further prompt changes or investing in operation-conditioned supervision.

Success must include the actual finding/presumption split with correct attribution and a complete hierarchy plan without filler, along with worthwhile expression changes and an acceptable drift tradeoff. Merely obtaining three and eight sentences is now known to be insufficient. Earlier stronger-realizer evidence concerned different narrow programs and does not establish success on these tasks. No stronger-model run or new annotation task was created during this review.

Private artifacts and log: `research_runs/readable_packing_ablation_v1/kaggle_version_1/`. Reproducible replay script, exact-span assessments and review hashes: `research_runs/readable_packing_ablation_v1/version_1_review/`. The eight relevant CPU regression tests passed; experiment code, original data, annotations, previous results and training gates remain unchanged.

[Aggregate results](readable_packing_ablation_v1_results.json). [Frozen design](READABLE_PACKING_ABLATION.md).
