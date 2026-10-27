# Groupwise realization v2: version 1 results

**The execution and sentence-format fix worked; the assembled writing was less faithful.** All 33 local calls returned valid single sentences and all nine assemblies matched the requested counts. However, group-local rewriting introduced major topic, actor, relation and quantity changes. No substantial reference-style improvement was established. Do not promote this version to training-data generation or train a mapper from this result.

Run: [oliverlds/groupwise-realization-v2](https://www.kaggle.com/code/oliverlds/groupwise-realization-v2), version 1 reported by the owner. [Prepared protocol](GROUPWISE_REALIZATION_V2.md); [aggregate results](groupwise_realization_v2_results.json).

## Execution and verification

All **42 generation calls** completed in **394.7 seconds (6.6 minutes)**, with no empty responses or cutoffs. All **18 paragraph attempts** were available, including nine whole-paragraph controls and nine assembled paragraphs. No failed sample was hidden or replaced. Paragraph and group verification completed for every available candidate.

All **nine historical controls reproduced exactly**. Uploaded pack, prompts, model/NLI revisions, sampling, budgets and runtime-code hashes matched preparation. Raw calls reconstructed the saved paragraphs exactly. CPU replay from saved full paragraph/group verdicts reproduced metrics and evaluation within 1e-12 floating-point tolerance. Source-group/output hash bindings and artifact hashes passed. This is scoring replay, not independent NLI execution or semantic certification.

The installed session-output endpoint ignores the CLI numeric suffix and serves the current session. The owner identified version 1; exact prepared experiment identity was verified. This used pinned base Gemma 3 4B without an adapter or training. Neither development-validation nor reserved-test data was used.

## Clear format improvement

| Measure | A: whole paragraph | B: separate groups |
| --- | ---: | ---: |
| Available paragraph candidates | 9/9 | 9/9 |
| Valid single-sentence group responses | Not applicable | **33/33** |
| Requested paragraph-count matches | 2/9 | **9/9** |
| Valid assembled paragraphs under piece-format checks | Not applicable | **9/9** |
| Mean words per sentence | 19.01 | 26.40 |
| Mean bounded surface-style benefit | 0.0225 | 0.0779 |
| Exact/near source copies | 0 | 0 |

This shows that focused local prompts and external assembly can control sentence boundaries. It does **not** establish faithful rhetorical or atomic-claim organization: a perfectly formatted sentence can express the wrong proposition. Assembly follows source-group order by construction; the content of each piece remains the model's responsibility.

## Content and coherence assessment

All 18 paragraphs were read against source and reference using the saved [graded policy](style_drift_tradeoff_v1/policy.json). The nine identical control judgments were retained after checking source/reference/output hashes and rereading. New assembled outputs were assessed separately. Condition/seed labels were hidden in the displayed texts, but controls were recognizable and aggregate results were visible. The same assistant designed and assessed the experiment. These judgments are **exploratory**, not independent blinded grading or calibrated accuracy.

| Judgment | A: whole paragraph | B: separate groups |
| --- | ---: | ---: |
| Slight style movement / generic formalization | 9 | 6 |
| No identified reference-style improvement | 0 | 3 |
| Moderate or large reference-style improvement | 0 | 0 |
| Minor semantic drift | 3 | 0 |
| Moderate semantic drift | 6 | 1 |
| High semantic drift | 0 | **8** |

Representative assembled failures are concrete:

- Dividend-initiation percentages become **individual or student enrollment rates**. The isolated source group states expected/actual percentages of those starting, but its local text does not repeat the dividend/firm context.
- Another group **reverses predicted and actual initiation rates**, despite the supplied group specifying the comparison direction.
- A sentence invents a **92% shortfall** for expected 5.2% versus actual 0.8%; those values imply approximately 84.6%. This error occurs even though the relevant quantities were present in the assigned group.
- Managerial contract supervision acquires unsupported risk-mitigation, resource-allocation or procurement mechanisms. Some outputs change the labor-market ability signal into a different compensation claim.
- In the premium example, one output makes investors' evaluations the object that has a probability of persisting, instead of negative premiums. Others lose horizon-specific probability distinctions.

Context loss is a plausible contributor to the enrollment and referent failures, but it is not established as the sole cause. The wrong numerical calculation and reversed relation show that restoring context cannot yet be assumed sufficient. The experiment changes both the group instruction and available context relative to v1.

Whole-paragraph NLI marked 3/9 controls consistent and flagged all nine assemblies. These are warning counts, not semantic accuracy. Local group checks marked seven consistent, 23 risky and three abstaining; their availability differs from the whole-paragraph control's positional checks, so the local rates are not a matched quality comparison.

Under the default ordinal style/drift utility, the exploratory mean was **0.104 for A versus −0.103 for B**. Those values express the saved policy, not calibrated effect sizes. Allowing minor drift does not rescue the current assembled outputs: the observed larger changes did not buy a substantial reference-style gain. The rise in the five-feature style proxy illustrates its inability to distinguish sentence-shape movement from faithful author-style transformation.

## Measured cost

| Within-run measure | A | B |
| --- | ---: | ---: |
| Generation calls | 9 | 33 |
| Input tokens | 5,175 | 8,073 |
| Output tokens | 1,087 | 1,175 |
| Generation seconds | 99.6 | 112.0 |

B used **56% more input tokens**, about **8% more output tokens** and **12.5% more measured generation time**. These timings exclude shared loading and later verification; they are not full strategy-level wall-clock or price estimates. The allocated output allowance remained 768 tokens per paragraph in both arms. Three selected training cases do not establish general cost or quality advantages.

## Decision and next useful check

Keep the successful format-control mechanism as diagnostic evidence, but do not use these generated paragraphs as faithful labels or proceed to full mapping/fine-tuning. The immediate unresolved issue is source-grounded group realization, including context, relation direction and quantity binding.

The next narrow mechanism check would restore clearly marked **read-only source context** while keeping the corrected group-only task and output instruction. Compare that with local-only input on a few failing groups before another full paragraph run. The instruction must identify which group is being written and keep context from becoming additional output. This would isolate the context-removal change more directly; it would not by itself demonstrate style transfer or solve numeric/relation errors. No follow-up prompt pack or GPU job was created during this review.

## Private evidence and preservation

Original outputs and logs remain under `research_runs/groupwise_realization_v2/kaggle_version_1/`. `version_1_review/` contains the verification checks, full text packet, hash-bound assessments, graded metrics, context/quantity failure evidence, aggregate summary and review manifest. All experimental code, packs, source data and prior outputs remain unchanged. Public files contain aggregate findings and error descriptions only, without full paragraphs, raw generations, detailed verdicts or credentials. No additional model calls or GPU jobs were made during review.
