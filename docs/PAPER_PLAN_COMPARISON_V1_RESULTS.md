# Six-paper ordinary-versus-plan results: reliable control, selective benefit

**Edit guidance controls the requested organization, but its incremental style benefit is selective.** Ordinary rewriting with the same source evidence already makes the intended edits in three of six cases. Guidance realizes all six reference programs; among the four paper extensions, one has a clear organization benefit, one is borderline because of sentence density, and two add little beyond ordinary rewriting. This supports optional structural control, not an unqualified author-style advantage or a decision to train a mapper.

Both files passed strict import: **12/12 records accepted**, with correct job IDs and prompt hashes, no missing/duplicate records and no repairs. Raw responses are retained byte-for-byte. The user confirmed **GPT-6 with High thinking effort** and explicitly raised prior-chat context concerns. Context independence is not assumed. The precise per-batch chat arrangement/first-attempt history was not separately reconfirmed; hidden context and decoding remain uncontrolled. No additional generation, GPU or training call was made during review.

## Matched comparison and verified evidence

A receives ordinary rewriting instructions plus readable source/claim evidence. B receives the identical scaffold and style instruction plus the existing edit program. Both share the same six training paragraphs, evidence and order. A is therefore a strong shared-evidence baseline, not bare-text prompting. Both arms use paired annotation information; B additionally uses reference-informed target grouping. References remain evaluation-only.

Verified package/parent hashes, original response hashes and raw copies, prompt identities and coverage. Replayed surface features, overlaps, style distances, warnings and counts. A separate exploratory review maps every source claim to exact output sentence spans and compares actual grouping with the reference program. This is an assistant interpretation, not independent semantic proof. The seven relevant CPU regression tests passed.

| Result | A: ordinary/shared evidence | B: readable edits/shared evidence |
| --- | ---: | ---: |
| Accepted outputs | 6/6 | 6/6 |
| Reference-derived count matches | 3/6 | 6/6 |
| Actual reference claim grouping, exploratory | 3/6 | 6/6 |
| Grouping matches on two prior anchors | 1/2 | 2/2 |
| Grouping matches on four paper extensions | 2/4 | 4/4 |
| Substantive claim-drift cases identified | 0/6 | 0/6 |
| Near-source copies, overlap ≥0.90 | 0/6 | 0/6 |
| Mean source overlap | 0.720 | 0.768 |

A is not required to reproduce a plan it never receives. Its nonmatching organization is not automatically a writing defect. Likewise, B's compliance cannot itself establish a useful style gain. High overlap in B can reflect the explicit KEEP instructions; it is not a copying failure when the intended local operation is performed.

## Four cross-paper extensions: primary incremental result

| Paper | Ordinary output | Guided output | Incremental assessment |
| --- | --- | --- | --- |
| New Lists 2003 | Already joins the profitability difference and interpretation in two sentences | Same grouping, close wording | Little or no useful difference |
| Disappearing Dividends 2000 | Already joins merger-rate evidence with its implication in four sentences | Same grouping, retains rate/cohort distinctions | Little or no useful difference |
| Agency Problems 1980 | Puts human-capital commitment with the management function, then discusses compensation; future wages and stake remain separate | Opens with human capital and compensation dependence; keeps management/labor-market evidence separate; connects future wages with personal stake at the end | **Clear reference-grounded argument-organization gain**, beyond a four-sentence count |
| Volatility Lessons 2018 | Clear volatility-to-risk explanation, with long-horizon risk in a separate final sentence | Combines short- and long-horizon risk qualifications under the volatility explanation | **Borderline gain:** closer integration/reference structure, but a longer, denser sentence; A is already clear |

The volatility result should not be turned into an automatic style win. It groups the correct claims rather than adding filler, yet whether that denser presentation is preferable depends on the writing objective. The conservative assessment yields **1/4 clear extension gains**; counting that borderline integration as worthwhile yields **2/4**. Thus the predeclared two-of-four trigger is **not unambiguously met**. Under either assessment this is a small, unblinded convenience sample, not significance testing or a demonstrated distinctive Fama/coauthor-style advantage.

## Two prior positive anchors: descriptive replication

**Hierarchy:** ordinary rewriting already merges the hierarchy/process claims and retains examples and informal monitoring separately. Guidance makes essentially the same organization. Correct realization replicates, but guidance adds no clear benefit.

**Market efficiency:** ordinary rewriting preserves the author's presumption and conditions but combines the finding, author earnings presumption and separate IPO presumption in one sentence. It also combines the listing evidence and attributed motivation. Guidance separates these into the intended eight-sentence sequence without misattribution or filler. This is a useful structural-control result, but it is a previously selected anchor and is excluded from the four-extension trigger. Ordinary's more compact version is not inherently incorrect.

## Meaning, style and copying remain separate

No substantive claim error was identified in the twelve outputs. Profitability qualifiers and the second-through-fifth-year scope remain intact. Merger and delisting rates remain attached to the correct payer/nonpayer groups; the distinction between payer counts and payer percentages is retained. Managers' likely compensation dependence, contracting role and current-versus-future wage distinction remain. The IPO/SEO citations, author presumption, separate IPO presumption and conditional correction arguments remain. Volatility/probability statements retain the chance interpretation and three/five/ten/twenty-year horizons without invented percentages or portfolio mechanisms.

There are surface warnings, including changes of quantifier/modality wording and expanding 1978–99 to 1978–1999. These are reviewed in context, not treated as automatic semantic failures. Conversely, no bounded relation conflict is not a certificate of equivalence. Small wording nuances remain interpretive; “none substantive identified” is an exploratory judgment, not verified semantic accuracy. No new NLI run was performed.

The outputs are mostly restrained, source-like economic prose. Target-like local organization is visible, but the experiment does not establish that readers would recognize the author or prefer the guided prose. Reference lexical overlap, surface style distances and sentence counts are not author classifiers. Source similarity below 0.90 does not establish meaningful transformation, and KEEP-heavy tasks should not be judged by an edit-distance quota.

## Limits and decision

The same assistant assessed the outputs with conditions, sources and references visible. Grades and claim placements are unblinded, uncalibrated and not independent verification. GPT-6 High is user-reported; hidden system context/decoding and prior-chat influence remain uncontrolled. All cases are inspected training paragraphs, including two retrospective anchors. The four extensions are merges; only the anchor supplies a split. These are oracle plans, not predictions from a learned source-to-target mapper or held-out generalization results.

The supported result is narrower and useful: **a capable external realization setup can apply these plans, but the plans add value mainly when a particular reorganization is desired and ordinary rewriting does something else.** The strong ordinary baseline already solves half the selected edits without the program.

Keep ordinary shared-evidence rewriting as the baseline and plans as an **optional structural-control mechanism**. Do not start mapper/QLoRA training or scale annotation from count compliance alone. The present comparison can be documented as a mixed research finding; it does not need another larger run merely to turn it into a positive style result. Before investing in author-style learning, evidence must distinguish author-specific expression gains from generic sentence repacking. Preserve the original direct-QLoRA results as the separate approach-A baseline; these generated rewrites are not new author-style targets.

Private imports: `research_runs/paper_plan_comparison_v1/response_audit_v1/`. Replay script, source/output-span assessments, declared model metadata and verification: `research_runs/paper_plan_comparison_v1/response_review_v1/`. Original responses, annotations, source/reference datasets and all reserved test examples remain unchanged. No new training data, prompt package or job was created during review.

[Aggregate results](paper_plan_comparison_v1_results.json). [Frozen design and provisional criterion](PAPER_PLAN_COMPARISON.md).
