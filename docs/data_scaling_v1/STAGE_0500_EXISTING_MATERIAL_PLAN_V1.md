# Reach 500 using existing material first

The capacity audit supports keeping the current four author groups and using
already downloaded paragraphs before expanding the roster or paper collection.
There are 200 frozen training pairs plus ten approved unused pairs. Another
290 unique targets must be qualified to reach 500, balanced at 125 per group.

| Author group | Frozen | Approved unused | Additional pairs to qualify |
| --- | ---: | ---: | ---: |
| Fama/coauthors | 50 | 0 | 75 |
| Shiller | 50 | 8 | 67 |
| Holmström/coauthors | 50 | 0 | 75 |
| Baker–Wurgler/coauthors | 50 | 2 | 73 |
| Total | 200 | 10 | 290 |

Existing training-designated extraction pools contain distinct targets outside
the current approved pool: 323 Baker–Wurgler, 176 Holmström, 358 Shiller and 59
Fama candidates. In addition, the older Fama training release has 211 distinct
targets outside the current pool. These are capacity counts, not newly approved
pairs or guaranteed yields. They can overlap other historical annotation work;
annotation histories must be reconciled before requesting generation. The older
Fama pairs require current-standard reassessment, not automatic admission.

Proceed in this order:

1. Preserve the exact 200-row prefix and add the ten approved unused rows.
2. Reuse qualifying older Fama pairs and other existing GPT-6 responses after
   deduplication, split checks and alignment reassessment. Do not regenerate
   annotations merely because their schema or version differs.
3. Select the remaining human targets across existing training papers using
   paper interleaving and source/OCR checks. Keep multiple paragraphs from the
   same paper only when they contribute distinct targets. Do not inflate counts
   with paraphrase variants or split fragments.
4. Generate only missing neutral inputs. Correct the Gemini trial's tendency
   to summarize before bulk generation. Keep all requests/results, versioned
   repairs and generator provenance. If teacher composition changes, label it
   explicitly; the result is a practical expansion curve rather than evidence
   isolating size from teacher effects.
5. Freeze 500 only after these checks, preserving held-out papers and the fixed
   development panel. No new author is needed to meet the current goal. Download
   more papers only if a group has insufficient qualified coverage after this
   recovery, rather than adding them speculatively.

Adding authors now would change the target mixture while changing sample size.
That belongs in a separate later experiment. New paper downloads likewise have
lower priority than using the existing twenty-paper-per-group inventory.

Private planning artifacts:
`research_runs/data_scaling_v1/stage_0500_planning_v1/`.
The public aggregate is `stage_0500_capacity_v1.json`. This is a capacity audit
and curation plan, not a 500-pair training-ready dataset. No new model calls,
manual annotation batches or GPU training were launched by this audit. All
inferencer use remains free-tier only.
