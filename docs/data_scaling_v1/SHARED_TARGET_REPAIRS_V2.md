# Seven user-adjudicated cases: shared target repair v2

User decisions dated 2026-11-06 are implemented in a new private version at
`research_runs/data_scaling_v1/shared_target_repairs_v2/`.
Archived PDFs, earlier datasets, response files and audit decisions remain intact.
Six target texts change; one target remains unchanged. One case replaces cohort
membership with an unused paragraph from the same training paper. These targets
are not all verbatim historical quotations: authorized corrections are explicit.

| Case | Decision implemented |
| --- | --- |
| P250 | Change the second Milwaukee “mean” to “median”, as explicitly adjudicated by the user. |
| P134 | Correct the capture sentence to Savannah, December 1778, and Charleston, May 1780. Keep the separate 1779 pay/retention concern and 1780–1781 mutinies. |
| P139 | Keep the human target unchanged. Remove the disputed-comparison criticism from the brief and reconstruction; do not invent a price-versus-yield interpretation. |
| P264 | Prepend the user-supplied page-16 text and join “World” with “Tax Database” on page 17. The full paragraph introduces data sources and defines the closedness index. |
| P291 | Replace BIS publisher boilerplate with paragraph `AC_ec77c5a9dd4736cf5654`, PDF page 6, on liquidity, collateral and transparency in money markets. Same author/paper, no existing exact target duplicate in the 500 cohort. |
| P458 | Separate the Conclusion heading from the paragraph. |
| P484 | Separate the Retail Investor Trades heading from the paragraph. |

The P134 dates are supported by the [National Park Service chronology](https://www.nps.gov/york/learn/historyculture/eventstoyorktown.htm):
Savannah was captured on December 29, 1778, and Charleston surrendered on May 12,
1780. Replacing the inaccurate state-wide capture description with Savannah also
avoids claiming that the whole state fell on one date. The PDF remains unchanged;
the dataset records a user-authorized factual correction rather than an OCR fix.

Fresh PDFKit extraction confirms the supplied P264 prefix and the P291 replacement
against native text. Evidence pages and PDF hashes are recorded privately. This
is native text comparison, not independent visual certification of entire PDFs.
P291's former target ID and mixed-pair ID remain in replacement provenance; the
new paragraph does not masquerade as a new paraphrase of publisher boilerplate.

## Candidate status and next annotation

The two earlier held reconstructions, P134 and P139, are recovered with local
edits, and their briefs are rebound to the corrected/accepted targets. This gives
**495 fresh-prose candidates and 495 matching brief/human candidates**. Those
briefs retain their actual version-1 format; no purpose labels are fabricated.

The remaining five targets require fresh briefs and reconstructions. One new
Stage 1 upload contains all five: P250, P264, P291, P458 and P484. Use
`stage_1_brief_batches/batch_01.md`; save `briefs_01.jsonl` in
`responses/stage_1_briefs/` under the new repair folder. No repeated annotation of
the other 495 is requested. After review, Stage 2 will receive only those briefs.

This small batch uses the [purpose-aware brief v2](BRIEF_PURPOSE_V2.md). The current
task/points/anchors format is extended with paragraph function, reader takeaway
and important content relationships. Existing discovery labels inform a compact
paragraph-level subset; full sentence/lexical taxonomies are not requested.

Private artifacts include the 500-slot shared target registry, the GPT-6 candidate
version and 200 prefix, exact target edits, exact source/brief recoveries, and a
Gemini target overlay. The registry applies to both teacher tracks. Six changed
target bindings require Gemini source reassessment; old Gemini review hashes must
not be silently certified against new targets. P139's target hash is unchanged.

No teacher calls or GPU jobs have been submitted. All candidates remain disabled
for training. Verify the artifact bindings with
`python3 scripts/data/repair_shared_targets_v2.py check`.

The local review catalog now includes “GPT-6 corrected targets v2 · 495 fresh /
5 pending”. Historical versions remain available. Six corrected target texts
appear as additional text versions in the viewer; this does not increase the
500-slot cohort size. P139's user-adjudicated original is no longer shown as an
unresolved target error.
