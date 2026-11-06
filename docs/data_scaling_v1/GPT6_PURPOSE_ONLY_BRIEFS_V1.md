# Full 500 brief return: audit and reconstruction handoff

All seventeen new files returned the expected 495 records. Combined with the
five unchanged v4 pilot briefs, the collection contains exactly 500 training-side
briefs, with matching IDs, order, target hashes and valid point/anchor references.
All responses report `ready` and empty uncertainty notes. Those are model-reported
statuses, not independent semantic approval.

All 500 are retained unchanged and included in the selected purpose-only
reconstruction round. No output repairs, purpose-label changes, note/edge sorting,
automatic rejection, or record removal was performed. Archived papers, corrected
human targets and all earlier data/experiments remain intact.

## Abstraction diagnostics

The median content-inventory/target word ratio is 0.658, counting topic, points
and fact anchors, but excluding JSON keys, labels and relationship annotations.
No brief triggers the existing twelve-ordinary-word copied-span screen. The notes
are alphabetically indexed in 499 cases by a literal string sort. The other,
P101, places a short label before its extension; punctuation explains the literal
sort difference, and no reordering was applied.

The primary-function field uses seventeen distinct categories, including 130
mechanism explanations, 83 result reports and 52 method descriptions. This is
model annotation distribution, not validated author-style classification.

Explicit relation graphs are sparse: **478 briefs have empty relation arrays;
22 contain at least one edge**. Of the new 495, eighteen have nonempty arrays.
The prompt permits an empty array when content notes already express the links.
Do not infer that these paragraphs lack semantic relationships or describe this
as a complete explicit-graph corpus. Keep the annotations as returned; relation
enrichment would be separate work before an experiment relying on those graphs.
It does not block the selected reconstruction, which excludes all relation fields.

## Content limitations and follow-ups

All records receive formal/reference and lexical/numeral screening. Direct content
checks focus on triggered numbers, compressed descriptions and technical claims.
This is not exhaustive claim alignment or an independent semantic-model review.
Fifty-one records trigger a lexical/numeral advisory screen; many differences
are percent signs, digit grouping, expanded dates, omitted footnote markers or
quantities derivable from the original. They are not fifty-one established errors.

Targeted follow-ups are preserved in the private audit:

- P139: the accepted target explicitly says bonds trade at a discount relative
  to stocks; the brief retains a liquidity premium and changed prices/returns
  without specifying that direction.
- P289: a permanent pool is stronger than a historical statement that the firms
  have never paid dividends; future nonpayment is not established.
- P085, P213, P233, P313, P329 and P353: inspect whether IPO terminology narrows a
  broader new-list population in the resulting prose. Some underlying references
  discuss IPOs, so these are wording/population checks rather than blanket errors.

These concerns remain visible for assessing the eventual input/target pairs. No
silent corrections are inserted into the user's briefs. Moderate compression and
shifts of emphasis remain acceptable. Passing an overlap screen does not establish
faithful meaning, author-style removal or downstream student-model improvement.

## Next manual step: all 500 reconstructions

Upload only `batch_01.md` through `batch_17.md` from:

`research_runs/data_scaling_v1/gpt6_purpose_only_0500_v1/stage_2_reconstruction_batches/`

Use GPT-6 with requested High effort. There are sixteen batches of thirty and one
of twenty. The jobs cover ALL 500 targets, including the five pilot targets.
They supply only topic, purpose, key points and anchors inside the brief, using
the same ordinary ID/hash/status/notes wrapper. Relations and their qualifications,
original human paragraphs and earlier prose are excluded from the upload payloads.

Save `rewrites_01.jsonl` through `rewrites_17.jsonl` in:

`research_runs/data_scaling_v1/gpt6_purpose_only_0500_v1/responses/stage_2_rewrites/`

Do not attach the saved `retained_enhanced_briefs.private.jsonl` file: it preserves
the full annotations for later use, while each individual batch is the intended
generator input. Stage 1 and Stage 2 batch membership differ because the five
reused briefs are inserted back into the original 500-slot order. Cross-chat
memory isolation is not assumed.

The private `brief_audit_v1/` folder records per-ID diagnostics, content follow-ups
and hashes. The Stage 2 manifest verifies all 500 payloads against the exact raw
brief projection. Verify with:

```sh
python3 scripts/data/audit_gpt6_purpose_only_briefs.py check
python3 scripts/data/prepare_gpt6_purpose_only_500.py check-reconstruction
```

No teacher calls, detector scans, training or model downloads were performed by
this audit/preparation. No development or reserved-test examples were used. The
candidate dataset remains disabled for training until reconstruction and assessment.
