# Full-cohort GPT-6 brief expansion: Stage 1 assessment

All eleven returned files contain the expected 320 records, in the expected
order, with matching target hashes and valid brief fields. Raw responses remain
unchanged. No training, teacher API calls or model downloads were performed.

Of the 320 briefs, 318 are retained unchanged and two receive local clarifications:

- P257: remove an unsupported benchmark-minus-event subtraction direction when
  describing cumulative wealth comparisons. Keep the idea of ratios versus differences.
- P495: retain the stated dates, percentages and total. Reconstruction does not
  need to infer the number of annual observations. Mark the brief usable with that
  scope instruction; the target's original date/count ambiguity remains unresolved
  in the audit, rather than silently substituting a corrected year or total.

Three exact, replayable edits affect these two records. All 320 proceed to Stage 2.
There is no request to repeat any of the manual Stage 1 work.

The brief notes do not trigger the twelve-word ordinary copied-span screen; the
maximum aligned ordinary span is six words. The median brief/target word ratio is
0.606, including task and anchors. Notes generally separate economic content from
authorial presentation rather than preserving whole sentence templates. These are
brief diagnostics, not evidence that generated paragraphs or student outputs will
have sufficient contrast or recognizable author style.

Assessment covers all brief structures, bindings and copied-span metrics, together
with content-note assessment. Direct comparisons with targets focus on flagged
numbers, compression and technical relationships. It is not exhaustive claim-level
alignment or independent semantic certification. Modest compression and emphasis
shifts remain acceptable; this audit does not attempt to restore authorial phrasing
for maximal literal fidelity. Original PDFs and historical assertions were not
independently reverified.

## Next manual step

Use only `batch_01.md` through `batch_11.md` in:

`research_runs/data_scaling_v1/gpt6_track_0500_v1/full_reconstruction_v1/stage_2_reconstruction_batches/`

Each file includes its instructions and reviewed briefs: ten batches of thirty,
then twenty. Use GPT-6 with requested High effort. Keep Stage 2 separate from
Stage 1; do not attach originals, earlier rewrites or Stage 1 files. No assertion
of cross-chat memory isolation is made.

Save `rewrites_01.jsonl` through `rewrites_11.jsonl` in:

`research_runs/data_scaling_v1/gpt6_track_0500_v1/full_reconstruction_v1/responses/stage_2_rewrites/`

The CPU checker verifies all 320 IDs and the brief-only allowlist. An audit manifest
binds raw response hashes, reviewed briefs, exact edits and generated packets.
Run `python3 scripts/data/review_gpt6_full_briefs.py check` to replay verification.

The existing 175 brief/reconstruction responses remain reused; 173 previously
qualified reconstructions are available. Across both rounds there are now 495
collected briefs. Five missing targets await source repair, and two existing
reconstructions retain source concerns. The full 500-target release remains
incomplete and disabled for training. No development or reserved-test examples
were used to prepare this packet.
