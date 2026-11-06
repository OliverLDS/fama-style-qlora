# Full 500 purpose-only reconstructions: observed output audit

All seventeen files contain the expected 500 outputs, in order, with matching
IDs and target hashes and valid output fields. The raw model reports 498
`rewritten` and two `source_problem` records. Every record has a paragraph.
All 500 prose/target candidates and all 500 enhanced-brief/target candidates are
staged. No output is edited or discarded, and none is marked training-ready.

## Contrast and coverage

The collection has 125 cases each for Fama/coauthors, Shiller, Holmstrom and
Baker–Wurgler. No development or reserved-test paragraphs were selected or assessed.

| Surface diagnostic over all 500 | Median |
| --- | ---: |
| Word-sequence similarity to human target | 0.139 |
| Ordinary shared five-gram coverage | 0.000 |
| Longest aligned ordinary-word span | 3 |
| Reconstructed/human word-count ratio | 0.703 |

None triggers the existing follow-up screen of sequence similarity at least
0.60 or an aligned ordinary span of at least twelve words. Eighteen outputs are
shorter than 40% of their target length; they remain included, with compression
as an advisory observation. Fixed terms, citations, formulas and technical scope
make these imperfect surface proxies.

There are 495 comparable earlier fresh reconstructions with identical repaired
human targets. Median sequence similarity changes from 0.158 to 0.139; it is lower
in 303 cases and higher in 192. This is a modest aggregate contrast change with
mixed individual results, not a count of style wins. No detector, author-style
classifier, independent semantic judge or student model was run. The evidence
concerns teacher-data construction, not demonstrated fine-tuning improvement.

## Confirmed targeted conflicts

Two cases require focused upstream regeneration before freezing a complete release:

| Case | Conflict | Required action |
| --- | --- | --- |
| P139 | The output introduces a **price premium**, while the user-accepted human target explicitly describes a discount relative to stocks. The brief had omitted that directional comparison. | Re-extract the brief with the accepted comparison explicit, then reconstruct from that corrected brief. Retain the human target unchanged and do not invent a yield interpretation. |
| P400 | The output describes **subsequent-return forecasting**. Native PDF page 16 places the original under “Using Sentiment to Explain Current Returns”. The brief did not state that timing. | Re-extract with the verified section scope as context, then reconstruct. Do not change the original target or silently replace its research question with forecasting. |

The native context comes from the archived extraction of B_BW2007, PDF page 16.
The private evidence copy and source-file hash are recorded. No full visual or
independent historical verification of the PDF is claimed.

A single two-record job is prepared at:

`research_runs/data_scaling_v1/gpt6_purpose_only_0500_v1/output_audit_v1/targeted_recollection_v1/batch_01.md`

Use GPT-6 with requested High effort; save `briefs_01.jsonl` in that folder's
`responses/stage_1_briefs/`. After return, check the briefs and prepare a separate
purpose-only reconstruction. The job changes the scoped prompt/context rather
than editing returned prose by hand. No bulk collection needs to be repeated.
The two current observations remain in the 500-candidate snapshot, visibly flagged;
they are not silently dropped or counted as qualified replacements.

## Model warnings and smaller follow-ups

- **P056:** the model warning treats young-stock demand under optimism as
  inconsistent with lower subsequent returns. Native A_BW2006 PDF page 17
  explains that returns are conditioned on prior-year sentiment. The observed
  paragraph preserves the −0.54% and +0.85% signs and explicitly describes future
  returns. The warning is nonblocking; raw metadata remains preserved.
- **P206:** a complete definition of b is absent from the excerpt, but the output
  preserves the original conditional small-b adjustment claim. A parameter's
  missing full definition is not, by itself, a reason to reject this paragraph.
- **P085, P313, P329, P353:** IPO wording can narrow a broader new-list population.
  P085 also introduces a financing-friction explanation not explicit in the local
  paragraph. These remain scope/mechanism advisories, not automatic exclusions.
- **P289:** the new prose avoids the brief's unsupported permanent-pool wording.
  P213 and P233 also retain the main new-list population rather than uniformly
  restricting it to IPOs. Their brief-stage concerns do not automatically become
  material output errors.

All-record formal, reference and surface screening is supplemented with triggered
case reading, fixed author-stratified output samples and targeted native-context
checks. This is **not exhaustive claim alignment or independent certification**;
the two identified conflicts are not an estimate that only two errors exist.
Moderate compression, explanatory additions and emphasis shifts remain acceptable.

## Private artifacts and reproducibility

Folder:
`research_runs/data_scaling_v1/gpt6_purpose_only_0500_v1/output_audit_v1/`

- `candidates_0500.private.jsonl`: all 500 observed prose/human pairs, raw model
  statuses, full enhanced briefs, provenance and follow-up labels.
- `candidates_0200.private.jsonl`: the exact first-200 prefix of that version.
- `brief_human_candidates_0500.private.jsonl`: all enhanced brief/human candidates;
  the complete brief is saved, not a final decision about a training serialization.
- `diagnostics.private.jsonl`: per-case overlap metrics and advisories.
- `context_evidence.private.jsonl`: native text supporting the timing adjudications.
- `manifest.json`: response, input-packet, evidence and artifact hashes.

Relation annotations remain sparse: 478 enhanced briefs have empty relation arrays.
They are saved as returned; this is not a complete explicit-relation corpus. The
observed reconstruction used topic, purpose, points and anchors, with no relation
annotations forwarded. GPT-6/High was requested; actual UI settings are not
independently verified.

Verify with `python3 scripts/data/audit_gpt6_purpose_only_outputs.py check`.
No teacher API calls, model downloads, detector scans or GPU training were made.

All 500 observed prose pairs are also available in the local read-only review
interface under “GPT-6 purpose-only 500 · observed candidates”. The two confirmed
conflicts have explicit regeneration-pending badges; earlier versions remain
available for comparison. Nothing is published by adding a local catalog entry.
