# Mixed-500 completed: better retention, more copying

Kaggle `oliverlds/corpus-500-pilot-4b`, version 1, completed all 500 examples,
250 updates and 48 outputs. SDK metadata confirmed the private notebook's current
version was exactly 1. Downloaded code, config, model revision, packages, training
order, adapter and output hashes match the prepared packet. Review required no
new training or external inference.

All 524 label checks passed without truncation. All 250 gradients were finite
and nonzero. The adapter changed and reloaded exactly: 136 FP32 tensors and
2,228,224 parameters. No output was empty or capped. Runtime was 2,235.7 seconds
(37.3 minutes); peak allocated GPU memory was 6.03 GB. Recorded common credential
patterns had zero matches in the saved log.

## Output comparison

| Endpoint | Existing inputs: base → adapter | Gemini inputs: base → adapter |
| --- | ---: | ---: |
| Outputs | 16 → 16 | 8 → 8 |
| Material claim issues identified | 11 → 3 | 6 → 3 |
| Median input word-sequence similarity | 0.330 → 0.668 | 0.266 → 0.513 |
| Median reference word-sequence similarity | 0.298 → 0.567 | 0.255 → 0.444 |
| Median longest shared input word run | 5 → 11 | 5 → 7.5 |
| Exact word-token copies | 0 → 0 | 0 → 0 |
| Token-weighted reference NLL | 3.369 → 0.893 | 4.042 → 1.282 |

Codex read all 48 outputs with snapshot labels masked as A/B, then bound judgments
to the saved key and output hashes. The agent knew the experiment context: these
are provisional, partially masked same-agent judgments, not independent calibrated
ground truth. Minor shifts were recorded separately rather than automatically
treated as material failures. All outputs were judged readable, with some literal
HTML subscript formatting.

The adapter retains more source information and is closer to references, but also
copies more. Lower NLL and higher lexical similarity do not establish distinctive
Fama-style improvement. The 24 input variants share 16 references across eight
reused development papers; no sealed test was accessed.

Remaining trained-output errors include **indispensable → indivisible** assets,
**usually → consistently**, debt-issuance **long-term share → long-term stock**,
general **price level → asset prices**, a misstated informativeness principle,
and **soon-to-underperform → already underperforming** assets. These change
economic assumptions, scope or timing; they are not harmless synonym penalties.

One development input/reference also contains an unresolved inherited discrepancy:
“previous 5 years” followed by “t − 1 through t − 6.” It is recorded as a source
consistency issue, not a new model error. No scoring text was silently repaired.

## Compared with the 200-pair simple-prompt run

The same 16 inputs/references, inference instruction and decoding are used. All
16 base outputs are byte-identical. The 500 adapter has higher input similarity
on **16/16** and higher reference similarity on **15/16** cases. Median input
similarity rises from 0.506 to 0.668; reference similarity from 0.418 to 0.567.

The earlier review flagged 6/16 trained outputs versus 3/16 now, but those judgments
were not newly harmonized. Identical base outputs received 10/16 flags previously
and 11/16 now, demonstrating judgment instability. Raw semantic-count differences
are descriptive, not calibrated improvement estimates. Teacher composition,
paper exposure and optimizer updates also change with sample size.

## Next: six matched Pangram scans

The [complete eight-case Pangram panel](STAGE_0500_PANGRAM_FULL_V1.md) has now
been imported: 24 reports, with two entirely-human trained outputs versus one
already-human input. Only one of seven AI-related inputs becomes entirely human;
human share falls in four cases. No additional training has started. The first
packet instructions below remain historical.

Update: all six PDFs have now been [imported](STAGE_0500_PANGRAM_V1.md). The
inflation example improves from AI-assisted to human-written; the Fisher example
worsens from 67% to 0.1% human share. Instructions below describe the completed
first packet, not a request to rescan it. The remaining prespecified six cases
are the recommended extension before another training run.

The first two Gemini cases were frozen before generation. Each contributes its
actual input, base output and trained output: **six texts, approximately ten
credits**. Both cases and all conditions remain included regardless of semantic
quality or detector outcomes. One content has generated price-level errors, which
must be considered alongside any detector gain.

Open the private page at
`research_runs/data_scaling_v1/corpus_500_pilot_4b_v1/review_v1/pangram_scans_v1/scan_round_1.html`.
Copy each full text separately using the usual Pangram settings. Save all six
PDFs in the adjacent `pdf_reports/` folder. Default filenames are fine; no CSV,
manual scores or sentence-level analysis is required. Estimated credit counts
may differ from Pangram's actual count. Raw mathematical markup is preserved;
its effect is a detector confound to report rather than edit away.

Already-human input labels would limit detector headroom. Report input/base/adapter
scores together and retain every outcome. Finish this small check and weigh the
preservation/transformation tradeoff before another training run or completing
the automatic Gemini track, followed later by manual GPT-6 completion.

Raw logs, adapter, assessments and receipts remain private under
`research_runs/data_scaling_v1/corpus_500_pilot_4b_v1/`. Public aggregates:
`stage_0500_results_v1.json`. Reviewer: `scripts/data/review_corpus_500_result.py`.
Four focused CPU tests passed for paired scans, full-text preservation, retention
of unfavorable outputs and tampered-download rejection. No reference or generated
paragraphs are published.
