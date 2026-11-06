# Mixed-500 Pangram check: one improvement, one deterioration

The [complete eight-case panel](STAGE_0500_PANGRAM_FULL_V1.md) is now imported.
This page retains the initial two-case findings. The full result is one
improvement, four deteriorations and three ties versus actual inputs; no further
scanning of the previous 24 texts is requested.

All six reports were imported from the two cases frozen before generation:
Gemini input, base output and mixed-500 adapter output for each. Every PDF was
bound to its complete scan text, not merely its filename. Four matched after
whitespace normalization; two required visual reading-order and punctuation/
whitespace normalization. PDF hashes and raw/layout extractions are retained.
All reports are Pangram 4.1, dated 5 November 2026. No new detector or model call
was made by the importer.

| Case | Gemini input | Base Gemma output | Mixed-500 output |
| --- | --- | --- | --- |
| Fama/coauthor inflation/bill-rollover paragraph | AI-assisted; 0.1% human | AI-assisted; 0.1% human | Human-written; 100% human |
| Shiller discussion of Fisher's crisis theory | Mixed; 67% human, 33% AI-assisted | AI-generated; 0.1% human | AI-assisted; 0.1% human |

Percentages are Pangram's **displayed document authorship shares**, not AI
probabilities. Tiny display floors are retained unchanged: displayed shares can
sum to slightly more than 100%. “AI-generated” and “AI-assisted” are distinct
native labels; moving from the first to the second is not equivalent to being
classified as human-written.

The first case provides a concrete positive example: the input had detector
headroom, the fine-tuned output became human-labeled, and the automated meaning
review found no material issue in that trained output. However, its input
word-sequence similarity is 0.868, and the output uses literal HTML subscripts.
Copying and formatting remain plausible confounds; the result does not isolate
learned human style or distinctive Fama choices.

The second case moves in the wrong direction relative to its input: human share
falls from 67% to 0.1%. Relative to base, the label changes from AI-generated to
AI-assisted but the human share does not improve. Both generated outputs also
change the general **price level** to **asset prices**, a material error already
recorded before detector results were available.

Thus the model improves one case and worsens the other against the actual input.
Neither input is entirely human-labeled, unlike the earlier eight-input control
panel, so this particular Gemini panel has usable detector headroom. Two cases
do not establish general Gemini detectability, a stable fine-tuning benefit,
detector accuracy, or Fama-style learning. No outcome was excluded for meaning
issues or unfavorable scores, and no account identifiers were collected.

The six-case remainder of the already frozen eight-case Gemini panel is the
next useful check before another training run or separate-track expansion:
eighteen additional input/base/adapter texts, scheduled within the ordinary
daily allowance. This remainder is now prepared under the private run's
`review_v1/pangram_remaining_v1/`: round 2 has nine texts at an estimated
15 credits, and round 3 has nine at 17 credits (32 total). Open `start_here.html`
or the individual `scan_round_2.html` and `scan_round_3.html` pages. Save all
18 PDFs in that packet's own `pdf_reports/` folder, leaving the first six in
their original folder. Default filenames are fine; no manual scores or CSV.
The remainder has not been scanned or automatically submitted. This
extends the prespecified panel rather than selecting cases after seeing results.
Do not launch a larger training stage on the strength of the single positive
example; preserve the current model and mixed findings.

Private receipts and extractions: `research_runs/data_scaling_v1/corpus_500_pilot_4b_v1/pangram_reports_v1/`.
Private observations and paired import: `pangram_import_v1/` under the same run.
Public aggregate: `stage_0500_pangram_v1.json`. Importer:
`scripts/data/import_corpus_500_pangram.py`. CPU tests cover positive/negative
triplets, native percentage floors and duplicate/missing-report rejection.
