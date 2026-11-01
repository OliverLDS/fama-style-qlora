# GPTZero and Pangram: first-round detector results

Update: [the complete twenty-versus-twenty comparison](DETECTOR_RESULTS_V2.md)
now includes the additional nine Pangram reports. This initial eleven-report
analysis and its artifacts are retained for provenance; its pending-scan
instructions are historical.

Imported **20 GPTZero basic-scan PDFs and 11 Pangram PDFs**, dated 1 November
2026. All reports were bound to the original frozen text; no user-entered scores
were needed. Document-level outcomes are sufficient for this checkpoint.
No advanced-scan allowance or sentence-level analysis is required.

**Recommendation:** use Pangram as the practical default for subsequent sparse
checks, retaining occasional GPTZero cross-checks when free allowance permits.
The observed overlap supports that operational choice, not a claim of equal
accuracy. Keep GPTZero as the primary endpoint of this already frozen checkpoint;
Pangram is an additional analysis, not a retrospective replacement that changes
the success criterion after seeing results.

## Matched quality comparison

Pangram covers the first eleven texts in the blinded scan order. This includes
three historical human references, two GPT-6 inputs and six Gemma outputs. The
partial set is not balanced across all four papers or model arms. Compare both
providers on these **same eleven texts**, not twenty GPTZero versus eleven Pangram
reports as if they were interchangeable samples.

| Known origin, shared texts only | GPTZero | Pangram |
| --- | --- | --- |
| Historical human references, n=3 | 3 Human | 3 Human Written |
| GPT-6 rewrites, n=2 | 1 Human, 1 Mixed | 2 Human Written |
| Gemma outputs, n=6 | 5 AI, 1 Mixed | 1 AI Generated, 5 AI Assisted |

For a broad **AI-related flag**—GPTZero AI/Mixed or Pangram AI Generated/AI
Assisted—agreement is **10/11**. Both flag all six Gemma outputs and neither flags
the three human references. The disagreement is one GPT-6 rewrite: mixed versus
human. Another GPT-6 rewrite is missed by both. These inputs are machine-produced
rewrites of human prose, so the observations do not establish reliable detection
of all AI rewriting. Conversely, they do show that the original unflagged GPT-6
control was not evidence that Gemma outputs would also be unflagged.

Counting any AI-related label as a positive finding gives 7/8 machine-produced
texts flagged by GPTZero and 6/8 by Pangram on the overlap. The difference is one
GPT-6 rewrite, not a difference on the six Gemma outputs. These are descriptive
counts for this construction and should not be presented as general accuracy.

No false positive was observed among the three shared human references, but
3/3 is far too small to estimate a dependable false-positive rate. The related
texts are clustered by paper; 11 agreements/disagreements are not independent
benchmark trials. Neither statistical equivalence nor broad superiority is
established. Do not treat AI Assisted as identical to AI Generated when reporting
native categories, even though both enter the declared broad-flag comparison.

## Complete GPTZero first round

| Frozen condition | Native document classifications | AI-related flags |
| --- | --- | ---: |
| Human references | 4 Human | 0/4 |
| GPT-6 inputs | 2 Human, 2 Mixed | 2/4 |
| Ordinary base Gemma | 4 AI | 4/4 |
| Base + source structural prior | 4 AI | 4/4 |
| Existing adapter | 3 AI, 1 Mixed | 4/4 |

There is **no broad-flag improvement** from either intervention on this completed
four-paper round. One adapter output changes from AI to Mixed and has a much
lower reported AI probability; it is still mixed, not human. Generic exposition
preferences and the previously reviewed meaning improvements remain separate
findings. The full milestone is not achieved, and no Fama-style gain is established.
The second frozen round has not been scanned; do not extrapolate these counts
to all eight cases.

### Preserve score semantics

GPTZero reports document-level AI probability and a native class. A **0% AI
probability can accompany a Mixed class**: zero probability of the fully-AI class
does not mean the human class was selected. The parser now preserves the native
`Mix of AI and Human` label instead of applying a probability threshold. See
[GPTZero's interpretation guide](https://support.gptzero.me/articles/4965048546-How-do-I-interpret-a-GPTZero-result).

Pangram reports authorship shares for AI Generated, AI Assisted and Human Written.
These are not GPTZero-style document probabilities. Preserve all three values
and the summary class; do not average scores across providers. Tiny displayed
0.1% categories can coexist with a displayed 100% dominant category, so sums such
as 100.2% are retained without rescaling. The five AI Assisted classifications
are not evidence that the known Gemma outputs were literally edited by a human.
They describe the detector's output, not verified generation history. See
[Pangram's AI-assistance description](https://pangram.cv/knowledge-hub/what-is-ai-assisted).

## Cost and next scans

Pangram's official policy confirms **20 free credits per day**, one credit per
100 words with each submission rounded upward. The eleven reports' own word
counts imply 20 credits, consistent with the user-reported quota usage.
[Credit policy, checked 1 November 2026](https://pangram.cv/knowledge-hub/what-is-an-ai-detection-credit).

The nine remaining first-round texts are expected to require **18 credits** at
the next daily refresh. This is an estimate from the frozen text counts, not a
claim about the account's remaining balance or future provider billing. Complete
those nine before deciding whether to spend more quota on round two. Keep the
same Pangram model version if available, record actual dates, and report the
different scan days explicitly rather than claiming a single identical snapshot.
No GPTZero rescans, paid plan or additional account is needed for this step.

The private page `manual_scans_v1/pangram_remaining_9.html` contains only those
unscanned texts in their original blind order. Selection is based on missing
coverage, not observed scores or writing quality. The user only scans and
downloads PDFs; default filenames are accepted and there is no CSV entry.

## Audit and reproducibility

Models: GPTZero `2026-10-07-base`; Pangram `4.1`. All current reports show
1 November 2026; report timezones and numeric class thresholds are not shown.
Original PDFs, raw and visual-order extraction, parsed values, exact text hashes,
paired comparisons and receipts remain private under
`research_runs/milestone_checkpoint_4b_v1/detector_comparison_v1/`.
Public aggregate results are in `detector_results.json` beside this report.

PDFKit's object-order text extraction displaced highlighted fragments in some
GPTZero files. The importer retains raw extraction and reconstructs visual glyph
order separately. Pangram's in-text segment labels and page footers are removed
only for full-text binding; numbers, words and polarity are not corrected.
Nineteen reports matched raw whitespace-normalized text, seven matched visual
order after punctuation-spacing normalization, and five required a versioned
visual confirmation of PDF word-spacing artifacts. Those five also have exact
non-whitespace character matches and filename corroboration; a filename alone
never suffices. A whitespace-compacted match without visual confirmation is
rejected, since whitespace can change word meaning.

Score/class extraction uses page one. All pages are available for matching when
a paragraph continues onto a second page; this was needed for one Pangram report.
No sentence-level score was invented. Unknown formats and conflicting repeat
reports remain failures rather than chosen best scores.

Offline implementation: `fama_style.research.detector_provider_comparison`.
Matched observations, aggregate findings and receipts reproduced byte-for-byte.
A subsequent audit-only code change verifies the five rendered evidence hashes;
its validation receipt is separate and the original result manifest is preserved.
The 630-test CPU suite passed, including mixed-class interpretation, authorship
share semantics, text binding across pages and matched-subset denominators.
No detector call or GPU job was initiated by the import/review tools, and no prior
results were overwritten.
