# Completed first round: GPTZero versus Pangram

The matched round is complete: **20 GPTZero reports and 20 Pangram reports**,
covering four frozen cases from four papers, with five conditions per case.
The nine newly supplied Pangram PDFs were matched to the complete frozen text;
all 31 previously imported PDFs and their extractions were preserved unchanged.
No new generation, detector submission or training was performed by the tools.

## Findings

| Condition | GPTZero native labels | Pangram native labels |
| --- | --- | --- |
| Four historical human references | 4 Human | 4 Human Written |
| Four GPT-6 inputs | 2 Human, 2 Mixed | 4 Human Written |
| Four ordinary-base outputs | 4 AI | 2 AI Generated, 2 AI Assisted |
| Four structural-prior outputs | 4 AI | 2 AI Generated, 2 AI Assisted |
| Four adapter outputs | 3 AI, 1 Mixed | 1 AI Generated, 3 AI Assisted |

Both providers flag **all twelve Gemma outputs** as AI-related. Neither flags any
of the four historical human references. Agreement on the broad human versus
AI-related distinction is **18/20**. The two disagreements are GPT-6 rewrites:
GPTZero calls them Mixed and Pangram calls them Human Written. Pangram misses all
four known GPT-6 rewrites in this construction; GPTZero misses two. These are
descriptive counts, not estimates of general detector accuracy.

Pangram is a reasonable practical default for future sparse **Gemma-output**
checks, given the daily free allowance and the same broad result here. It is not
established as equally accurate overall, particularly for GPT-6 rewrites. Keep
occasional GPTZero cross-checks within the existing free allowance. Preserve the
already frozen GPTZero primary endpoint for this checkpoint rather than replacing
it retrospectively after seeing results.

## Implication for post-training

There is **no improvement in the declared broad AI-related flag rate**: each
Gemma arm is flagged on 4/4 cases by both providers. Pangram shifts one additional
adapter paragraph from AI Generated to AI Assisted relative to base, while
GPTZero shifts one adapter paragraph from AI to Mixed. Those are native-category
changes, not Human classifications or success on the broad-flag endpoint.

The previous generic-exposition preferences and meaning review remain separate
findings. Detector labels do not establish writing quality or Fama style. The
full operational milestone has not been achieved on the observed round. No new
training is authorized by this report.

Four cases are a small, paper-clustered sample. No false-positive observation
among four historical paragraphs does not establish a dependable false-positive
rate, and complete agreement on Gemma flags does not prove detector equivalence.
The second frozen round remains unscanned. There is no need to spend additional
quota immediately: this first round already exposes the current model-output
problem. Focus the next research decision on improving the rewriting behavior,
then evaluate a newly frozen checkpoint rather than repeatedly rescanning these
same outputs.

## Score and account provenance

All reports show **1 November 2026**. GPTZero's version is `2026-10-07-base`;
Pangram's is `4.1`. Document-level results suffice; no advanced scan is needed.
GPTZero AI probabilities and Pangram authorship shares retain their different
semantics. Mixed with 0% or 1% fully-AI probability is not Human. Pangram's AI
Assisted category is a prediction, not proof of human editing. The detailed
[initial report](DETECTOR_RESULTS.md) explains these distinctions.

The user reports eleven Pangram scans from the first account and nine from a
second account, **both using the same default settings**. This is recorded as
user-reported configuration, with the account batches retained separately and
without names, emails or credentials. No independent inspection of account
configuration was made. Both batches show the same date and model version; both
flag every Gemma output they contain and label their human references Human
Written. The account split is a deviation from the planned single-account
collection protocol, not an independent model replication. No additional account
registration or scans were performed by the tools.

## Saved evidence

The initial partial result remains unchanged under `detector_comparison_v1/`.
The complete round is under
`research_runs/milestone_checkpoint_4b_v1/detector_comparison_v2/`:

- `raw_pdfs/` and `layout_extracted/`: forty preserved reports and extractions.
- `visual_checks/` and `visual_confirmations.private.json`: the five previously
  confirmed PDF word-spacing artifacts; no new visual exception was needed.
- `account_provenance.private.json`: initial account-batch record.
- `account_provenance_v2.private.json`: subsequent same-default-settings clarification.
- `results/`: initial complete comparison before that clarification.
- `results_v2/`: final comparison with the clarified settings provenance,
  observations, paired results, receipt and manifest.

Public aggregates are in `detector_results_v2.json` beside this report. The final
review reproduced byte-for-byte. Text-binding methods for forty reports were
24 raw-whitespace matches, eleven visual-reading-order/punctuation-spacing
matches, and the same five visually confirmed glyph-spacing exceptions.
All scores come from page one; matching can use the full scanned paragraph
across report pages. No filenames alone, user-entered CSV scores or invented
sentence-level measurements were used.

Validation: 631 CPU tests passed, including complete, unique, PDF-bound account
provenance coverage. Public-file checks found no credentials or private-paragraph
exposure.
