# Frozen eight-case 4B checkpoint — version 1

Update: [the completed first-round detector report](DETECTOR_RESULTS_V2.md) imports
twenty PDFs from each provider. Both flag all twelve first-round Gemma outputs as
AI-related. Earlier scan-pending descriptions below record the quality-review
stage and are superseded by that detector report.

The run completed successfully. The existing adapter and source-only structural
prior both look better than ordinary base rewriting in this small development
comparison, but neither reliably preserves all claims. The full milestone is
still unestablished: detector measurements are pending and no distinctive
Fama/coauthor style measure has been demonstrated.

## Execution and integrity

The exact notebook is `oliverlds/milestone-checkpoint-4b`, version 1. The supplied
name included an extra `v1`; read-only account lookup resolved the URL slug.
Version-specific download retrieved the log, runtime, evaluation, generation
configuration, state and token audit. Runtime source, frozen packet, original
epoch-2 adapter, package versions, revision, decoding and all 24 prompt audits
passed the export checks.

All fourteen missing outputs completed in 273.36 seconds measured by the runtime,
with approximately 4.65 GiB peak allocated GPU memory on one T4. Together with ten
archived ordinary outputs, the checkpoint contains 24 outputs on eight cases from
four papers. No output is empty or token-limited. No optimizer updates occurred.
The ten archived outputs retain their historical text and provenance; this is
not a fresh reproduction of all 24 outputs in one session.

## Provisional automated quality assessment

The three outputs for each case were masked as A/B/C. Source-first judgments and
generic exposition preferences were saved before opening the arm key. The same
reviewer knows the project and some historical outputs, so this is not independent
or human validation. The frozen one-minor-shift rubric was used; equivalent
wording does not count as drift. These are small-sample judgments, not calibrated
semantic error probabilities or statistically established superiority.

| Measure | Ordinary base | Base + structural prior | Existing adapter |
| --- | ---: | ---: | ---: |
| Outputs with material claim issues | 8/8* | 5/8 | 4/8 |
| Eligible: no material issue and at most one minor shift | 0/8 | 3/8 | 3/8 |
| Generic exposition preference vs base: win/tie/loss | — | 6/0/2 | 6/1/1 |
| Exact input copies | 0/8 | 0/8 | 0/8 |
| Median source word-sequence overlap | 0.288 | 0.351 | 0.369 |
| Median reference word-sequence overlap | 0.265 | 0.338 | 0.356 |
| Median output/source word-count ratio | 0.965 | 0.957 | 0.938 |

*One reused ordinary-base insurance output was previously classified as a minor
coverage issue. The new rubric flags the omitted equal-outcomes qualification
as material. Retaining the historical boundary instead gives **7/8** base material
issues; the previous report is preserved. The disagreement and both notes are
saved in the private audit. Neither boundary changes the conclusion that fidelity
remains weak. Other minor/material distinctions also remain judgment-dependent.

The adapter has four outputs without an identified material issue, but its
insurance output has two minor omissions and exceeds the one-minor local budget.
Thus four nonmaterial outputs do not mean four eligible outputs. For the structural
prior, its three nonmaterial outputs all meet that budget. The arms' eligible
cases differ; there is no uniformly best method.

Failure examples include omitted probabilities and supporting gambles, weakened
quantifiers, changed economic roles, and added explanations about uncertainty or
coordination that the source did not assert. The structural prior also adds an
exclusive internal-monitoring purpose in one case. Some adapter outputs preserve
named relationships and organize the argument better, while other outputs retain
these same failure families.

Exact copying is absent, but lexical overlap rises for both alternatives.
Higher reference overlap is descriptive and does not prove style transfer; higher
source overlap is likewise not automatically harmful when technical claims must
survive. The structural prior is static, repeats the source and uses longer
prompts, so its apparent gain does not isolate rhetorical planning. Original
context flags and uncertainties on the frozen cases remain unresolved provenance
limits. No reserved test cases were used.

## Sparse detector stage

The complete export contains forty blinded exact texts: five matched conditions
per case, with no exact duplicates. They total **4,899 whitespace-counted words**.
Including the saved 366-word calibration-pair allowance gives **5,265 estimated
project words**, below the 8,000-word planning ceiling. Provider counting and the
remaining free balance still need checking.

Recommended next action: complete the original pending matched human-control
calibration if feasible, then scan the **first complete four-paper round**:
20 texts, approximately **2,493 words**. Keep the human references, GPT-6 inputs
and all three model arms, including outputs with claim errors. Do not select only
the three eligible outputs or regenerate texts in response to detector scores.
Record native labels, scores, date/version and unavailable statuses. If the
endpoint provides useful discrimination and allowance remains, complete the
second frozen round. A detector that labels all relevant conditions alike cannot
establish improvement; report that limit rather than optimizing against it.

No scans have been made by the tools. Detector benefit, full milestone success
and a distinctive Fama-style gain remain unestablished. Further SFT or preference
training is deferred while this checkpoint is completed.

The current private offline page is
`manual_scans_v1/scan_round_1_pdf_only.html`: twenty first-round texts in frozen
blind order with exact-text copy/download buttons. **The user only downloads
GPTZero basic-scan PDFs. No manual CSV fields, score entry or file renaming is
required.** Default report filenames are accepted. Save reports in
`manual_scans_v1/pdf_reports/`, or tell Codex their existing download location.
The older CSV-entry page remains historical and is superseded by this PDF-only
workflow.

`fama_style.research.checkpoint_detector_pdfs` extracts page one locally using
macOS PDFKit (or optional pypdf elsewhere), matches the complete scanned text
with whitespace normalization only, and writes scores and provenance. Original
PDFs and extraction hashes are preserved. Unknown layouts, incomplete text,
ambiguous matches and conflicting repeat reports are flagged rather than assigned
guessed results. Missing version/date metadata remain explicitly unknown. Equal
PDF bytes are de-duplicated; differing repeat reports never trigger best-score
selection. New imports use fresh versioned output folders.

The page makes no network calls and has no detector login access. Available tools
cannot submit scans through the user's GPTZero browser session; only scanning and
PDF download remain manual. The page was checked for exact text preservation,
all twenty records, no external resources and JavaScript syntax. It is generated
by `scripts/build_detector_scan_page.py` without exposing private paragraphs in
public source.

The PDF importer was replayed against the existing real GPTZero report and
recovered its native label, 3-percent AI probability, date and version with an
exact scanned-text match. Synthetic tests cover missing metadata, wrong or
ambiguous text, arbitrary filenames, duplicate bytes and conflicting scans.
The expanded CPU suite passed 625 tests; no new detector score was observed.

## Artifacts and reproduction

- Raw download: `research_runs/milestone_checkpoint_4b_v1/kaggle_v1/`.
- Exact texts, blind keys, blank score template and budget:
  `research_runs/milestone_checkpoint_4b_v1/export_v1/`.
- Masked judgments and versioned review:
  `research_runs/milestone_checkpoint_4b_v1/review_v1/`.
- Manual scan order and working worksheet:
  `research_runs/milestone_checkpoint_4b_v1/manual_scans_v1/`.
- Public aggregate results: `results.json` beside this report.

The offline review command is:

```sh
PYTHONPATH=src python3 -m fama_style.research.milestone_checkpoint_review \
  --packet data/milestone_checkpoint_4b_v1 \
  --exported research_runs/milestone_checkpoint_4b_v1/export_v1 \
  --judgments research_runs/milestone_checkpoint_4b_v1/review_v1/masked_judgments.private.json \
  --historical research_runs/teacher_pair_reverse_pilot_v2/pilot_review_v1/results/per_output.private.jsonl \
  --output research_runs/milestone_checkpoint_4b_v1/review_v1/results
```

Use a fresh output directory for reproduction; immutable outputs are not
overwritten. Source paragraphs, model generations and private judgments are
excluded from public artifacts. No training or external detector is called.

Validation: 621 CPU tests passed, including evidence-binding, missing-judgment,
meaning-budget and identical-text ranking checks. Review reproduction was
byte-identical. The original training implementation was not changed.
