# Pre-2020 human-prose annotation pilot

Prepared **100 paragraphs from 20 non-Fama economics/finance papers, representing
33 named authors**, in 20 manual GPT-6 batches. Each paper contributes five
paragraphs. All original works were published in **1937–2019**, using a conservative
cutoff of publication before 1 January 2020. Publication evidence and selected-copy
version dates are in the [source register](human_prose_annotation_v1/SOURCE_DATES.md).
Repository upload, scan and indexing dates are not used as publication dates.

The package supports research on meaning-preserving neutral-to-human academic
prose rewriting. It contains no generated rewrites or annotation responses, and
supplies no evidence yet of writing quality, semantic fidelity, style-transfer
success or detector performance. No model inference, detector calls, training or
GPU jobs were submitted by the builder.

## Active private package and manual workflow

`research_runs/human_prose_annotation_v1/package_v2/`

Start with its `START_HERE.md`. For each batch, paste `batches/batch_XX/TASK.md`
and attach the corresponding `paragraphs.jsonl` plus the two shared JSON files
from `common/`. Each batch requests `responses_XX.jsonl` and `coverage_XX.md`.
The full prompts are prepared; the user generates the responses manually.

| Batches | Purpose | Papers | Paragraphs | Response directory relative to the run |
|---|---|---:|---:|---|
| 01–12 | Training/discovery | 12 | 60 | `responses/training/` |
| 13–16 | Development | 4 | 20 | `responses/development/` |
| 17–20 | Reserved evaluation | 4 | 20 | `responses/reserved_test/` |

All of a selected author's papers remain in one split. The four reserved source
papers were newly added to this package. Fourteen other papers had already been
exposed in earlier comparison-author research. Neither the date cutoff nor the
new-source status establishes absence from language-model pretraining. The
original Fama training/development/reserved split is not repurposed.

The batches mask author names and paper provenance in upload inputs, while the
private key retains attribution, publication evidence, PDF checksums and page/line
locations. Textual clues such as citations are preserved. Separate conversations
reduce direct carryover but do not establish independent annotation contexts;
record the actual visible model, effort setting and conversation arrangement.

## Selection, extraction and annotation limits

The sample begins with within-paper length-strata candidates, followed by a
recorded text-quality review. It is a purposive sample of academic economics,
not a random sample of human prose. Complete paragraphs range from 40 to 265
words. Six older papers extend the previous 14-paper comparison pool. Additional
Coase and Mandelbrot pages were extracted to obtain usable complete prose.

The private audit excludes 33 visibly damaged, merged, mathematical or heavily
context-dependent blocks from selection. Twenty selected rows have 26 limited
extraction corrections, checked against source PDF native text or rendered pages.
These include misplaced OCR letters/words, a missing table number/year-range dash,
and superscript footnote callouts. No original author prose was deliberately
rewritten. Noisy native-layer proposals were not accepted wholesale. Upstream
corpora, PDFs and original text hashes remain unchanged, and every correction has
before/after evidence in the new package. Full transcription is not independently
verified; unresolved limitations remain subject to the later annotation audit.

Each response retains the supplied original exactly and generates a neutral
rewrite, a semantic ledger, the two three-level rhetorical schemas, and finite
claim-bound source-to-original differences. Already-clear and pending-review
outcomes are valid; no wording-distance target is imposed. The paired ledger and
schemas are generated jointly, with explicit uncertainty and exact quote evidence.
They are not independent semantic verification or author-intent annotations.

The 132-category provisional vocabulary is reused without changing IDs,
definitions, applicability or restrictions. A reversible shared-rule view reduces
repeated catalogue boilerplate; CPU checks recover all original category objects
exactly. The catalogue remains proposed and its coverage for these authors is
unproven. Future oracle plans must exclude reference wording, and deployable
constraints must come from the source alone.

All 20 manual response batches have now been received and audited. See the
[results and limitations](human_prose_annotation_v1/AUDIT_RESULTS.md) and
[aggregate counts](human_prose_annotation_v1/audit_results.json). Seven program
repairs are recorded in a derived version without changing raw text. The small
16-output oracle comparison is now complete; see
[realization findings](human_prose_annotation_v1/REALIZATION_RESULTS.md). Guidance
shows no consistent incremental benefit in this low-contrast sample. Semantic
quality remains only partially assessed; no training is authorized or started.
The [five-example grouping diagnostic](human_prose_annotation_v1/GROUPING_RESULTS.md)
is complete: the oracle target plan matches grouping in 5/5 cases versus 1/5
ordinary rewrites. This demonstrates limited grouping control, with more source
wording retained, and does not establish author style or a source-only planner.
The [CPU source-only baseline](human_prose_annotation_v1/SOURCE_ONLY_GROUPING.md)
has now been evaluated on 57 eligible training pairs under whole-paper holdouts:
retrieval matches 22 reference layouts, versus 29 for KEEP. No generation or LLM
training was performed by that check. The subsequent
[nested boundary classifier](human_prose_annotation_v1/BOUNDARY_CLASSIFIER.md)
improves merge F1 to 0.447 but does not establish reliable reference-layout recovery
or generated style gains. A [versioned mismatch audit](human_prose_annotation_v1/BOUNDARY_ERROR_AUDIT.md)
is complete, keeping plausible alternatives, optional reference choices and scope
risks distinct. It changes no labels or scores and proposes a targeted annotation
pilot before broad data expansion.

## Public files and reproduction

- [Manual instructions](human_prose_annotation_v1/START_HERE.md)
- [Annotation prompt](human_prose_annotation_v1/ANNOTATION_PROMPT.md)
- [Output format](human_prose_annotation_v1/OUTPUT_FORMAT.md)
- [Machine-readable output schema](human_prose_annotation_v1/annotation_output.schema.json)
- [Preparation counts](human_prose_annotation_v1/preparation.json)
- [Source metadata](human_prose_annotation_v1/sources.json)

Builder: `src/fama_style/research/human_prose_annotations.py`.
Tests: `tests/test_human_prose_annotations.py`. The builder uses stdlib and local
private source artifacts; no GPU, model weights, credentials or ML imports are
required. Use a fresh output directory for each reconstruction. The manifest pins
registry, OCR, dictionary, taxonomy, code, exclusion/selection review, extraction
repairs and generated packet bytes. Private prose and responses are ignored by
Git; public files contain only methods, schemas, metadata and aggregate counts.
