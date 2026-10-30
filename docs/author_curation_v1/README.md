# Author source extraction and curation v1

Completed 30 October 2026. This is a human-source curation release for later
GPT-6 annotation, not a new paired training dataset or training result.
The 13-paper Fama archive and all existing train/development/reserved assignments
remain unchanged. The three non-Fama groups each have 13 source papers.

## Result

| Group | Papers fully extracted | Paragraph candidates for annotation |
|---|---:|---:|
| Robert Shiller | 13 | 368 |
| Holmström/coauthors | 13 | 242 |
| Baker–Wurgler/coauthors | 13 | 432 |
| Total | 39 | 1,042 |

All **1,250 PDF pages** were extracted. Native PDFKit text was used on 922 pages;
**328 pages** received CPU-only macOS Vision OCR. No pages remain unprocessed.
Full page coverage describes extraction, not independent transcription verification.
Every one of the 54,731 nonempty selected extraction lines is accounted for
exactly once in a paragraph block or a reason-coded exclusion.

The release preserves 6,623 blocks: 1,042 mechanically screened paragraph
candidates and 5,581 held blocks. Holds include short fragments, tables, notation,
uncertain boundaries, unresolved hyphenation, low OCR confidence and extraction
artifacts. A further 21,231 individual lines are excluded from prose blocks
with explicit reasons, including headers, footers, references and notes. Their
text is retained privately. The [aggregate report](summary.json) gives counts
and all 39 per-paper outcomes.

**Every candidate remains `needs_review: true` and `training_ready: false`.**
Meaning/context annotation and rewrite generation have not occurred. Candidate
status does not certify economic completeness, correct paragraph boundaries,
perfect transcription, or suitability for model training.

Subsequent annotation exposed four isolated-paragraph holds recorded in a
[versioned overlay](post_annotation_holds_v1.json), including one merged PDF
paragraph boundary. Future selections must respect that overlay; the original
1,042-row candidate snapshot and its hashes remain unchanged.

Some difficult theoretical/older scans yield little under the current filters:
A_S1981 contributes one candidate, B_H1991 five, and B_H1994 three. Equal paper
counts do not produce equal source quality or usable paragraph counts. No weak
rows were promoted to meet a per-paper quota.

## Private outputs

The curated data are under `data/author_style_corpus_v1/`:

- `paragraphs.private.jsonl`: 1,042 paragraph candidates, with source hashes,
  exact authors, style-group IDs, version, PDF page ranges, line references,
  extraction method, context flags and repair history.
- `all_blocks.private.jsonl`: every block before final selection, including holds.
- `holds.private.jsonl`: 5,581 blocks with explicit hold reasons.
- `excluded_lines.private.jsonl`: all line exclusions with source locations.
- `repair_audit.private.jsonl`: normalization operations, unresolved hyphens and
  raw/clean hashes. Raw extraction is preserved in the block records.
- `coverage.private.json`: all 39 papers, page totals, OCR coverage and no remaining pages.
- `duplicate_links.private.json`: the detected duplicate link and similarity.
- `source_registry.private.json`, `protocol.json`, `summary.json`, `manifest.json`:
  source identities, thresholds, aggregate results and reproducibility hashes.

The extraction evidence is under `research_runs/author_curation_v1/`: full native
pages, six OCR task/output batches, both fallback audits, source/environment
metadata, worker logs and retained diagnostic previews. PDFs stay in the existing
private archives. No reference paragraphs, PDFs or per-example texts are public.

## Extraction and repairs

The implementation is [the native/OCR extractor](../../scripts/data/extract_author_pages.swift)
and [the CPU curation module](../../src/fama_style/research/author_curation.py).
It uses installed macOS PDFKit/Vision plus Python’s standard library; no downloaded
model weights, remote annotation API or GPU is required.

PDFKit reads every page and records native line geometry. Pages with absent
or damaged text layers receive OCR with accurate English recognition, language
correction disabled and CPU-only execution. Native text that glued words together
in the older Shiller sources prompted a separately audited OCR pass. No guessed
space insertion was applied to those native layers; the original extraction and
OCR replacement are both retained.

The rotated multitask article B_H1991 uses separate rendering regions for its
two adjacent printed pages and excludes the blank right margin. This prevents
interleaving their prose. Page/column IDs remain part of every line reference.

Repairs are limited to logged Unicode NFKC normalization, whitespace cleanup
and dictionary-supported line-break/soft-hyphen joins. There are 6,654 normalization
events across all blocks, mostly whitespace cleanup; this is not a count of
substantive text corrections. Uncertain hyphenation is held rather than guessed.
Words, quantities, mathematical notation and claims were not reconstructed from
economic knowledge. There are 239 logged adjacent page/column continuation joins;
the rule requires unfinished text, a lowercase continuation, matching section and
no intervening semantic barrier. Other uncertain paragraph boundaries are held.

Headings and backmatter boundaries are separated from prose. The curation rule
distinguishes a true appendix heading from an in-text appendix reference.
Shiller’s 1984 comments/discussion beginning at PDF page 43 are excluded from his
main article. Foreword, metadata, separately authored comments, table/figure notes,
references and mathematical/proof blocks are outside the intended narrative task.
Geometry and pattern filters are imperfect; residual contamination can remain.

Exact and five-word-shingle near duplicates are screened across candidate
paragraphs before any new split assignment. One exact duplicate paragraph was
found at two locations within B_H1991; one copy is retained and the other is held
with its link. No cross-paper candidate pair exceeded the 0.85 Jaccard threshold.
That is not proof of no related passages or paraphrased overlap. Re-run checks
when any held text is repaired or annotation adds variants. Every variant of a
paragraph and every version of a work must remain in the same future partition.

Printed-page fields remain null where not reliably recovered; PDF page and line
locations are explicit. OCR confidence is an engine signal, not a calibrated
probability of transcription correctness. Native confidence is explicitly unmeasured.

## Validation and next step

The CPU tests check conservative hyphen repairs, real versus in-text appendix
references, adjacent continuation joins, section barriers, split discussion
headings, duplicate retention and detection of lost/repeated source lines.
Rebuilding curation from the frozen extraction produces byte-identical files.
Exact repeatability of a fresh macOS OCR run is not asserted.

To reproduce with private inputs present and a fresh output path:

```sh
PYTHONPATH=src python3 -m fama_style.research.author_curation \
  --extraction research_runs/author_curation_v1 \
  --output /tmp/author_style_corpus_reproduction
```

Native and OCR extraction can be repeated with the Swift extractor and the
saved private task JSONs, always using fresh output paths. The extraction
protocol records macOS 12.7.6, Swift 5.7.2, Python 3.9.6, rendering settings
and input hashes; output geometry can depend on the OS/framework version.

Next, select a small training-side batch across papers for the agreed
[direct-rewrite versus neutral-reconstruction comparison](../author_inventory_v1/NEUTRAL_RECONSTRUCTION.md).
First annotate source/context quality, semantic claims and paired rhetorical
structure with automated consistency checks. Quarantine uncertain originals or
missing economic context. New-paper split roles must be fixed before generation
outcome inspection; no reserved paper is repurposed by this release. Do not
fine-tune directly on these source-only records.
