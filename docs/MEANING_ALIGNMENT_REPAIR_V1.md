# Four-pair meaning-alignment repair audit v1

Created a **private four-row replacement overlay** after checking the flagged pairs against their archived paragraphs, nearby training-paper context and PDF evidence. Three cases are clarified. The New Lists case now includes the reference's decline claim but remains context-bound because its comparison baseline is not specified in the focal paragraph. All original author targets and the frozen corpus remain unchanged.

There are **nine logged input edits**. No new reference wording was generated, and no model request or training occurred. These rows are not automatically activated for generation or training.

## Decisions

| Paper / flag | Decision | Repair and boundary |
| --- | --- | --- |
| New Lists 2003: low level versus decline | **Clarified, context-bound** | Restore the explicit decline rather than only a low profitability level. Do not invent a comparison period, monotonic within-period trend or decline magnitude. Keep the focal 1980–2001 period; nearby passages use other period endpoints, which are not silently substituted. |
| Fama–Jensen 1983: institutional/technical scope | **Clarified** | Retain ratification and explain it using the paper's own definition: selecting proposals for implementation. Restore human capital and mutual monitoring, rather than assuming skills/colleague monitoring are exact substitutes. Also restore the focal all-types/common-core characterization and large-organization qualifiers explicitly present in the target. |
| Market Efficiency 1997: confidence in presumptions | **Clarified** | Both earnings presumptions explicitly retain the author's safe-to-assume qualification. The IPO presumption is expressed as the content of that assumption, not merely the possibility of displaying strong earnings. It remains the author's assumption, not an empirical finding attributed to cited researchers. |
| Agency Problems 1980: rental representation | **Clarified** | Restore the human-capital wealth rental and rental rate signaled by the managerial labor market, retaining likely dependence on firm success/failure. The separate management-role and current/future wage arguments remain. No sale, ownership transfer or new compensation mechanism is invented. |

The Fama–Jensen review found additional scope omissions in the same paragraph, so the log includes those explicit corrections as well as the initially flagged terms. Changes are confined to these four pairs. Necessary technical terms may reduce source–target lexical distance; no edit-distance quota or forced synonym substitution is applied.

The New Lists flag is not declared fully context-free. Both the revised input and target leave the decline's comparator unspecified. Its `remaining_context` field preserves that limitation. A standalone future use must respect it rather than fill in a baseline from inference.

## Evidence used

All four paper keys are assigned to **train** in the frozen whole-paper split. Only those papers' archived datasets/PDFs were opened for context. Neither the 21 development rows nor the 45 reserved test rows entered this audit.

The archived files were checked against archive-manifest hashes and curated provenance. Focal paragraphs and selected neighboring records were saved without modification. Relevant source locations are:

- **NL2003:** focal PDF page 20 / printed page 18; nearby discussion and PDF pages 17–20. The focal text explicitly says decline. Context does not justify inventing a more precise comparator for this isolated pair.
- **FJ1983:** focal PDF page 24 / printed page 323; the paper's ratification definition is on PDF page 4 / printed page 303. The gloss is grounded in that definition, not an external synonym dictionary.
- **ME1997:** focal PDF pages 4–5 / printed pages 3–4. Both confidence formulations are present on PDF page 4.
- **F1980:** focal PDF pages 5–6 / printed pages 291–292. Rental wording starts at the end of page 5 and continues onto page 6. Both pages received targeted visual inspection.

Extraction used **native macOS PDFKit through the local Swift toolchain**. This reads the archived PDF text layer; it is not new OCR. Raw extraction, page numbers and hashes are retained privately. The Fama–Jensen layer joins words, and the first Agency Problems page contains OCR spelling noise. Matching ignores spacing/punctuation/case only and cannot guess corrupted letters; the noisy rental passage also received visual inspection. No PDF, archived transcription or original target was edited.

This is an in-session assistant review, not independent semantic validation or full-page visual proofreading of all four documents. Targeted evidence supports the recorded corrections; it does not certify the entire corpus.

## Versioned output and activation policy

Private release directory:

`data/meaning_alignment_repair_v1/`

- `repaired_pairs.private.jsonl`: four revised rows, with their original IDs, original author targets, revised input and regenerated prompt.
- `repair_log.private.json`: exact before/after source spans, replacement text, target evidence, rationale and remaining uncertainty.
- `pdf_evidence.private.json`: anchored selected PDF text and page-text hashes.
- `summary.json` and `manifest.json`: decisions, readiness flags and input/output hashes.
- `README.md`: local overlay instructions.

This is an **overlay of replacements by original ID**, not four additional examples to append. No merged 247-row corpus was emitted. `data/fama_corpus_v1/train.jsonl`, its original prompts, all split assignments, source archives, previous model outputs and result hashes remain intact. The original Fama/coauthor paragraphs are byte-identical in both `completion` and `original_paragraph`.

The prompt field in each repaired row is rebuilt with the existing frozen corpus instruction and its revised source. Prompt format and training implementation are not changed. Target-oriented legacy taxonomy metadata is retained explicitly as historical interpretation of the unchanged target, not represented as a fresh annotation of the revised input.

Every revised row has:

- `source_annotations_status: stale_requires_reannotation`
- `old_source_plan_reuse_allowed: false`
- `training_ready: false`
- `automatic_generation_ready: false`

Old source offsets, claim inventories, confidence tags and plans cannot simply be rebased and claimed valid: the repairs restore semantic distinctions as well as change characters. Future use requires fresh source bindings and plan revalidation, with the NL2003 context flag retained. These fields are audit/activation instructions, not a new universal gate added to every historical trainer; no existing training entry point automatically selects this overlay.

## Checks and reproduction

```bash
PYTHONPATH=src python3 -m fama_style.research.meaning_alignment_repair \
  --specification research_runs/meaning_alignment_repair_v1/specification_v1/repairs.private.json \
  --evidence-dir research_runs/meaning_alignment_repair_v1/evidence_v1 \
  --output-dir data/meaning_alignment_repair_v1
```

The output directory must be fresh. A reproduction should use a new directory and compare hashes/data with the active release. The private specification holds source/target quotations; the public implementation does not distribute the four paragraphs. PDF extraction/render helpers and review-method notes are preserved with the private evidence.

All **337 CPU tests passed**, and the original `python3 train.py --check` passed. Tests cover exact reversible patches, repeated occurrences, unsorted-edit/evidence binding, overlap and unsupported-anchor rejection, changed-parent/held-out rejection, target preservation, regenerated prompts and an equivalent private release rebuild. Normalization tests reject guessed OCR letter corrections. All validation precedes release creation; existing files are never overwritten by the builder.

Implementation: `src/fama_style/research/meaning_alignment_repair.py`; tests: `tests/test_meaning_alignment_repair.py`. The public [aggregate decisions](meaning_alignment_repair_v1_results.json) contain no paragraph text. Repaired pairs, PDF pages, context and detailed logs remain excluded from the public repository selection.

## Implications

This confirms that some apparent reference-style differences carried source information that needed to be explicit. Earlier generations were assessed against the **old inputs**; their prior results are not rescored as if they had received these repaired inputs. No improved generation quality, model accuracy or style gain is claimed from the repair itself.

The next prerequisite is fresh source claim/evidence binding on these four revisions, followed by revalidation of any reused operation plan. Keep the context-dependent pair distinguished. That can be prepared without changing author targets or starting a GPU job. The three richer style-control candidates remain untested, and the ordinary/default prototype remains unchanged. No further generation or training has begun.
