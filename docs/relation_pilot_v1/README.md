# Forty-paragraph relation-aware pilot

**Current status:** [Fixed-rule replication completed](LENGTH_RULE_RESULTS.md).
The unchanged 30-word rule makes 19/24 correct decisions, versus 17/24 for KEEP,
on faithful, decisive cases from four additional papers. Its 16 total selections
include six judged readability improvements, five losses (one with a local
shift) and five ties. These are provisional same-model judgments.

The earlier [CPU selector comparison](SELECTOR_RESULTS.md) found that
length-only selection achieved 24/35 correct decisions, versus 17/35 for KEEP
and 8/35 for the learned selector, under whole-paper validation. The result is
conditional on earlier proposed merges and provisional model judgments.

An [additional-paper replication](LENGTH_RULE_REPLICATION.md) used eight
candidate-generation batches, with the cutoff and previous judgments withheld.
All [32 generated candidates passed mechanical checks](LENGTH_RULE_GENERATION_AUDIT.md)
and their [fixed-candidate verification is now audited](LENGTH_RULE_RESULTS.md).
These reuse training-pool neutral text; they are not project-unseen paragraphs.

[Preferred-merge verification](VERIFICATION_RESULTS.md) was audited.
Of 52 earlier merge-preferred candidates, 18 are now judged faithful and better
merged, 24 read better separated and ten are comparable. These are provisional
same-model judgments, not validated training data or evidence of author style.

[Stage 2](STAGE2_AUDIT.md) covers 39 paragraphs and 152 boundaries that pass
mechanical checks. Annotations permit BOTH at 130 boundaries and KEEP at 22,
while preferring MERGE at 52. The later verification remains a same-model
assessment, not independent semantic certification. GPT-6 is confirmed;
effective High/Medium effort in Stage 2 is uncertain.
[Stage 1](STAGE1_AUDIT.md) and its forty originals are preserved, with one pending
rewrite held offline. The workflow below records the original preparation.

The [verification packet](VERIFICATION.md) preserves the thirteen-batch
preparation; [results](VERIFICATION_RESULTS.md) assess fidelity and readability
separately. No training data is automatically approved.

The [boundary-error audit](../human_prose_annotation_v1/BOUNDARY_ERROR_AUDIT.md)
found plausible alternatives among reference mismatches, missed explicit relations
and prospective scope/density risks. This pilot separates faithful neutral
rewriting, source-only permissible grouping and reference-author layout.

Forty complete narrative paragraphs are selected from four new papers, ten each:

| Original paper | Publication | Metadata evidence |
|---|---:|---|
| Modigliani & Miller, *The Cost of Capital, Corporation Finance and the Theory of Investment* | 1958 | [Publisher PDF](https://www.aeaweb.org/aer/top20/48.3.261-297.pdf) |
| Spence, *Job Market Signaling* | 1973 | [Publisher record](https://academic.oup.com/qje/article-abstract/87/3/355/1909092) |
| Grossman & Stiglitz, *On the Impossibility of Informationally Efficient Markets* | 1980 | [Journal issue](https://www.jstor.org/stable/i331305) |
| Hart & Moore, *Property Rights and the Nature of the Firm* | 1990 | [Publisher record](https://www.journals.uchicago.edu/doi/10.1086/261729) |

The cutoff is publication before 2020, not an upload/scan date. The four papers
and seven named authors are distinct from the earlier twenty-paper pilot;
project-author exclusions are also enforced. PDF hashes, publication/version
years, acquisition URLs and printed/PDF page locations are retained privately.
The sources span a purposive, classic economics/finance sample, not a
representative corpus of human writing. They may be highly exposed in pretraining.
They are new discovery sources, not a new reserved test split or proof of
independent model context.

## Selection and extraction

Each paper contributes five candidate continuation paragraphs and five scope/
argument-transition paragraphs. Those are **sampling strata**, not forced MERGE/
KEEP labels. Selection uses source content and extraction quality, without model
outputs or detector scores. Source paragraphs contain 53–236 words; no passage
was truncated to meet a word budget. The length ranges overlap, but there is no
claim of exact topic/length matching or class balance after neutral rewriting.
Some discourse referents depend on neighboring paragraphs; annotations must retain
and flag those gaps rather than fabricate context.

Original PDFs are private in `research_runs/relation_pilot_v1/acquisition_v1/`.
Two scans lack useful native text and another has a noisy text layer. Local macOS
PDFKit/Vision OCR was used with CPU recognition, English language, correction off
and recorded page/column geometry. OCR covers 60 selected pages across the four
papers; it is not full-paper dataset extraction or a claim of exhaustive coverage.
Native extraction, OCR attempts, dictionary-supported line-break joins, excluded
lines, intermediate candidates and PDF page images are preserved.

All forty selections were checked by the coding assistant against rendered
source pages, with publisher PDF text also consulted for metadata/first-page
transcription. This is **not independent verification**. Automatic segmentation
sometimes mixed footnotes into prose, dropped short lines or split a paragraph
at a page/column boundary. These proposals were not accepted wholesale. Twenty-
eight selected records have further transcription/boundary differences relative
to their normalized OCR proposal. Private before/after text, token diffs, reasons,
candidate IDs, PDF/image hashes and locations record those repairs. This count
includes punctuation/glyph cleanup and boundary restoration; it is not a count
of substantive author-wording changes. Unambiguous footnote callouts are removed,
while citations, quantities, printed unusual spelling and inferential qualifiers
are retained. No intended substantive author correction was made.

Some complete paragraphs required restoring lines that a footnote/heading filter
had incorrectly removed. Cross-column continuations were rejoined and merged
OCR blocks were split back into their actual authored paragraphs. No unrelated
authored paragraphs were combined. Abstracts, footnotes, author notes, references,
standalone equations and tables are excluded. Two selected prose examples contain
simple inline numerical payoffs; these remain part of the claims, not rewritten
as different quantities. The original PDFs, earlier corpora and prior labels and
scores stay unchanged.

## Manual workflow

Active private packet:
`research_runs/relation_pilot_v1/package_v1/START_HERE.md`.

**Run Stage 1 only now.** Use GPT-6 with High thinking effort, consistent across
eight batches of five records. For each `stage1/batch_XX/`, paste `TASK.md` and
attach its `paragraphs.jsonl` plus `common/stage1_output.schema.json`. Save
`responses_01.jsonl` through `responses_08.jsonl` in the packet's
`responses/stage1/`, along with any `coverage_XX.md`. Keep the first complete
unedited responses and record retries or changes. Separate chats can reduce
direct carryover but cannot guarantee independence from automatic retrieval.
Do not attach PDFs, source metadata, selection strata, repair logs, prior results
or other batch answers.

Stage 1 requests original-text integrity, a meaning-preserving neutral rewrite,
atomic claims and both versions' exact sentence/claim alignments. A back-reference
is distinct from expressing a full claim. Already-clear and uncertain/pending
cases are valid; there is no wording-distance or sentence-count target. It does
not request preferred grouping or the full three-level taxonomy. This keeps the
pilot focused on the diagnosed boundary problem rather than expanding annotation
complexity. Paired annotation is explicitly not independent semantic verification.

After responses return, audit them before constructing **Stage 2** inputs. The
Stage 2 prompt and output schema are prepared as specifications, but its actual
source-input batches do not exist yet: they must contain the returned neutral
paragraphs. The projection exports only `paragraph_id` and `source_paragraph`.
No originals, target layouts, paired claim propositions/IDs, taxonomy assignments
or sampling preferences are forwarded. The judge extracts fresh source-only
D-claims and sentence bindings, then labels every boundary with a finite relation,
restrictions, permissible KEEP/MERGE/BOTH/UNRESOLVED actions and a **separate**
presentation preference. The original author's layout remains offline.

For a MERGE/BOTH judgment, a concrete candidate sentence is requested rather than
certifying abstract compatibility alone. Candidate fidelity stays `unverified`
until a subsequent evidence check. Pairwise compatibility does not authorize
chaining every accepted merge into a long paragraph sentence. The decision aims
at reasonable, meaningful organization, not merely grammatical semicolon joining.
No desired class balance or exact-reference recovery is imposed.

Although the Stage 2 payload excludes references, GPT-6 annotation/realization,
pretraining and possible cross-chat retrieval limit independence. Payload
separation is an observable procedure, not a guarantee about hidden context.
No human review panel is required. This pilot has not produced validated labels,
new model performance, approved training data or evidence of distinctive author
style/human authorship. No classifier fitting, generation submission, Kaggle run
or LLM training was initiated by the preparation.

## Artifacts and reproduction

Public files contain [counts](preparation.json), [source metadata](sources.json),
[Stage 1 prompt](STAGE1_PROMPT.md), [Stage 1 schema](stage1_output.schema.json),
[Stage 2 template](STAGE2_PROMPT.md) and [Stage 2 schema](stage2_output.schema.json).
Source text, PDFs, repair evidence and responses remain private and Git-ignored.
The blank `SESSION_NOTES.md` is user-editable and excluded from the packet's
file-hash manifest. Model/effort/context will be recorded as reported, without
inventing hidden snapshots or decoding settings.

```sh
PYTHONPATH=src python3 -m fama_style.research.relation_pilot --registry research_runs/relation_pilot_v1/source_registry.private.json --selection research_runs/relation_pilot_v1/selected_originals_v1.private.jsonl --excluded-sources docs/human_prose_annotation_v1/sources.json --output /tmp/relation_pilot
python3 -m unittest discover -s tests -v
```

Use a fresh output directory. The builder verifies source PDF/selected-text hashes,
new paper/author groups, dates, duplicate originals and per-paper strata. It pins
input/code and packet hashes. Private acquisition hashes also pin OCR, native
text, extraction scripts/candidates and page images. CPU tests cover cutoff and
old-source exclusion, immutable originals, duplicate/stratum failures, balanced
five-record packets, mutable session notes, schema abstention/multiple permissible
actions and source-only projection with reference/claim fields deliberately
poisoned. These checks validate transport and isolation, not semantic truth.
