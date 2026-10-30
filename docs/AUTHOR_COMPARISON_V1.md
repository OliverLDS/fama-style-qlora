# Topic-matched author comparison v1

**A private working corpus and CPU diagnostic are complete.** It contains 146
prose candidates (73 per class) from seven Fama-including training papers and
seven non-Fama comparison papers. Matching is at the paper-topic level, followed
by paragraph length/location matching; these are not meaning-equivalent rewrite
pairs. Original training data and experiments remain unchanged.

The initial diagnostic does not establish an author-distinctive rhetorical
signature. Rhetorical cues classify 9/14 held-out papers correctly, the same as
the lexical baseline. Two descriptive feature patterns warrant independent
replication, but are not validated style rules or generation improvements.

## Sources and matching

| Training paper | Comparison paper | Topic | Matched prose pairs |
|---|---|---|---:|
| Fama 1965 | Mandelbrot 1963 | Speculative-price distributions | 7 |
| Fama 1980 | Holmstrom 1979 | Agency, monitoring and incentives | 9 |
| Fama–Jensen 1983 | Alchian–Demsetz 1972 | Economic organization and control | 12 |
| Fama 1997 | Malkiel 2003 | Market efficiency and anomalies | 12 |
| Fama–French 2000 draft | Baker–Wurgler, accepted proof / published 2004 | Dividend propensity | 12 |
| Fama–French 2003 | Ritter–Welch 2002 | New listings and IPOs | 12 |
| Fama–French 2018 | Pastor–Stambaugh 2011 draft | Long-horizon risk | 9 |

The original comparison papers were downloaded from publicly accessible paper
copies: [Mandelbrot](https://moldham74.github.io/AussieCAS/papers/Mand1963.pdf),
[Holmstrom](https://personal.utdallas.edu/~nina.baranchuk/Fin7310/papers/Holmstrom1979.pdf),
[Alchian–Demsetz](https://competitionandappropriation.econ.ucla.edu/wp-content/uploads/sites/95/2017/08/AlchianDemsetz.pdf),
[Malkiel](https://gceps.princeton.edu/wp-content/uploads/2017/01/91malkiel.pdf),
[Baker–Wurgler](https://www.hbs.edu/ris/Publication%20Files/disdiv_53ec8668-4ce2-4441-9362-3943c147233b.pdf),
[Ritter–Welch](https://site.warrington.ufl.edu/ritter/files/2016/01/A-Review-of-IPO-Activity-Pricing-and-Allocations-2002-08.pdf),
and [Pastor–Stambaugh](https://rodneywhitecenter.wharton.upenn.edu/wp-content/uploads/2014/04/1101..pdf).
Bibliographic metadata and match limitations are in
[the public source register](author_comparison_v1_sources.json).

Topic matching is purposive, not random or exact. The organization pair has an
11-year gap; some other pairs differ in venue, audience, theory/review/empirical
genre or stance. The dividend proof has an inconsistent placeholder header:
2003 is used only as an accepted-year matching/nuisance proxy, not a verified
draft date; publication was in 2004. The long-horizon control is the 2011 revision,
not the 2012 published text. These dating and genre limits remain explicit.

Control authors are screened against Fama, French, Jensen and Blume, the author
surnames represented in the project's source collection. This is not a complete
lifetime coauthor-graph audit. All positive papers include Fama; the experiment
cannot separate Fama's contribution from coauthors or prove an individual style.

## Same extraction path for both classes

Both groups were re-extracted from PDFs. The comparison does not mix existing
ChatGPT-extracted Fama paragraphs with a different control extraction pipeline.
The original frozen training JSONL is not rewritten or augmented.

The extractor uses built-in macOS Vision OCR with accurate English recognition,
CPU-only execution, language correction disabled and 1800-pixel page rendering.
No OCR package, language-model weights, external model or GPU service is required.
Two-column documents are cropped by column; Mandelbrot's gutter received a
recorded layout adjustment. The sandbox initially prevented Vision image-buffer
creation; the successful extraction ran with the requested sandbox exception.

There are 224 processed pages: a fixed cap of sixteen main-text pages per paper
after front matter. This is partial source coverage, not a complete-paper corpus.
Initial eight-page windows yielded too little clean prose in the older mathematical
papers, so the cap was extended uniformly **before classification**. Earlier failed
OCR/layout passes and quality reports are retained and excluded from the release.

Layout-derived candidates retain PDF/page/column/line IDs, OCR confidence, raw
OCR, normalized text, source hashes and transformation logs. Headers, detected
references, small bottom text, captions, math/metadata fragments, uncertain
boundaries, low-confidence lines and unresolved hyphenation are filtered.
Normalization uses Unicode NFKC, whitespace collapse and dictionary-supported
line-break joins, including simple inflections. Every join is logged; unsupported
joins are rejected rather than guessed. No substantive wording is rewritten.

The final pool has 305 eligible candidates. Duplicate text is removed before
matching. A deterministic, no-replacement matcher uses word count and relative
window position only: 25–220 words, maximum length ratio 1.2, position difference
at most 0.55, six to twelve matched pairs per topic. The realized maximum word
ratio is 1.192; average length difference is about three words. No rhetorical
score or classifier output was used to select paragraphs or tune extraction.

**This is a working research corpus, not verified transcription or training-ready
material.** A spot-check of fourteen selected candidates and one visual layout
page found remaining OCR truncations, inline notation and genre differences.
Selected confidence scores are all 1.0 despite observed errors, so confidence
cannot certify accuracy. Unrecognized words and inline notation are flagged.
All rows retain `training_ready: false`; paragraph boundaries and transcription
are not independently verified. These limitations constrain author-style claims.

## Executed CPU diagnostic

One entire topic pair is held out in each of seven folds. Neither paper nor any
of its paragraphs appears in that fold's training data. Exact cross-fold duplicate
text is rejected. Scaling and lexical vocabulary are fitted inside the training
fold; author names and citation years are masked for text features.

Fixed L2 logistic classifiers use paper-balanced weights, 350 gradient steps,
penalty 0.2 and probability threshold 0.5. A paper-balanced multinomial lexical
baseline uses alpha 1. No hyperparameter or feature-panel search was performed
on held-out outcomes. The primary diagnostic used thirty-five small CPU classifier fits; these are
not Gemma/adapter training. A second run under a different Python hash seed
reproduced identical summaries and predictions (another 35 fits, not independent
evidence). All code uses Python's standard library.

| Feature panel | Held-out papers correct | Topic-pair ranking wins |
|---|---:|---:|
| Rhetorical cue proxies | 9/14 | 2/7 |
| Surface structure, without length | 7/14 | 3/7 |
| Rhetorical cues + structure | 9/14 | 4/7 |
| Length, year, author-count and acquisition metadata | 7/14 | 4/7 |
| Content lexical baseline | 9/14 | 5/7 |

A ranking win means the held-out Fama-including paper receives a higher mean score
than its matched control. Paper accuracy and pair ranking tell different stories:
the rhetorical-only panel's 9/14 accuracy conceals weak within-topic ordering.
The nuisance panel includes length/sentence counts, year, author count, column
layout, window position, OCR confidence, normalization count and unrecognized-word
fraction. Its chance-level accuracy does not prove an absence of confounding.
Paragraph accuracy is descriptive only; paragraphs are not independent author
samples. Fourteen papers do not support a confident population-performance claim.

The rhetorical features count explicit connectors, modal/definition/evidence and
attribution cues, first-person forms and initial sentence cues. Surface features
include punctuation, morphology and a simple passive-pattern proxy. They do not
contain topical word identities, but can still reflect topic or genre indirectly.
They are **not** parsed syntax, semantic claims or fresh three-level LLM taxonomies.
The lexical baseline also captures wording style, not exclusively topic.

## Tentative feature associations

Across all seven topic pairs, the Fama group has lower average frequencies of:

- The marker set `which/that/whose/whom` (median paired difference about −0.47 occurrences per 100 words).
- A simple `be + optional adverb + word ending in -ed` pattern (median paired difference about −0.47 per 100 words).

Both also receive negative coefficients in all seven combined-feature training
folds. The marker count includes uses other than relative clauses, and the passive
proxy misses irregular constructions and can count adjectival uses. These results
therefore do **not** establish that Fama uses fewer actual relative clauses or
passives. They are descriptive candidates from multiple inspected features, not
statistical significance tests or independently validated findings.

Other proposed attributes are inconsistent across topics. No rewrite policy,
source-only mapper, taxonomy or training prompt was changed. Turning these counts
into blanket editing rules could remove needed conditions or qualifications.
Before using them for style transfer, confirm the associations with additional
comparison authors/papers and then test their writing benefit independently.

## Private artifacts and reproduction

- `data/author_comparison_v1/`: matched paragraph candidates, eligible pool, rejection ledger, registry, matches, summary and hashes.
- `research_runs/author_comparison_v1/acquisition_v1/`: seven control PDFs, task lists, OCR passes, logs, provenance and quality audit.
- `research_runs/author_comparison_v1/diagnostics_v1/`: fold predictions/models, features, aggregate results and manifest.

PDFs, OCR text, paragraph prose, individual features and predictions stay private.
Public outputs contain methods, bibliography and aggregates only. The publication
audit now recognizes the exact OCR Swift script and checks the new private text
fields for accidental prose publication. Original archive PDF hashes and the
frozen 247-row training hash were verified unchanged. The 21 validation and 45
reserved test examples were not used in this study.

With the private papers, registry, OCR cache and macOS dictionary available:

```sh
PYTHONPATH=src python3 -m fama_style.research.author_comparison_corpus \
  --registry research_runs/author_comparison_v1/registry.private.json \
  --ocr-pages research_runs/author_comparison_v1/acquisition_v1/ocr_release.private.jsonl \
  --output-dir research_runs/author_comparison_v1/corpus_rebuild
PYTHONPATH=src python3 -m fama_style.research.author_comparison_diagnostics \
  --corpus-dir data/author_comparison_v1 \
  --output-dir research_runs/author_comparison_v1/diagnostics_rebuild
python3 -m unittest discover -s tests -v
python3 train.py --check
```

Use fresh output directories. `scripts/data/extract_comparison_ocr.swift` accepts a
private task list with PDF paths, one-based pages, column count and optional gutter
overlap. The final full task list is preserved as `ocr_tasks_release.private.json`.
A new OCR run requires macOS; corpus matching and CPU diagnostics run from cached
JSON on other platforms. Local integration tests skip when private inputs are
absent. No credential, Hugging Face access, external LLM call, GPU job or adapter
training was used for this task.
