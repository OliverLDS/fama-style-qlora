# Author and paper inventory v1

Inventory and acquisition checkpoint, 30 October 2026. No new annotations,
generations, training pairs or GPU jobs. All previous datasets and split roles
remain unchanged. The [machine-readable inventory](inventory.json) includes
bibliography, author participation, paper roles, PDF hashes and existing pair
counts. It contains no paragraph text.

## What is already here

The project had **37 registered papers by 44 authors**: 13 Fama/coauthor papers,
20 papers in the historical multi-author annotation corpus, and four additional
papers used in the relation pilot. An additional four PDF artifacts were already
excluded for extraction quality or attribution problems. They remain archived,
with their exclusion reasons recorded. A detector-report PDF is evidence rather
than a paper. These categories explain why a raw PDF count exceeds the corpus.

The Fama corpus contains 247 selected training pairs from seven papers, 21
validation pairs from FF1997 and 45 reserved test pairs from FF2014. Four other
archived Fama papers contributed zero selected pairs under the original curation
rules. Having a PDF does not guarantee usable examples. Dates here preserve the
existing source-version metadata; they are not a new verification of every
Fama article's publication year.

The historical non-Fama annotation corpus has 100 selected paragraphs from 20
papers, with one paper per author group. Its current milestone packet contains
58 provisional training pairs from 12 papers and 18 development pairs from four
papers; four pairs are quarantined. Forty-two training candidates still carry
review flags. These are candidate counts, not certified training-ready counts.
The four reserved papers and their 20 paragraph bodies were not opened for this
inventory. The Fama reserved paragraph bodies were also left unopened.

## Initial author roster and completed acquisitions

Start with four style labels, preserving joint authorship explicitly:

| Style label | Previously archived papers | New downloads | Available papers | Existing provisional multi-author training pairs |
|---|---:|---:|---:|---:|
| Fama/coauthors | 13 | 0 | 13 | Separate Fama corpus: 247 selected training pairs |
| Robert J. Shiller | 1 | 3 | 4 | 5, of which 2 carry review flags |
| Bengt Holmström | 1 | 3 | 4 | 5, of which 4 carry review flags |
| Malcolm Baker and Jeffrey Wurgler | 1 | 3 | 4 | 5, all carrying review flags |

This is a practical initial roster based on existing training-side coverage,
economics/finance overlap and repeated authorship. It is not a finding that the
four groups have distinguishable rhetorical styles. A Baker–Wurgler label denotes
their joint prose; it cannot identify either individual's contribution. Fama's
broader coauthor grouping remains the project's explicit convention.

Nine papers were downloaded into the private `archive/author_expansion_20261030/`
folder. Their URLs, version dates and caveats are in the
[acquisition plan](acquisition_plan.json); receipts and opening-page samples stay
under private archive/research-run folders. All nine PDF containers were opened
and their titles checked against native text or opening-page OCR. These checks
do not establish complete extraction, paragraph boundaries or semantic quality.

| Group | New paper | Downloaded version / publication |
|---|---|---|
| Shiller | Do Stock Prices Move Too Much to be Justified by Subsequent Changes in Dividends? | 1981 journal article |
| Shiller | From Efficient Markets Theory to Behavioral Finance | October 2002 Cowles draft; journal publication 2003 |
| Shiller | Speculative Asset Prices | February 2014 revision of the 2013 prize lecture |
| Holmström | Moral Hazard in Teams | 1982 journal article |
| Holmström | Managerial Incentive Problems: A Dynamic Perspective | 1999 journal article; earlier versions are the same work |
| Holmström | Understanding the Role of Debt in the Financial System | January 2015 BIS working paper |
| Baker–Wurgler | Market Timing and Capital Structure | 2002 journal article |
| Baker–Wurgler | A Catering Theory of Dividends | 2004 journal article |
| Baker–Wurgler | Investor Sentiment and the Cross-Section of Stock Returns | 2006 journal article |

Primary bibliographic checks: [Holmström's MIT publication list](https://economics.mit.edu/people/faculty/bengt-holmstrom/publications),
[Wurgler's NYU publication list](https://pages.stern.nyu.edu/~jwurgler/main.htm),
[Shiller's 2003 AEA article record](https://www.aeaweb.org/articles?id=10.1257/089533003321164967),
[the Nobel lecture copy](https://www.nobelprize.org/uploads/2018/06/shiller-lecture.pdf),
and [the 1981 AEA article copy](https://www.aeaweb.org/aer/top20/71.3.421-436.pdf).
Publication/version dates are distinguished from web upload or retrieval dates.
All nine new sources satisfy the existing conservative pre-2020 cutoff.

There are now **46 registered papers**, still by 44 distinct registered authors,
and 51 PDF files including the four excluded artifacts and detector report.
No manual downloads are currently needed. Newly acquired PDFs remain private;
public access does not itself grant redistribution rights.

## What this enables and what remains

Four papers per added group are enough to design a small within-author,
paper-held-out pilot. They are not evidence of enough clean paragraphs for a
full-scale adapter. Keep the original author-held-out pooled milestone separate
from a new author-conditioned corpus. Do not mix old reserved papers into its
development data or relabel an existing development paper as fresh test data.

Next, extract the nine additions with page-grounded provenance, remove headers,
references, tables and mathematical fragments from prose candidates, and check
cross-paper duplicate/near-duplicate passages. The two dividend papers may
share material. Exclude forewords and separately authored discussant comments
from the Holmström debt paper. Record journal/draft/lecture genre and date
differences: these can confound apparent author style.

Assign new paper roles only after source-quality and duplication checks, before
examining generation outcomes. A provisional design can use three works for
training and one different work for development per added group; this is not an
assignment made by this inventory and creates no new sealed test set. Freeze a
separate protocol before author-conditioned modeling. Never count another
version of a work as an independent paper.

Then prepare a small comparison of direct rewriting and
[neutral reconstruction](NEUTRAL_RECONSTRUCTION.md) before scaling annotation.
Use training-side material for method selection. Detector outcomes do not select
the corpus. Supporting nine additional author labels is a later expansion;
there is no need to collect all of them now.

## Reproduce locally

With the private source archives and acquisition receipts present:

```sh
python3 scripts/data/build_author_inventory.py --output /tmp/author_inventory_reproduction.json
```

The builder is offline, uses the standard library and refuses to overwrite its
output. It verifies PDF hashes and records input hashes. It reads nonreserved
pair files and reserved split metadata, never reserved paragraph bodies. New PDF
opening samples are private metadata checks, not a new extraction dataset.
