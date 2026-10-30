# Thirteen papers per author group

Source acquisition completed 30 October 2026. This v2 expansion preserves the
v1 inventory, PDF bytes, dataset hashes and all previous split assignments.
It adds 27 PDFs: nine each for Shiller, Holmström and Baker–Wurgler.
No manual downloads are needed. No new training pairs, model calls or GPU jobs.

| Style group | Previously available | Added in v2 | Distinct papers now |
|---|---:|---:|---:|
| Fama/coauthors | 13 | 0 | 13 |
| Robert Shiller | 4 | 9 | 13 |
| Holmström/coauthors | 4 | 9 | 13 |
| Baker–Wurgler/coauthors | 4 | 9 | 13 |

The selected roster has **52 papers**. The full project inventory has **73 registered
papers by 51 authors**, including comparison and discovery papers outside this roster.
There are 78 PDF containers: 73 registered papers, four previously excluded source
artifacts and one detector report. These are source counts, not clean-pair counts.

New files are private under `archive/author_expansion_v2_20261030/`; the nine
v1 additions remain in `archive/author_expansion_20261030/`. Older PDFs retain
their original archive/research-run paths. The [inventory JSON](inventory.json)
maps every paper ID to its actual path, hash, bibliography and old split role.
The [acquisition plan](acquisition_plan.json) records URLs and version caveats.

## Attribution and source checks

Baker and Wurgler are both authors on all 13 group papers: nine have exactly
those two authors and four include additional coauthors. The user approved this
expansion on 30 October 2026. Holmström has five sole-authored papers and eight
coauthored papers; retain these exact groups rather than attributing all wording
to him individually. Shiller is the sole author of the 13 intended main texts.
The 1984 Brookings PDF also includes separately authored comments, which must be
excluded from his target prose.

All 27 new PDF containers were opened. Title/author/date checks used native
opening-page text, OCR where needed, and a rendered-page visual check for the
1991 multitask article. That scan has unequal adjacent page images and blank
margins; it needs layout-aware cropping before paragraph extraction. The 1994
incentive-system article required a third-page OCR date check. These checks do
not establish full-text completeness, paragraph alignment or semantic quality.

Dates refer to the downloaded paper version, not the upload date. For example,
the dividend-reference-point paper is a 2012 draft of a 2016 publication, the
bank-capital paper is a 2013 draft related to the 2015 AER article, and the new
Holmström lecture copy is dated 2016 rather than the 2017 AER typesetting.
All new downloaded versions and associated publication dates are before 2020.
A draft and its published version count as one work, even when their titles differ.

Primary bibliography sources: [Shiller at Yale/Cowles](https://cowles.yale.edu/publications/author/Robert%20J.%20Shiller),
[Holmström at MIT](https://economics.mit.edu/people/faculty/bengt-holmstrom/publications),
and [Wurgler at NYU](https://pages.stern.nyu.edu/~jwurgler/main.htm). Individual
download/evidence URLs and exact local PDF hashes are retained in the inventory.
Some entries are working papers, surveys or prize lectures; genre must remain
explicit rather than treating all as comparable empirical journal articles.
Public access is not a license to redistribute the PDF or its reference paragraphs.

## Paper lists

V1 roles are retained. Every new acquisition remains unassigned; this inventory
does not move existing development/reserved papers into training or create a
new held-out evaluation. Publication years for older Fama sources were not
newly researched; the table reports their existing version years.

### Fama/coauthors

| Paper ID | Downloaded version year | Title | Authors |
|---|---:|---|---|
| F1963 | 1963 | Mandelbrot and the Stable Paretian Hypothesis | Eugene F. Fama |
| F1965 | 1965 | The Behavior of Stock-Market Prices | Eugene F. Fama |
| FB1966 | 1966 | Filter Rules and Stock-Market Trading | Eugene F. Fama, Marshall E. Blume |
| F1970 | 1970 | Efficient Capital Markets: A Review of Theory and Empirical Work | Eugene F. Fama |
| F1980 | 1980 | Agency Problems and the Theory of the Firm | Eugene F. Fama |
| FJ1983 | 1983 | Separation of Ownership and Control | Eugene F. Fama, Michael C. Jensen |
| FF1988 | 1988 | Permanent and Temporary Components of Stock Prices | Eugene F. Fama, Kenneth R. French |
| FF1997 | 1997 | Taxes, Financing Decisions, and Firm Value | Eugene F. Fama, Kenneth R. French |
| ME1997 | 1997 | Market Efficiency, Long-Term Returns, and Behavioral Finance | Eugene F. Fama |
| FF2000 | 2000 | Disappearing Dividends: Changing Firm Characteristics or Lower Propensity to Pay? | Eugene F. Fama, Kenneth R. French |
| NL2003 | 2003 | New Lists: Fundamentals and Survival Rates | Eugene F. Fama, Kenneth R. French |
| FF2014 | 2014 | A Five-Factor Asset Pricing Model | Eugene F. Fama, Kenneth R. French |
| VL2018 | 2018 | Volatility Lessons | Eugene F. Fama, Kenneth R. French |

### Robert J. Shiller

| Paper ID | Downloaded version year | Title | Authors |
|---|---:|---|---|
| A_S1981 | 1981 | Do Stock Prices Move Too Much to be Justified by Subsequent Changes in Dividends? | Robert J. Shiller |
| B_S1984 | 1984 | Stock Prices and Social Dynamics | Robert J. Shiller |
| R_S1998 | 1998 | Human Behavior and the Efficiency of the Financial System | Robert J. Shiller |
| A_S2002 | 2002 | From Efficient Markets Theory to Behavioral Finance | Robert J. Shiller |
| B_S2007a | 2007 | Low Interest Rates and High Asset Prices: An Interpretation in Terms of Changing Popular Economic Models | Robert J. Shiller |
| B_S2007b | 2007 | Understanding Recent Trends in House Prices and Home Ownership | Robert J. Shiller |
| B_S2008 | 2008 | Derivatives Markets for Home Prices | Robert J. Shiller |
| B_S2011 | 2011 | Irving Fisher, Debt Deflation and Crises | Robert J. Shiller |
| B_S2013 | 2013 | Reflections on Finance and the Good Society | Robert J. Shiller |
| A_S2014 | 2014 | Speculative Asset Prices | Robert J. Shiller |
| B_S2014 | 2014 | Why Is Housing Finance Still Stuck in Such a Primitive Stage? | Robert J. Shiller |
| B_S2017 | 2017 | Narrative Economics | Robert J. Shiller |
| B_S2019 | 2019 | Narratives About Technology-Induced Job Degradation Then and Now | Robert J. Shiller |

### Holmstrom/coauthors

| Paper ID | Downloaded version year | Title | Authors |
|---|---:|---|---|
| C_H1979 | 1979 | Moral Hazard and Observability | Bengt Holmstrom |
| A_H1982 | 1982 | Moral Hazard in Teams | Bengt Holmstrom |
| B_H1987 | 1987 | Aggregation and Linearity in the Provision of Intertemporal Incentives | Bengt Holmstrom, Paul Milgrom |
| B_H1991 | 1991 | Multitask Principal-Agent Analyses: Incentive Contracts, Asset Ownership, and Job Design | Bengt Holmstrom, Paul Milgrom |
| B_H1994 | 1994 | The Firm as an Incentive System | Bengt Holmstrom, Paul Milgrom |
| B_H1996 | 1996 | Private and Public Supply of Liquidity | Bengt Holmstrom, Jean Tirole |
| B_H1997 | 1997 | Financial Intermediation, Loanable Funds, and the Real Sector | Bengt Holmstrom, Jean Tirole |
| B_H1998 | 1998 | The Boundaries of the Firm Revisited | Bengt Holmstrom, John Roberts |
| A_H1999 | 1999 | Managerial Incentive Problems: A Dynamic Perspective | Bengt Holmstrom |
| B_H2001 | 2001 | Corporate Governance and Merger Activity in the U.S.: Making Sense of the 1980s and 1990s | Bengt Holmstrom, Steven N. Kaplan |
| B_H2010 | 2010 | A Theory of Firm Scope | Oliver Hart, Bengt Holmstrom |
| A_H2015 | 2015 | Understanding the Role of Debt in the Financial System | Bengt Holmstrom |
| B_H2017 | 2016 | Pay for Performance and Beyond | Bengt Holmstrom |

### Baker-Wurgler/coauthors

| Paper ID | Downloaded version year | Title | Authors |
|---|---:|---|---|
| B_BW2000 | 2000 | The Equity Share in New Issues and Aggregate Stock Returns | Malcolm Baker, Jeffrey Wurgler |
| A_BW2002 | 2002 | Market Timing and Capital Structure | Malcolm Baker, Jeffrey Wurgler |
| B_BW2003a | 2003 | When Does the Market Matter? Stock Prices and the Investment of Equity-Dependent Firms | Malcolm Baker, Jeremy C. Stein, Jeffrey Wurgler |
| B_BW2003b | 2003 | The Maturity of Debt Issues and Predictable Variation in Bond Returns | Malcolm Baker, Robin Greenwood, Jeffrey Wurgler |
| C_BW2003 | 2003 | Appearing and disappearing dividends: the link to catering incentives | Malcolm Baker, Jeffrey Wurgler |
| A_BW2004 | 2004 | A Catering Theory of Dividends | Malcolm Baker, Jeffrey Wurgler |
| A_BW2006 | 2006 | Investor Sentiment and the Cross-Section of Stock Returns | Malcolm Baker, Jeffrey Wurgler |
| B_BW2007 | 2007 | Investor Sentiment in the Stock Market | Malcolm Baker, Jeffrey Wurgler |
| B_BW2009 | 2009 | Catering Through Nominal Share Prices | Malcolm Baker, Robin Greenwood, Jeffrey Wurgler |
| B_BW2012a | 2012 | Comovement and Predictability Relationships Between Bonds and the Cross-Section of Stocks | Malcolm Baker, Jeffrey Wurgler |
| B_BW2012b | 2012 | Dividends as Reference Points: A Behavioral Signaling Approach | Malcolm Baker, Jeffrey Wurgler |
| B_BW2012c | 2012 | The Effect of Reference Point Prices on Mergers and Acquisitions | Malcolm Baker, Xin Pan, Jeffrey Wurgler |
| B_BW2013 | 2013 | Would Stricter Capital Requirements Raise the Cost of Capital? Bank Capital Regulation and the Low Risk Anomaly | Malcolm Baker, Jeffrey Wurgler |

## Next data work

Extract and check the new sources before preparing annotation batches. Keep
page locations, exclude comments/forewords/tables/references from narrative
targets, and audit cross-paper overlap before assigning new paper roles.
Related dividend/sentiment papers and survey summaries particularly need
near-duplicate checks. Equal paper counts do not guarantee equal usable prose;
balance training by curated paragraph counts and paper contribution later.

Retain direct rewriting and the proposed [neutral reconstruction comparison](../author_inventory_v1/NEUTRAL_RECONSTRUCTION.md).
The new PDFs have not been annotated and cannot yet expand the training set.
Reserved text remains unopened; source availability does not authorize training.

## Reproduce

```sh
python3 scripts/data/expand_author_inventory.py --output /tmp/author_inventory_v2.json
```

This offline builder reads metadata and hashes the source PDFs. It verifies
all 13-paper counts, exact title/PDF deduplication, both Baker–Wurgler names,
pre-2020 new-source dates and receipt/check hashes. Full paragraph duplication
and economic equivalence are later curation checks. The private inventory
snapshot and its fresh reproduction were byte-identical.
