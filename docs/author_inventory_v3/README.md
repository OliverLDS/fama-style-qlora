# Twenty-paper author inventory v3

The selected roster now has **80 distinct works: twenty per group**. The twenty-eight additions were downloaded from public academic sources into the private archive, with URL, byte count and SHA-256 receipts. All source versions precede 2020. Earlier inventory snapshots and split roles remain unchanged.

This is source availability, not eighty paired training papers. Some works yield no eligible paragraph under the current filters. See the [pairing release](../author_pair_expansion_v1/README.md) for selection, holds and the manual GPT-6 step.

[Inventory metadata](inventory.json) · [Declared acquisition plan](acquisition_plan.json)

## Source and version checks

Bibliographic discovery used [Fama’s reading list](https://faculty.chicagobooth.edu/eugene-fama/teaching/readings), [Shiller’s Cowles listing](https://cowles.yale.edu/publications/author/Robert%20J.%20Shiller), [Holmström’s MIT publications](https://economics.mit.edu/people/faculty/bengt-holmstrom/publications), and [Wurgler’s publication list](https://pages.stern.nyu.edu/~jwurgler/main.htm). Local opening pages were checked for titles, author surnames and source years. This does not independently verify the prose.

Downloaded metadata overrides publication-list shorthand where necessary: the survey proof bears 2013, the managerial-variables article is titled *Predicting Returns with Managerial Decision Variables*, and the promises manuscript is a September 1995 draft with a longer title; no journal publication year is asserted for it. The promises listing’s 1998 date is retained in the original acquisition plan. Baker–Wurgler–Yuan author order follows the downloaded paper. These corrections are recorded in the inventory, with the original plan preserved.

Working drafts, survey chapters and journal articles remain distinct genres. Related surveys, earlier governance work and nearby asset-pricing papers require passage-level overlap checks. A second version of a work must never count as a new paper or enter a different split. Exact titles/PDF hashes were screened; there is no claim that all related prose has been detected.

## Added works

### Fama/coauthors

| ID | Source year | Added work |
| --- | ---: | --- |
| E_FF1989 | 1989 | [Business Conditions and Expected Returns on Stocks and Bonds](https://faculty.chicagobooth.edu/eugene-fama/teaching/readings) |
| E_FF1992 | 1992 | [The Cross-Section of Expected Stock Returns](https://faculty.chicagobooth.edu/eugene-fama/teaching/readings) |
| E_FF1993 | 1993 | [Common Risk Factors in the Returns on Stocks and Bonds](https://faculty.chicagobooth.edu/eugene-fama/teaching/readings) |
| E_FF1996 | 1996 | [Multifactor Explanations of Asset Pricing Anomalies](https://faculty.chicagobooth.edu/eugene-fama/teaching/readings) |
| E_FS1977 | 1977 | [Asset Returns and Inflation](https://faculty.chicagobooth.edu/eugene-fama/teaching/readings) |
| E_FJ1983 | 1983 | [Agency Problems and Residual Claims](https://www.journals.uchicago.edu/doi/abs/10.1086/467038) |
| E_FF2004 | 2004 | [The Capital Asset Pricing Model: Theory and Evidence](https://www.aeaweb.org/articles?id=10.1257/0895330042162430) |

### Robert J. Shiller

| ID | Source year | Added work |
| --- | ---: | --- |
| E_S2005 | 2005 | [Behavioral Economics and Institutional Innovation](https://cowles.yale.edu/node/145026) |
| E_S2004 | 2004 | [Radical Financial Innovation](https://cowles.yale.edu/node/145089) |
| E_S2003 | 2003 | [The Invention of Inflation-Indexed Bonds in Early America](https://cowles.yale.edu/node/145192) |
| E_S2001 | 2001 | [Bubbles, Human Judgment, and Expert Opinion](https://cowles.yale.edu/node/145257) |
| E_S1998a | 1998 | [Designing Indexed Units of Account](https://cowles.yale.edu/node/145408) |
| E_S1998b | 1998 | [Indexed Units of Account: Theory and Assessment of Historical Experience](https://cowles.yale.edu/node/145400) |
| E_S1996 | 1996 | [Why Do People Dislike Inflation?](https://cowles.yale.edu/node/145465) |

### Holmstrom/coauthors

| ID | Source year | Added work |
| --- | ---: | --- |
| E_H2017 | 2017 | [Banks as Secret Keepers](https://economics.mit.edu/people/faculty/bengt-holmstrom/publications) |
| E_H2003 | 2003 | [The State of U.S. Corporate Governance: What's Right and What's Wrong?](https://www.nber.org/papers/w9613) |
| E_H1998 | 1998 | [LAPM: A Liquidity-Based Asset Pricing Model](https://www.nber.org/papers/w6673) |
| E_H2000 | 2000 | [Liquidity and Risk Management](https://economics.mit.edu/people/faculty/bengt-holmstrom/publications) |
| E_H1989 | 1989 | [Agency Costs and Innovation](https://www.econstor.eu/handle/10419/94833) |
| E_H1998b | 1995 | [Examples for and Questions About: An Economic (?) Theory of Promises](https://economics.mit.edu/people/faculty/bengt-holmstrom/publications) |
| E_H1988 | 1988 | [Common Agency and Exclusive Dealing](https://economics.mit.edu/people/faculty/bengt-holmstrom/publications) |

### Baker-Wurgler/coauthors

| ID | Source year | Added work |
| --- | ---: | --- |
| E_BW2012 | 2012 | [Global, Local, and Contagious Investor Sentiment](https://pages.stern.nyu.edu/~jwurgler/main.htm) |
| E_BW2012b | 2013 | [Behavioral Corporate Finance: An Updated Survey](https://pages.stern.nyu.edu/~jwurgler/main.htm) |
| E_BW2011 | 2011 | [Benchmarks as Limits to Arbitrage: Understanding the Low-Volatility Anomaly](https://pages.stern.nyu.edu/~jwurgler/main.htm) |
| E_BW2010 | 2010 | [Can Mutual Fund Managers Pick Stocks? Evidence from Their Trades Prior to Earnings Announcements](https://pages.stern.nyu.edu/~jwurgler/main.htm) |
| E_BW2009 | 2009 | [Multinationals as Arbitrageurs: The Effect of Stock Market Valuations on Foreign Direct Investment](https://pages.stern.nyu.edu/~jwurgler/main.htm) |
| E_BW2007 | 2007 | [The Effect of Dividends on Consumption](https://pages.stern.nyu.edu/~jwurgler/main.htm) |
| E_BW2006 | 2006 | [Predicting Returns with Managerial Decision Variables: Is There a Small-Sample Bias?](https://pages.stern.nyu.edu/~jwurgler/main.htm) |

## Private source locations

PDFs and per-download receipts: `archive/author_expansion_v3_20261101/`. Opening checks and acquisition attempt: `research_runs/author_inventory_v3/`. Native/OCR pages and worker logs: `research_runs/author_curation_v2/`. Final paragraph candidates: `data/author_style_corpus_v2_1/`. The initial curation pass is retained in `data/author_style_corpus_v2/`; it is superseded for new selections. No PDFs, source paragraphs or generated annotations are published.
