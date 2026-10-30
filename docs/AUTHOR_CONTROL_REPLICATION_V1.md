# Comparison-author sensitivity check v1

This check revisits two descriptive OCR-derived patterns from the original
comparison of Fama/coauthor papers with other economics papers. It adds one new
comparison paper in each of seven topics, excludes authors used in the first
control set, and keeps the original 73 target paragraphs fixed. It tests
sensitivity to comparison-paper choice; it does not add independent Fama target
papers or establish an author-specific style.

The two frozen proxies are (1) occurrences of `which`, `that`, `whose`, or
`whom` per 100 words, and (2) occurrences of `is/are/was/were/be/been` followed
by an optional adverb and an `-ed` word per 100 words. Expected target-minus-
control direction was negative for both. These are literal string counts: the
first includes complementizers and the second can count adjectives while
missing irregular passives. They are not parsed relative clauses or verified
passive voice.

## Results

| Proxy | Topics with lower target rate (of 7) | Median target-minus-new-control difference per 100 words | Original six-of-seven directional rule | Adequate matching in all topics? |
|---|---:|---:|---|---|
| Relative-clause markers | 6 | −0.538 | Met descriptively | No |
| Passive `-ed` proxy | 5 | −0.469 | Not met | No |

The preregistered six-match minimum was reached in five topics. Price
distributions and agency incentives each yielded only three feasible matches;
the former had seven fixed target anchors and the latter nine. The original
calipers, paragraph-quality filters, and targets were retained. Primary means
use all original anchors as specified; matched-anchor sensitivity is also
reported. The marker proxy points lower in six topics using all anchors, but
only five using matched anchors. One target/control paragraph in the Shavell
sample includes an inline subsection lead-in missed by the frozen segmentation
filter; excluding it leaves both proxies lower in five topics.

Therefore neither proxy passes the full replication gate. The first meets its
raw direction count but fails matching adequacy and matched-anchor sensitivity;
the second also misses its raw direction count. The old comparison set had lower
target rates in all seven topics for both proxies. These results weaken the case
for treating either pattern as a stable Fama-style feature.

## Sources and matching

| Topic | Target | Earlier control | New control | New-control matches / fixed anchors | Main limitation |
|---|---|---|---|---:|---|
| Price distributions | Fama (1965) | Mandelbrot (1963) | Granger–Morgenstern (1962) | 3 / 7 | Spectral/random-walk analysis differs from return-tail emphasis. |
| Agency incentives | Fama (1980) | Holmström (1979) | Shavell (1979) | 3 / 9 | Formal principal-agent analysis; new copy is retypeset. |
| Organization and control | Fama–Jensen (1983) | Alchian–Demsetz (1972) | Hansmann (1988) | 8 / 12 | Five-year gap; law-and-economics focus. |
| Market efficiency | Fama (1997) | Malkiel (2003) | Shiller (1998) | 8 / 12 | Review/survey genre; different emphasis. |
| Dividend propensity | Fama–French (2000 draft) | Baker–Wurgler (2003 accepted version) | DeAngelo–DeAngelo–Skinner (2002 draft) | 6 / 12 | Concentration and propensity are related but distinct questions. |
| New listings | Fama–French (2003) | Ritter–Welch (2002) | Pagano–Panetta–Zingales (1997 draft) | 12 / 12 | Six-year gap; Italian firms versus US firms. |
| Long-horizon risk | Fama–French (2018) | Pastor–Stambaugh (2011 draft) | Boudoukh–Israel–Richardson (2019) | 6 / 9 | Same journal/period, but different long-horizon estimands. |

The complete bibliographic metadata and direct source links are in
[`author_control_replication_v1_sources.json`](author_control_replication_v1_sources.json).

## OCR spot check and limitations

The same local CPU OCR script, dictionary, paragraph filters, length limits,
and position/length calipers were used. A predetermined first occurrence of
each proxy in each of 14 target/new-control papers was checked against the PDF
image (28 crops total). The counted literal expression was visible in all 28
crops. This positive-only spot check does not estimate missed-OCR rate, validate
the complete paragraph transcription, or confirm that the expression represents
the proposed grammatical construction. Paragraph segmentation remains
unverified. No source text was corrected.

The two replacement papers were selected before calculating any new feature
scores. Osborne (1959) was rejected because the scanned copy produced no
eligible paragraphs under the existing punctuation and boundary rules.
Grossman–Hart (1980 draft) yielded three eligible blocks; Ross (1973) yielded
two in its six-page paper. They were replaced by Granger–Morgenstern and
Shavell without loosening filters. A later Hansmann book excerpt contained text
by another author; only the original Yale journal copy was retained. The
pre-feature source-quality amendment and source hashes are recorded privately.

The sample remains small, OCR-derived, topic-matched rather than
meaning-matched, and unbalanced in new-control coverage. It cannot distinguish
author effects from period, venue, discipline, paper argument, or extraction
quality. No p-values, classifier fitting, feature search, validation/test-row
analysis, language-model calls, generation, GPU jobs, or adapter training were
performed. Full private rows, PDFs, OCR, excerpts, and sample identifiers stay
local. No copyrighted paragraphs are included in the public files.
