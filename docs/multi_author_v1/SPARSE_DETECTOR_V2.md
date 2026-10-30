# Sparse external detector evaluation and first report

The user reports a **10,000-word free allowance** and no paid-account budget.
We plan around one such allowance, not additional accounts. GPTZero becomes a
sparse external checkpoint measure, not a frequent epoch, prompt-selection or
debugging metric. The allowance is a user-provided constraint; provider billing
and the account's actual remaining balance are not independently known.

This is a versioned update to the earlier full-calibration plan. The original
v1 packet, 12 controls, six-pair screen and hashes remain intact. Completing all
12 scans is not a new execution prerequisite. The v1 feasibility screen remains
incomplete; no pass or corpus-wide detector-performance claim is made.

## First observed control

Page 1 of the user-supplied report shows:

| Field | Observed value |
|---|---|
| Control origin from the private key | GPT-6 rewrite |
| GPTZero native prediction | **Human Generated** |
| Document AI probability | **3%** |
| Reported provider version | `2026-10-07-base` |
| Report date | 30 October 2026; timezone not shown |
| Submitted text check | Matches the assigned control apart from PDF line wrapping |

The original PDF is retained in Downloads and copied into the ignored run.
Only page 1 is extracted/rendered and interpreted; the second page is not used.
The PDF/page hashes and observation receipt pin the evidence. The UTC scan date,
numeric classification threshold and actual charged words remain unknown.
The original CSV template is unchanged; a derived working CSV records this result.

The 3% value is the provider's document-level AI probability, **not a percentage
of AI-written words**, and does not establish human authorship. This GPT-6 input
already has no binary AI flag to reduce. A future Gemma base output could score
differently, so a raw-input floor does not establish a Gemma-output floor or
invalidate the base-versus-adapter endpoint. One scan establishes neither detector
accuracy nor training effectiveness.

## What to scan next

Scan only the matched human original `DC_76546a44e08b69a7` next, about 185
whitespace-delimited words, and hold the other ten controls for now. Report the
first label, score/scale, date and visible version. A PDF for every scan is
optional; a saved result entry or screenshot is sufficient if fields can be
verified. Never edit or rescan a text to obtain a preferred detector label.

If both members are unflagged, record the pair's binary detection floor. Do not
generalize it to all inputs or retrospectively seek only controls with favorable
separation. Any later additional calibration uses the already fixed controls,
with coverage and human false positives reported.

The initial pair costs about 366 words; all twelve original controls total
about 1,825 words. These are whitespace estimates, not provider billing. The
budget is therefore manageable, but there is no need to spend it on eleven PDF
exports immediately.

## Frozen sparse checkpoint subset

Eight development examples are now fixed privately, **two per development paper**,
before any Gemma outputs exist. Both neutral source and human reference must
contain at least 500 characters, a study length screen rather than a claim about
the provider's minimum. Eligible examples are selected by fixed hash order.
This is a length-selected sample, not a random population sample. Reserved
examples stay sealed; all 18 development examples remain available for local
quality/copying evaluation.

For the initial milestone, preserve matched source, human reference, ordinary
base, structural-prior base and future pooled-adapter ordinary outputs. Freeze
all candidate bytes before a checkpoint scan and use a consistent provider
version/date/rule. Do not scan every epoch or regenerate against detector feedback.

| Estimated allocation | Words |
|---|---:|
| Eight fixed GPT-6 inputs | 1,017 |
| Eight human references | 1,103 |
| Three model arms, assuming reference-length outputs | 3,309 |
| Initial matched control pair | 366 |
| Total with initial pair | **5,795** |
| Total if all twelve calibration controls are eventually scanned | **7,254** |

Plan for at most **8,000 project words**, leaving 2,000 of the reported allowance
as headroom. Model outputs do not yet exist: their lengths and scoreability are
unknown. Before submitting them, count actual text, include calibration usage
and check the actual account balance. If the complete batch does not fit, scan
complete paper-balanced rounds of four examples, then eight; retain every model
condition for those examples. Never trim inputs or select good-looking outputs.
Future additional adapter/author arms need a new budget, not assumed free space.

Reuse an unchanged text's saved scan where valid, keeping date/version visible;
do not present results from differing provider versions as a matched snapshot.
Record too-short, failed, unavailable and unmapped outcomes rather than dropping
them. No alternative accounts, paid access or detector API calls are required
by this preparation.

## Routine development and artifacts

Frequent checks use local copying/length diagnostics, protected-claim checks,
blinded automated readability/meaning assessment and diagnostic target loss.
These remain imperfect proxies, and loss alone does not establish style learning.
Detector results provide sparse external evidence, with fidelity/readability
reported separately. Model execution still needs explicit authorization; no
baseline, adapter, GPU job or provider call was started by this update.

Private receipt, working CSV, sparse subset and budget:
`research_runs/multi_author_milestone_v1/sparse_detector_v2/`.
The PDF and extracted page are in `detector_results_v1/`. Raw report text and
images stay ignored. Public [observation/budget aggregates](sparse_detector_v2.json)
and [protocol](sparse_detector_protocol_v2.json) contain no source prose.
CPU tests cover report/source matching, probability interpretation, native-label
handling without invented thresholds, missing timezone, fixed paper-balanced
selection, original-template preservation and no claimed feasibility from one scan.
