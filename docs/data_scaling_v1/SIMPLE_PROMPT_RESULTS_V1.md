# Simpler prompt: less copying, more rewriting, mixed semantic results

Follow-up: the user has now scanned both prompt packets. See the
[completed 32-report Pangram comparison](PANGRAM_PROMPT_COMPARISON_V1.md).
The scan instructions and unobserved-status statements below retain the
pre-scan checkpoint; the follow-up supplies the subsequently observed outcomes.

Run: `oliverlds/corpus-200-simple-prompt-4b`, version 1. All 32 outputs completed:
16 base and 16 using the saved 200-pair adapter. The inference worker took 542
seconds (about nine minutes, excluding notebook dependency setup), with peak
allocated GPU memory of 4.51 GiB on one T4. There were zero training updates.

The saved adapter hashes, source texts, development IDs, base-model revision,
package versions, greedy decoding, EOS/padding configuration and prompt audits
were verified against the previous run and preparation record. No input was
truncated; no output was empty or token-limited. The execution log was retrieved
and contained no matches for the checked credential patterns. Kaggle metadata
confirmed saved version 1 before and after retrieval; the current CLI ignores
the version suffix when listing outputs, so the suffix alone was insufficient.

The only intended experimental change was the [simpler inference instruction](SIMPLE_PROMPT_INFERENCE_V1.md).
The adapter remains the one trained with the stricter instruction.

| Same 16 development passages | Strict base | Simple base | Strict adapter | Simple adapter |
| --- | ---: | ---: | ---: | ---: |
| Median source word-sequence overlap | 0.283 | 0.330 | 0.884 | **0.506** |
| Median reference word-sequence overlap | 0.270 | 0.298 | 0.700 | 0.418 |
| Median longest shared source-word run | 5.5 | 5 | 58 | **7.5** |
| Outputs with source overlap ≥ 0.8 | 0 | 0 | 10 | **0** |
| Exact word-token copies | 0 | 0 | 0 | 0 |
| Outputs with substantive meaning issues identified | 15 | 10 | 3 | 6 |

All 16 adapter outputs have lower source overlap with the simpler prompt. The
large copying reduction therefore occurs throughout this small panel. Lower
reference overlap accompanies it; that statistic cannot determine style quality
or semantic faithfulness. The 0.8 threshold is descriptive, not a success gate.

The meaning counts are provisional same-agent automated judgments, with model
conditions and earlier findings known. The strict-prompt counts come from the
earlier review, not a fresh blind adjudication of all four conditions. Local
wording shifts are recorded separately from substantive issues, and **no meaning
threshold filters the detector packet**. These counts are evidence about the
trade-off, not calibrated error rates or a requirement to restore the old prompt.

Substantive adapter issues include replacing a general-price-level decline with
asset-price declines, changing two relationship-specific assets into one asset,
altering an insurance compensation condition, and removing the condition on a
fund-performance comparison. Some local changes are modest qualifications or
compressions. A five-year/window-index inconsistency already present in both
source and reference is carried forward and is not counted as newly generated
drift. Exact output-bound notes remain private.

The simpler adapter outputs are fluent and perform substantially more rewriting.
They also often resemble the base model's generic academic phrasing. Neither
surface overlap nor this automated reading establishes a human-style detector
benefit or distinctive Fama/coauthor-style learning. The next measurement is the
already planned sparse Pangram comparison, with these meaning notes retained.

## Replacement Pangram rounds

Use the **new** pages under:

`research_runs/data_scaling_v1/corpus_200_simple_prompt_v1/review_v1/pangram_scans_v1/`

| Page | Texts | Estimated credits |
| --- | ---: | ---: |
| `scan_round_1.html` | 8: four matched base/adapter cases | 14 |
| `scan_round_2.html` | 8: four matched base/adapter cases | 14 |

The original eight detector cases remain fixed: two per author group, one per
group in each round, with both outputs included. All selected texts exceed 50
whitespace-counted words; provider word counting and actual remaining quota may
differ. Use the same default settings, copy full texts unchanged, and save PDFs
in the adjacent `pdf_reports/` folder. Default filenames are fine. No CSV fields,
score entry, new GPT-6 batches, or input/reference rescans are requested. Use
round 2 after the ordinary daily refresh. The old strict-prompt rounds remain
on hold. Detector results for these new outputs are not yet available.

## Reproducibility and scope

Private raw files, a version-checked download manifest, text-bound semantic
assessments and their provenance are retained under the new run root.
`scripts/data/review_corpus_200_simple_prompt.py` verifies the execution and
builds `review_v1/`, including the replacement scan packets. Only aggregate
results are copied to the public documentation. Raw paragraphs remain ignored.
Six focused CPU tests pass for prompt bindings, adapter hashes, notebook source,
exact scan text, matched pairing, and retention of unfavorable/short outputs.

This comparison reuses 16 development examples from eight papers within known
author groups. Reserved test data remain unused. No new GPU job, training,
external verifier call or detector scan was initiated during this review.
