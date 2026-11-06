# 200-pair pilot: stronger fidelity, substantially more input copying

Run: `oliverlds/corpus-200-pilot-4b`, version 1, supplied 4 November 2026.
All 200 training pairs were used once. The capped run completed successfully;
this is an executed pilot, not a prepared or simulated result.

## Verified execution

- **100 updates**, 200 unique example exposures in the frozen seeded order,
  and 100 finite, nonzero gradient checks.
- **216 label checks**, no truncation, matching prompt/target boundaries and a
  successful manual-versus-built-in masked-likelihood check.
- **32 complete held-out generations**: 16 base and 16 reloaded-adapter outputs,
  with identical inputs, prompts, decoding and pinned base revision. No empty or
  token-limited output.
- Saved adapter hashes match. The language q/v adapter contains 136 F32 tensors
  and 2,228,224 parameters; weights changed and the runtime verified exact reload.
- One Tesla T4; worker elapsed time **1,163 seconds / 19.4 minutes**, with
  **5.41 GiB** peak allocated GPU memory. The worker time excludes setup before
  worker launch; the notebook's outer 90-minute cap included installation.
- Downloaded runtime and configuration match the prepared hashes. Credential
  pattern checks found no matches in the retained log; credential values were
  never printed during review.

The original 200-pair cohort, raw results and earlier experiments are unchanged.
No additional GPU job, inference call or detector scan was launched during review.

## Matched held-out comparison

| Measure, same 16 examples | Base | Epoch 1 |
| --- | ---: | ---: |
| Token-weighted reference NLL | 2.81008 | **0.93173** |
| Outputs with material claim issues identified | 15/16 | **3/16** |
| Within local meaning budget | 1/16 | **12/16** |
| Median source word-sequence overlap | 0.28346 | **0.88419** |
| Median reference word-sequence overlap | 0.27010 | 0.70008 |
| Median longest shared source-word run | 5.5 words | **58 words** |
| Exact source copies / exact word-token copies | 0 / 0 | 0 / 0 |

The meaning budget permits at most one minor shift, no material change and no
unresolved judgment. These are **provisional automated judgments**, not measured
human-review accuracy. A/B outputs were read before the condition-key lookup;
execution metadata and losses were already known, and the conditions are often
inferable from their wording. The same agent performed curation and review.

Twelve cases changed from a material issue to none, three retained material
issues and one had none in either condition. No new material-error case was
identified. One additional trained output has two local qualifier shifts and
therefore falls outside the one-minor budget despite having no material issue.

| Group, four development outputs each | Base material issues | Trained material issues | Trained within budget |
| --- | ---: | ---: | ---: |
| Fama/coauthors | 4 | 1 | 3 |
| Shiller | 4 | 2 | 2 |
| Holmström/coauthors | 4 | 0 | 4 |
| Baker–Wurgler/coauthors | 3 | 0 | 3 |

The remaining trained material issues are a missing asset-selection condition
in the CAPM test discussion, omitted judgments about the current importance and
historical underappreciation of Fisher's theories, and substantial loss of
attribution, timing and detail in the Kohl/Balkans account. The additional
multiple-minor case softens two qualifiers in the fund-performance discussion.
Exact text-bound notes for every output remain private.

## Interpretation

The trained model more often keeps the supplied claims and avoids the base
model's generic additions. It also performs much less rewriting. Ten of sixteen
outputs have source word-sequence overlap of at least 0.8, versus none for base;
this cutoff is descriptive, not a predeclared success criterion. The median
fraction of generated words participating in matching source sequences is
0.911 after training. These measures include normal shared technical wording
and are not plagiarism verdicts.

Lower target NLL and higher reference overlap are encouraging learning signals,
but they do not establish distinctive author-style transformation. The checkpoint
often reproduces the neutral input with small edits. Earlier Pangram results
classified four GPT-6 inputs as Human Written, so lower detectability here could
also reflect input retention. Those old scans do not establish scores for the
new inputs. A Pangram result must be interpreted jointly with copying and the
remaining semantic errors.

The cases come from eight development papers within known author groups. They
are not a new author-held-out or sealed-test benchmark. This panel differs from
earlier small pilots, so cross-run error percentages do not isolate dataset size.
No Fama-specific success, general detector accuracy or production-quality claim
is made.

## Next: scan the new model outputs

The eight cases recorded in the pre-generation config are retained exactly,
with both model outputs for every case. **No training inputs or human references
are being rescanned.** Selection has not been filtered by quality or prospective
detector scores. The selected trained outputs include one material-issue case
and one multiple-minor case; both stay in the analysis.

Private scan folder:
`research_runs/data_scaling_v1/corpus_200_pilot_4b_v1/pangram_scans_v1/`

| Packet | Texts | Estimated credits |
| --- | ---: | ---: |
| `scan_round_1.html` | 8: four matched cases | 15 |
| `scan_round_2.html` | 8: four matched cases | 14 |

Use round 1 when its quota is available and round 2 after the ordinary daily
refresh. Copy the complete texts unchanged and save every report in `pdf_reports/`
beside the pages. Default filenames are fine; no CSV or score entry is needed.
The importer will retain full-text bindings, provider-native labels and shares,
dates, versions and unmatched reports. Do not regenerate or shorten an output
after seeing a detector result.

Pangram documents [20 daily free credits, one per 100 words rounded up](https://www.pangram.com/knowledge-hub/what-is-an-ai-detection-credit)
and a [50-word minimum](https://pangram.cv/knowledge-hub/minimum-and-maximum-input-sizes).
All prepared texts meet that minimum; their lengths range from 62 to 175
whitespace-counted words. Actual provider word counting and remaining balances
may differ. Shorter output length and scans spanning different days are limits
to record rather than silently correct. Detector results remain **unobserved**.

## Saved artifacts and checks

Private run root:
`research_runs/data_scaling_v1/corpus_200_pilot_4b_v1/`

Raw logs, reports, label/gradient audits and adapter are under `raw/` with a
download manifest. `masked_bundle.private.jsonl`, `masked_key.private.json`,
`semantic_assessments.private.jsonl` and `assessment_provenance.json` retain the
review. `review_v1/` contains joined text/assessment records and the verified
summary. Copyrighted paragraphs and private generated texts are not distributed.

Offline builders: `scripts/data/review_corpus_200_result.py` and
`scripts/data/prepare_corpus_200_output_scans.py`. Independent CPU rebuilds
reproduce the review records, scan text files and pages. The
[public summary](stage_0200_results_v1.json) contains aggregates only.
Two focused CPU tests pass for scan-text identity, quota accounting, retention
of flagged outputs and rejection of changed pre-generation case selection.
Neither a larger stage nor an additional epoch is automatically authorized.
