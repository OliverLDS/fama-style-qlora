# Revised-prompt Pangram results: no new human-label conversions

All **19 PDFs** in the supplied folder have been imported. The user's message
said eighteen, but the folder contains all nineteen requested reports, with no
missing scan IDs or duplicate reports. Four exact prior Gemini-input observations
were reused. No published-human scans, new generations or training were requested.

**Reducing copying through the revised instruction did not produce convincing
AI-detection reduction on this panel.** The adapter's only Human Written result
corresponds to an input already classified Human Written. No scored AI-related
input becomes Human Written.

## Paired results

Displayed **human-written document share**, retained without rescaling:

| Content | Gemini route | Input | Base | Adapter |
| --- | --- | ---: | ---: | ---: |
| CPI inflation definition | Existing paraphrase | 100% | 0.1% | 100% |
| Fisher crisis theory | Existing paraphrase | 67% | 0.1% | 0.1%* |
| Factory-modeling example | Existing paraphrase | 1% | 0.1% | 0.1% |
| Fund characteristics | Existing paraphrase | 4% | 0.1% | 0.1% |
| CPI inflation definition | Fresh reconstruction | 0.1% | 0.1% | 0.1% |
| Fisher crisis theory | Fresh reconstruction | 0.1% | 0% | Unavailable: task failure |
| Factory-modeling example | Fresh reconstruction | 0.1% | 0% | 0% |
| Fund characteristics | Fresh reconstruction | 0.1% | 0% | 0.1%* |

These are Pangram authorship shares, **not document AI probabilities**. Native
AI Generated, AI Assisted and mixed labels remain distinct in the imported data.
Small 0% versus 0.1% display differences are retained but are not persuasive
evidence of an improvement when the native classification remains AI Generated.

Across seven scored input/adapter pairs:

- Human-written share increases in **zero**, decreases in four and is unchanged
  in three. One decrease is only a 0.1-point display-floor difference.
- **Zero of six scored AI-related inputs** becomes Human Written. The seventh
  AI-related input has a failed adapter response and no detector score.
- One of seven scored adapter outputs is Human Written, preserving the label
  of the already-human input. This is not a new AI-to-human conversion.
- The adapter has higher human share than base in two pairs: one restores the
  already-human input's label; the other changes 0% to 0.1% while remaining
  AI Generated.

The two input routes behave differently but neither demonstrates the intended
conversion. Existing-input adapter labels are one Human Written and three mixed
AI/AI-assisted. Fresh-input adapter labels are one AI Assisted and two AI Generated,
plus one unscanned task failure.

There are some AI Generated → AI Assisted component shifts relative to base. For
the existing Holmström input, base has 83% AI Generated and 17% AI Assisted, while
adapter has 54% and 46%. For existing Baker–Wurgler, base has 100% AI Generated,
while adapter has 77% AI Generated and 23% AI Assisted. Their human shares stay
at 0.1%, and the actual inputs had 1% and 4% human share. Component shifts should
not be reported as successful human-written classification.

## Scan coverage and protocol differences

The full frozen panel has 24 text slots: **19 new reports + 4 reused input
reports + 1 retained unscanned failure**. New PDFs and reused observations all
display Pangram **4.1**. New reports are dated 6 November 2026; four reused input
reports are dated 5 November. Scores are not necessarily contemporaneous even
though the displayed version matches.

Seventeen new reports match the full frozen scan text after the established
whitespace/visual-reading-order normalization. The two starred results scan
**paragraph bodies only**, omitting model-generated nonparagraph text:

- Fisher, existing Gemini input: the prefatory re-expression explanation is absent.
- Fund characteristics, fresh reconstruction: the “Rewritten Paragraph:” heading
  is absent.

Both bodies otherwise match the frozen outputs completely. A versioned deviation
audit records the removed prefixes and both full-output and scanned-body hashes.
No original model output or report score was edited. Scores are attributed to
the scanned bodies, not silently described as exact full-output scores. No
rescan is requested merely to remove this qualification.

Excluding both body-only reports leaves **five strict full-text triplets**;
there are still **zero new human-label conversions** among their four AI-related
inputs, and no increase in human share over any actual input. The main conclusion
does not depend on including the two deviations.

The fresh Shiller adapter output was a 15-word request for a missing passage.
Its score remains unavailable. It is a failed rewriting case, not a zero score,
a successful low-detection output, or an omitted favorable-denominator case.

## Interpretation and next direction

The previous prompt experiment established a useful intervention on copying.
This checkpoint shows that **more lexical change alone does not deliver the
detector milestone**. In particular, all four fresh Gemini reconstructions are
AI-related by Pangram, so that route supplies measurable detector headroom;
the adapted outputs do not use it successfully in this small panel.

This does not establish that the corpus is unusable, that larger data would
never help, or that Pangram is an objective measure of Fama-specific writing
style. Four contents reused across two routes are exploratory measurements.
Published-human calibration is already available and does not need routine
repetition. Same-agent content notes and visible generation failures remain
separate from detector labels.

Keep the 500-pair data, current adapter, raw responses and all reports. The
proposed next intervention remains a **small inference-only adapter-contribution
sweep** under the revised prompt, using existing inputs. If fractional adapter
contribution offers no quality advantage over base, a controlled training change
is more informative than another large manual annotation batch or more detector
scans. This follow-up is proposed, not executed. No new GPU job has been submitted.

## Evidence and reproducibility

Private root:
`research_runs/data_scaling_v1/purpose_500_prompt_input_result_v1/`

- `pangram_reports_v1/`: all nineteen raw PDFs by hash, complete local PDFKit
  extraction, extraction receipt and file manifest.
- `pangram_import_v1/observations.private.jsonl`: 23 scored observations with
  native labels, displayed shares, dates, versions and text bindings.
- `pangram_import_v1/panel_coverage.private.jsonl`: all 24 intended text slots.
- `pangram_import_v1/protocol_deviations.private.jsonl`: the two body-only audits.
- `pangram_import_v1/paired_results.private.jsonl`: all eight route/content cases,
  including unavailable scores.
- `pangram_import_v1/summary.json` and `manifest.json`: aggregates and provenance.

Importer: `scripts/data/import_purpose_500_pangram.py`. Tests verify that the
failed case cannot acquire a fabricated score, AI Assisted cannot become Human
Written through aggregation, duplicate scores cannot be selected favorably,
and prefix removal cannot conceal substantive paragraph changes. Extraction,
binding and import used local CPU tools; no detector/provider calls were made.
