# Revised-prompt Pangram checkpoint: Gemini input routes

The user requested detector testing of outputs rewritten from Gemini-generated
inputs. The packet is prepared for manual Pangram scans; **no detector results
are available yet** and no detector calls have been made by the agent.

## Frozen scope and decision value

Use the same four held-out contents and both Gemini input routes from
[`purpose-500-prompt-input-4b`, version 1](PURPOSE_500_PROMPT_INPUT_RESULTS_V1.md):

- Previously used Gemini direct paraphrase.
- Fresh Gemini purpose-only brief reconstruction.

For each route, compare the actual Gemini input, base Gemma output, and the
saved 500-pair adapter output using the **explicit re-expression instruction**.
The simple-prompt outputs and GPT-6 inputs are outside this detector packet.
The instruction is fixed before any new detector score. This yields eight
paired route/content cases from four underlying contents, not eight independent
papers or authors.

The distinct question is whether adapted outputs under the revised prompt
receive less AI-related classification than **their actual Gemini inputs** and
than base outputs. Native AI Generated, AI Assisted and Human Written labels
must stay distinct. Displayed authorship shares are not document AI probabilities.

## Reuse and credit budget

The prior [published-human calibration](STAGE_0500_REFERENCE_CONTROLS_V1.md)
already answers the genre-level reference question: eight published reference
paragraphs were Human Written at 100%, in addition to earlier controls. **No
published-human rescans are requested.**

Four existing Gemini inputs exactly match prior scan-text hashes in the
complete mixed-500 panel. Their raw PDF evidence, extracted evidence and import
manifests were verified. Reuse those observations with their original date and
version (Pangram 4.1, 5 November 2026):

| Existing Gemini input | Previously displayed human-written share |
| --- | ---: |
| CPI inflation definition | 100% |
| Fisher crisis theory | 67% |
| Small factory-modeling programs | 1% |
| Fund characteristics | 4% |

The already-human input is a ceiling/control case, not an opportunity to claim
AI reduction. It remains in the panel; cases were not filtered by favorable
detector headroom.

| Coverage | Texts | New estimated credits |
| --- | ---: | ---: |
| Reused exact input observations | 4 | 0 |
| Round 1: revised-prompt base/adapter outputs from existing Gemini inputs | 8 | 13 |
| Round 2: fresh inputs and their revised-prompt outputs | 11 | 16 |
| Known short task failure, retained without a detector score | 1 | 0 |
| Full panel | 24 | **29** |

Credits use the user's earlier rule of rounding each text up per 100 words.
Provider tokenization/accounting may differ. Each round fits the previously
reported 20-credit allowance; continue after ordinary quota refreshes if needed.
No quota workarounds or paid inference are part of this packet.

The fresh Shiller adapter output is a **15-word request for the missing
passage**, despite receiving a passage. It is retained as a failed rewrite,
not padded for a detector, silently removed from coverage, or counted as a
humanization success. Its detector field will remain unavailable. Comparisons
must report that missing score alongside the task failure.

## Manual workflow

Open:

`research_runs/data_scaling_v1/purpose_500_prompt_input_result_v1/review_v1/pangram_scans_v1/start_here.html`

1. Round 1: eight scans, approximately 13 credits.
2. Round 2: eleven scans, approximately 16 credits.

Copy the full text from each page using **Copy exact text**, using the usual
default Pangram settings. Do not include scan IDs, edit prose, remove headings,
trim prefatory explanations, pad short outputs or combine paragraphs. Formatting
failures remain part of the actual model output.

Export each report as a PDF into the packet's adjacent **`pdf_reports/`** folder.
Default filenames are fine. No CSV fields, renaming or manual scores are needed.
Keep all 19 results regardless of label. When they are ready, the next import
will bind PDFs to complete scan texts, compare native labels/shares, retain the
prior four input scores and report content/readability issues alongside detector
changes.

If the new reports use a different detector version, record that discontinuity
before interpreting changes against historical input scans. Do not silently
pool versions or automatically repeat scans. The existing base/adapted outputs
will still permit a same-round comparison.

The packet preserves all text slots in `PRIVATE_KEY.jsonl`, with separate reused
observations and unscannable-failure files. New scan text files and HTML pages
contain only the 19 needed texts. Texts and raw report evidence remain private.

Builder: `scripts/data/prepare_purpose_500_pangram.py`. CPU tests check exact-score
reuse, conflicting prior observations, retained failures and safe HTML embedding.
The builder refuses to overwrite a frozen packet. No new generation, GPU job,
training or published-human scan was initiated.
