# Input-route × prompt diagnostic: copying is prompt-sensitive, quality remains mixed

Reviewed [`oliverlds/purpose-500-prompt-input-4b`](https://www.kaggle.com/code/oliverlds/purpose-500-prompt-input-4b),
version 1. **The user's prompt hypothesis is supported on this small panel.**
Explicitly requesting different wording and sentence structure substantially
reduces copying by the saved adapter. It does not establish a reliable
posttrained advantage over base Gemma or solve all generation failures.

## Execution checks

All 48 generations completed, with **zero optimizer steps**, on one Tesla T4.
Recorded inference runtime was 11.7 minutes, peak allocated GPU memory 4.48 GiB.
Installation time is outside that recorded runtime. The 24 prompt audits
confirmed no truncation or reference text in generation messages. Model
revision, packages, runtime source, both instructions and greedy decoding match
the frozen packet. Saved adapter hashes remained unchanged and all weights were
frozen. The log passed the credential-pattern screen.

Kaggle SDK metadata confirmed current version 1 before and after download; the
numeric version-label endpoint was rejected, as in the previous download.
All six downloaded files and the receipt are hash-bound. The eight original
simple-prompt controls reproduce their earlier base outputs **8/8** and their
earlier adapter outputs **8/8**, byte for byte. The intervention therefore did
not change those control results.

## Copying: compare each input with itself under the other prompt

The table reports median input/output word-sequence overlap. This is a sequence
similarity statistic, not a percentage of copied words. Each cell contains only
four contents, one per author group.

| Input route | Base: simple | Base: re-express | Adapter: simple | Adapter: re-express |
| --- | ---: | ---: | ---: | ---: |
| Existing GPT-6 paraphrase | 0.265 | 0.217 | **0.934** | **0.343** |
| Existing Gemini paraphrase | 0.299 | 0.202 | **0.924** | **0.306** |
| Fresh Gemini reconstruction | 0.414 | 0.285 | **0.616** | **0.314** |

For the adapter, high-overlap outputs (overlap ≥ 0.8) fall from **7/12 to 1/12**,
and exact word-token copies fall from **2/12 to 0/12**. The revised prompt lowers
overlap on 10 of the 12 paired inputs. It also lowers base-model overlap on
11/12, so the instruction has an effect on both models, with a larger observed
change in the adapted model's near-copy behavior.

This is a useful causal intervention on the instructions for these fixed inputs
and fixed decoding. It is not proof that the original prompt alone caused the
training outcome, that copying disappears in general, or that lower overlap
means better writing. The adaptation still has higher median input overlap than
base in all three routes under the revised prompt.

## Reading the outputs

All 48 outputs were inspected against the actual source they received. The
private notes are unblinded same-agent observations, not independent human
judgments or a calibrated semantic/style score. Modest changes in emphasis or
expression are recorded rather than automatically disqualifying a rewrite.

There are useful changes. The Holmström narrative and Baker–Wurgler results
paragraphs receive substantial reformulation under the revised instruction.
For the fresh Baker–Wurgler input, the adapted revised-prompt output preserves
the large-fund-entry/small-fund-exit distinction, while both base outputs extend
the large-fund advantage to exit timing. That is a local content advantage for
the adapter, although it adds an unwanted heading and stronger robustness
language.

However, failures remain:

- **Fresh Shiller input, simple prompt:** the adapter invents an expectations/
  spending/price explanation and loops until the 512-token output cap.
- **The same input, revised prompt:** it asks for the original passage even
  though the passage is supplied. The very low copying of that response is
  task failure, not a style improvement.
- **Fresh inflation-method input:** base under either instruction and adapter
  under re-expression turn measuring inflation into forecasting prices. The
  prompt change does not resolve that content error.
- Two adapted revised-prompt outputs include a heading or prefatory explanation,
  violating the paragraph-only format. These are retained verbatim for review.

Much of the new language is generic formal academic prose. These four contents
do not demonstrate distinctive Fama/coauthor style or a consistent style
advantage over base. The goal is meaningful rewriting with an acceptable
content tradeoff, not strict identity preservation; repetition, requests for
missing input and unsupported economic mechanisms remain relevant failures
under that more permissive goal.

## What the fresh-input comparison shows

Fresh reconstruction inputs do not uniformly solve the problem. They reduce
median adapter copying under the simple prompt compared with the close
paraphrases, but include the only output-limit failure. The revised prompt
improves overlap statistics across routes yet still fails on the fresh Shiller
case.

Only four underlying contents are represented. Fresh reconstruction changes
the teacher/input route and some information scope, so route differences do not
isolate a single causal factor. References were not supplied to Gemma, and no
failure to recover information absent from a fresh input was counted as losing
a fact supplied to the model. The reserved test set was not opened.

## Next decision

Keep the explicit re-expression instruction as the next **diagnostic prompt**,
while retaining the old instruction as a historical control. Do not treat the
saved adapter as a successful style model, discard the 500-pair collection, or
automatically add training epochs.

The smallest next intervention is an **inference-only adapter-strength sweep**
on these frozen inputs with the revised prompt: base (already available), and
the same saved adapter at fractional and full contribution. This would test
whether partial adaptation retains useful reformulation while reducing the
repetition and instruction-following failures. It would not change the dataset,
require new manual annotations, or perform training. The repository's earlier
SFT-strength experiment changed training on a different small preference-era
dataset; it does not answer this saved-500-adapter question.

If fractional strength offers no meaningful advantage over base, the next
training change needs a controlled hypothesis rather than another data batch.
This recommendation is not an executed experiment. No new GPU job, teacher
inference or detector scan was started during this review, and no new Pangram
packet is requested while visible quality issues remain.

## Private artifacts and checks

`research_runs/data_scaling_v1/purpose_500_prompt_input_result_v1/` stores:

- `raw/`: runtime, execution log and all original result files.
- `download_manifest.json`: version metadata and hashes.
- `reading_notes.private.jsonl`: one output-hash-bound note for each generation.
- `review_v1/`: metrics, paired comparisons, all outputs with reading notes,
  and the review manifest.

`scripts/data/review_purpose_500_prompt_input.py` verifies the frozen packet,
parent adapter, prompt audits and result bindings with CPU-only code. It refuses
to overwrite an existing versioned review. No model weights are loaded during
this verification. Real prose, response bodies and checkpoints stay private.
