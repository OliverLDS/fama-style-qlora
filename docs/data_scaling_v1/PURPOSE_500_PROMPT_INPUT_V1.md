# Saved adapter diagnostic: input reconstruction and rewrite instruction

Status: **prepared, not executed on Kaggle**. The user authorized a small
inference-only comparison and raised the hypothesis that the rewrite instruction
encourages copying. No further training is part of this experiment.

## Design

Four existing development contents, one per author group, are selected by a
fixed hash of their IDs from the eight contents that already have both GPT-6 and
Gemini input variants. Selection uses neither favorable generation outcomes nor
detector scores. Papers remain disjoint from the 500 training examples.

| Factor | Conditions |
| --- | --- |
| Input route | Existing GPT-6 paraphrase; existing Gemini paraphrase; fresh Gemini brief reconstruction |
| Instruction | Current simple rewrite prompt; explicit request to re-express wording and sentence structure |
| Model | Pinned base Gemma 3 4B; exact saved 500-pair adapter |

This yields **4 × 3 × 2 × 2 = 48 generations**. There are only four underlying
contents; variants are paired observations, not 48 independent examples.
Decoding is unchanged: greedy, maximum 512 new tokens, repetition penalty 1.0.
Changing decoding at the same time would obscure the instruction comparison.

Current instruction:

> Rewrite the following passage in natural, human-written academic economics prose.
>
> Return only the rewritten paragraph.

Explicit re-expression instruction:

> Rewrite the following passage in natural, human-written academic economics prose. Re-express it using different wording and sentence structure instead of copying or lightly editing the original sentences.
>
> Return only the rewritten paragraph.

Neither prompt adds an edit-percentage requirement or a long meaning-preservation
checklist. Copying, readable transformation and substantive content changes will
be assessed afterwards. A copying reduction accompanied by circular prose or
severe omissions is not a successful rewriting result.

## Fresh input preparation

Exactly **two free-tier Gemini 3 Flash requests** generated four briefs and four
reconstructions automatically. The configured model is `gemini-3-flash-preview`,
using the previously user-confirmed free project, low thinking, temperature 0.5,
and no paid fallback. All four returned prose outputs are retained unchanged.
No new manual ChatGPT batch is needed.

Stage 1 reused the version-4 abstraction specification. Gemini returned named
key-point fields instead of a list, and some relation labels did not satisfy the
finite catalog. A recorded mechanical conversion joined each subject heading
and its unchanged value into a note. Raw responses remain unchanged. Relation
annotations remain unqualified metadata; they were not corrected, used to gate
the experiment, or forwarded to the reconstruction call.

Stage 2 received **topic, purpose, key points and anchors**, not the original
human paragraph or relationships. The four reconstructions have new wording,
but also some compression and scope shifts: the Shiller text omits the explicit
anniversary framing; the Holmström example becomes a general modeling account;
the Baker–Wurgler text generalizes a conditional growth-fund finding. These are
retained diagnostic inputs, not certified new training pairs.

This is procedure-matched reconstruction using Gemini, **not an exact GPT-6
teacher replication**. Route comparisons also include content/emphasis changes;
they cannot isolate the reconstruction procedure from teacher identity or
semantic compression. The prompt comparison is cleaner: each input's exact text
is held fixed between its two instruction conditions.

The original human paragraph remains available for later reference alignment
and style inspection, but is never placed in Gemma's generation messages.
For content preservation, judge each output primarily against the actual input
it received. Do not treat failure to recover an omitted autobiographical detail
as proof that Gemma lost a fact it was given.

## What the comparison can establish

- If the re-expression instruction reduces copying without repetitive or
  content-losing output, it is a useful inference intervention on this panel.
- If it helps base and adapter similarly, the prompt contributes to the behavior
  but is not necessarily a problem specific to posttraining.
- If the adapter still copies or collapses under both prompts and fresh inputs,
  input mismatch or the simple instruction alone is insufficient to explain the
  earlier failures. Review training strength next; do not automatically collect
  another large annotation batch.
- If fresh inputs help while close paraphrases remain problematic, that supports
  an input-route sensitivity hypothesis, subject to the teacher/content limits.

No result is presumed. Lower copying alone does not establish Fama-specific
style, detector reduction or preserved meaning. No new Pangram scan packet or
human-reference calibration is requested.

## Manual Kaggle upload

1. Create a **new private Dataset**, suggested title **Purpose 500 Prompt Input v1**,
   from both files in `data/purpose_500_prompt_input_v1/upload/`:
   `purpose_500_diagnostic_inputs.jsonl` and
   `purpose_500_diagnostic_config.json`.
2. Upload `kaggle/purpose_500_prompt_input_4b.ipynb` as a **new private notebook**.
3. Attach the Dataset **and** the saved output of
   `oliverlds/gpt6-purpose-500-4b`, **version 1**, through
   **Add Input → Notebook Output**. The runtime verifies the exact report and
   adapter hashes before model loading.
4. Enable Internet, **HF_TOKEN** in Secrets and **GPU T4 x2**. Code exposes only
   GPU 0. Then **Save & Run All**.
5. Download `purpose_500_prompt_input_v1/` and the execution log. Required results
   include `evaluation.json`, `state.json` and `prompt_audit.json`. Keep partial
   results if the job fails.

The notebook is self-contained and below Kaggle's 1 MB source limit. It freezes
all model parameters, performs no optimizer steps or backward passes, checks
the adapter hashes before and after inference, and records every source route,
instruction and prompt hash. The **45-minute cap includes installation**.
HF_TOKEN comes from Kaggle Secrets and is never embedded or printed. The builder
does not submit a Kaggle job.

## Private artifacts and CPU verification

- `research_runs/data_scaling_v1/purpose_500_prompt_input_v1/`: selection, raw
  teacher responses, enhanced briefs, projection audit, fresh inputs and hashes.
- `data/purpose_500_prompt_input_v1/`: frozen upload packet and manifest.

```bash
python3 scripts/data/prepare_purpose_500_diagnostic.py check
python3 scripts/build_purpose_500_prompt_input.py --check
python3 -m unittest discover -s tests -p 'test_purpose_500_prompt_input.py' -v
```

Existing source data, teacher responses, the saved adapter and earlier results
are preserved. Real prose, response bodies, upload files and checkpoints remain
private; the notebook contains code and hashes only.
