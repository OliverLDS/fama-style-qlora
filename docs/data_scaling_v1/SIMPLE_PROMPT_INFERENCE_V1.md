# Simpler-prompt inference comparison

Status: executed by the user as `oliverlds/corpus-200-simple-prompt-4b`, version 1.
See the [completed comparison and replacement scan rounds](SIMPLE_PROMPT_RESULTS_V1.md).
The instructions below retain the preparation protocol. No new training occurred.

Reuse the base Gemma 3 4B model and the frozen adapter from
`oliverlds/corpus-200-pilot-4b`, version 1. Generate both conditions on the same
16 development passages using the original greedy decoding settings. Only the
inference instruction changes:

```text
Rewrite the following passage in natural, human-written academic economics prose.

Return only the rewritten paragraph.

Passage:
{plain_english}
```

Upload `kaggle/corpus_200_simple_prompt_4b.ipynb` into a new private Kaggle
notebook. Attach the version 1 output of the existing pilot through **Add Input →
Notebook Output**. Enable `HF_TOKEN` in Secrets, Internet, and GPU T4 x2, then
Save & Run All. The code exposes GPU 0 and caps execution at 30 minutes including
dependency installation. No separate Dataset upload is needed.

The original report and adapter hashes are checked before loading the model.
The notebook contains no reference paragraphs; it reads the attached report and
supplies only source text to the model. The runtime retains references for
subsequent evaluation. CPU preflight requires no ML imports or downloaded weights.

After execution, retrieve `corpus_200_simple_prompt_v1/evaluation.json`, its
supporting artifacts and the log. Compare meaning changes, copying and reference
alignment separately. Prepare replacement Pangram packets for the same eight
preselected cases, assessing detector scores alongside meaning and readability.
The existing strict-prompt Pangram rounds are on hold; preserve them as historical
outputs rather than spending scan credits on them now.

The adapter was trained with the original stricter instruction. This inference
check tests its response to a simpler instruction; it does not alter the learned
adapter or establish that copying or style transfer has improved. Reserved test
examples remain unused.
