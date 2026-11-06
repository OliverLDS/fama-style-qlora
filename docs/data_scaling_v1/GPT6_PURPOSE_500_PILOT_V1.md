# GPT-6 purpose-only 500: released pairs and prepared Gemma 3 4B pilot

Status: **prepared, not executed**. No GPU job, provider inference or detector
scan was submitted. The historical mixed-500 experiment is unchanged.

## Dataset and adjudications

All 500 freshly reconstructed GPT-6 paragraphs are retained, 125 for each of
Fama/coauthors, Shiller, Holmström and Baker–Wurgler. Training direction is
**AI prose → human paragraph**. Neither enhanced briefs nor taxonomies are
serialized into model inputs in this experiment.

The reconstructions received topic, purpose, key points and anchors, without
the original paragraph or the brief's relations list. The teacher was reported
as GPT-6 with High effort requested; actual UI settings cannot be independently
verified. Qualification permits the user's stated tradeoff between style change
and moderate semantic variation; it does not certify every claim as equivalent.

The private release is `data/gpt6_purpose_500_v1/`:

- `training_0500.private.jsonl`: all 500 training pairs with original raw responses,
  provenance, historical follow-ups and current eligibility.
- `adjudications.private.jsonl`: all 500 decisions and before/after text hashes.
- `manifest.json`: released files and dependency hashes.
- `upload/`: only the two files to attach as a private Kaggle Dataset.

The user's latest decisions supersede earlier regeneration requests:

| Slot | Released treatment |
| --- | --- |
| P139 | Preserve the first two claims; clarify the last clause as a liquidity premium with higher prices and lower expected returns relative to stocks. This is user-authorized wording, not a verbatim quotation. |
| P400 | Accept GPT-6's prose unchanged under the user's timing interpretation. Retain the earlier context warning in history. |
| P056, P206 | Keep both texts unchanged; prior warnings are nonblocking. |

There are **zero AI-source edits, one human-target edit, and zero discarded
pairs**. Other scope advisories remain in the private history. Earlier candidate
datasets, teacher responses and source registries are not overwritten. P139's
clarification applies only to this release: the existing Gemini target registry
has not been rebound, so that slot is not currently target-identical across tracks.

The local viewer reads the released dataset first and retains older versions.
Click **Reload local data** at `http://127.0.0.1:8765/`. The latest label is
“GPT-6 purpose-only 500 · RELEASED v1 · P139 clarified / P400 accepted”.

## Prepared experiment

Question: can a pooled Gemma student learn useful rewriting from the complete
purpose-only reconstruction cohort using the simple rewrite instruction?

- Notebook: `kaggle/gpt6_purpose_500_4b.ipynb` (self-contained, below 1 MB).
- Fresh `google/gemma-3-4b-it` at revision
  `093f9f388b31de276ce2de164bdc2081324b9767`; no prior adapter continuation.
- Same previously executed NF4/FP32 QLoRA recipe as mixed-500: attention q/v
  projections, rank 8, alpha 16, dropout 0.05, learning rate 2e-5 with linear
  decay, batch size 1 and accumulation 2, one epoch, exactly 250 updates.
- One visible T4 (`CUDA_VISIBLE_DEVICES=0`), 1,536-token context with no
  truncation. The runtime checks all target masks and generation budgets using
  the actual tokenizer before training. CPU checks do not substitute for that.
- The same instruction during training and evaluation:

  > Rewrite the following passage in natural, human-written academic economics prose.
  >
  > Return only the rewritten paragraph.

- Reuse all 24 previous development input variants: 16 existing inputs and
  8 Gemini variants, covering 16 reference contents from 8 held-out papers.
  Paper IDs **and canonical PDF hashes** must be disjoint from training.
- Generate 24 base outputs before training and 24 adapted outputs after saving
  and exactly reloading the adapter. References are used for target loss, never
  inserted into generation prompts. No reserved-test paragraphs are loaded.
- Persist `evaluation.json`, label checks, finite gradients, optimizer logs,
  copying diagnostics, adapter-change/reload checks, runtime and partial state.
  Numerical surface flags are diagnostics, not semantic verdicts.
- Stop after one epoch or the 90-minute notebook cap, including installation.
  No automatic retries or larger training follow-up.

The prior mixed-500 job took about 37 minutes; this is a planning reference,
not a runtime guarantee. Budget up to 90 minutes of one T4 execution. Kaggle may
account for its T4 x2 allocation even though code exposes only GPU 0.

This is pooled academic rewriting, not author-conditioned generation or a
validated Fama-specific style model. It also changes both the training cohort
and training instruction relative to mixed-500, so differences cannot be
attributed to teacher data alone. Development references are reused and some
teacher inputs differ semantically; lower loss or lower copying is not proof
of better style. Compare source, base and adapted output for meaning, readable
transformation and copying before choosing any further training. No new manual
detector scans are requested by this packet.

## Manual Kaggle setup

1. Create a **new private Dataset**, suggested title **GPT6 Purpose 500 v1**, with:
   `upload/gpt6_purpose_500_pairs.jsonl` and
   `upload/gpt6_purpose_500_config.json`.
2. Upload `kaggle/gpt6_purpose_500_4b.ipynb` into a **new private notebook** and
   attach that Dataset. Enable Internet, select **GPU T4 x2**, and enable
   **HF_TOKEN** in Kaggle Secrets before **Save & Run All**.
3. Use a token whose account has accepted Gemma access on Hugging Face. The
   runtime retrieves it from `UserSecretsClient`; credentials are neither
   embedded nor printed. Arbitrary model/provider exception strings are not
   serialized into reports.
4. Download `gpt6_purpose_500_v1/` and the notebook execution log when done.
   Expected outputs include `evaluation.json` and `epoch_1/adapter/`.
   Preserve partial outputs on failure and do not automatically rerun.

The repository builder never uploads or submits the notebook. No changes were
made to the root Gemma 1B CLI or historical notebooks.

## Reproduce CPU preparation

```bash
python3 scripts/build_gpt6_purpose_500.py
python3 scripts/build_gpt6_purpose_500.py --check
python3 -m unittest discover -s tests -p 'test_gpt6_purpose_500.py' -v
node --test tests/review_pairs.test.js
python3 train.py --check
```

An existing release is hash-verified rather than overwritten. Rebuilding a new
version requires an explicit new release directory/protocol. Real paragraphs,
private upload files, responses and checkpoints remain ignored by Git; the
public notebook contains code and frozen file hashes, not the prose dataset.
