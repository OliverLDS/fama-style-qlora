# Version 2 failure fixed; real CPU preflight passed locally

Kaggle version 2 failed because of a bug in the preflight implementation. Its
exported tokenizer bundle allowed the failure to be reproduced and fixed locally.
**No further Kaggle run is needed to complete the tokenizer gate.** The failed
Kaggle execution and the successful local execution are kept as separate records.

## Cause and correction

In the pinned Transformers `5.20.1`, `apply_chat_template` defaults to returning
a `BatchEncoding` containing fields such as input IDs and attention masks. The
preflight treated that container as a flat list of integer token IDs. For the
first pair it saw a container length of two instead of the expected 161 prompt
tokens, and the strict boundary check failed before completing a pair.

The fix explicitly passes `return_dict=False` and validates the returned flat
integer list. It does not weaken prefix identity, BOS/end-marker checks,
decoded-response matching, archived token hashes or context limits. No pairs
were removed, no text changed and no alternate tokenizer was substituted.

The original downloaded runtime matches the frozen v1 source hash. Replaying
that exact runtime locally reproduces the first-pair ValueError. The corrected
runtime then completes all seventeen pairs using the same exported assets,
with HF network access disabled and no local credential. Jinja2 was installed
as a local tokenizer dependency; missing Jinja2 was not the Kaggle failure.

## Verified real-tokenizer results

| Check | Result |
| --- | ---: |
| Candidate pairs | 17/17 |
| Chosen/rejected response sequences | 34/34 |
| Chosen full token-ID hashes matching the archived processor output | 17/17 |
| Maximum full sequence | 483 / 1,024 tokens |
| Minimum supervised response length | 49 tokens |
| Prompt masks, shared prefixes and end markers | Pass |
| Model weights loaded / GPU used / optimizer updates | 0 / 0 / 0 |

The completed local run reproduced byte-for-byte in a second fresh output
directory, including encoded sequences, audits and tokenizer assets. A separate
offline review rechecked all sequence hashes, masks, chosen archives, coverage,
group weights and summary bounds. It also verified that all six tokenizer files
match the files exported from the failed Kaggle run.

Local execution used Python `3.11.11`, Transformers `5.20.1`, Hugging Face Hub
`1.33.0`, Tokenizers `0.23.3`, Jinja2 `3.1.6` and SentencePiece `0.2.2` in an
isolated temporary environment. PyTorch is absent in that environment. These are
real tokenizer checks; causal-scoring and DPO loss checks still use synthetic
log-probability values, not measured model scores.

## What this establishes

The input-token/mask gate has passed. It does **not** establish semantic gold,
better writing, actual reference log probabilities, trainer/collator compatibility,
GPU memory fit, finite gradients or a successful preference experiment. Those
remain gates for a separately approved bounded GPU implementation and run.
No model inference, generation, detector call or training was performed here.

The original seventeen candidate pairs remain frozen. Their historical pending
tokenizer fields describe their state when curated; this versioned recovery
audit resolves the input gate without rewriting those records. Training readiness
and GPU authorization remain false.

## Preserved artifacts

- `data/actual_error_preflight_v1/`: original frozen packet, unchanged.
- `data/actual_error_preflight_v2/`: corrected runtime/config lineage; the paired
  JSONL bytes are identical to v1.
- `research_runs/preference_token_preflight_v1/kaggle_v2/`: original failed log,
  runtime, state and credential-free tokenizer bundle.
- `research_runs/preference_token_preflight_v1/local_reproduction_before_fix/`:
  failure replay using the exact old runtime.
- `research_runs/preference_token_preflight_v1/local_recovery_v2/`: successful
  real CPU run, all 34 audits and seventeen encoded pairs.
- `research_runs/preference_token_preflight_v1/recovery_review_v2/`: independent
  artifact-integrity checks, receipts and manifest. This is not independent
  semantic review or a second model judge.
- `research_runs/preference_token_preflight_v1/fix_v2/`: diagnostic record, local
  environment and the prepared notebook before the correction.

Public aggregates: [recovery_results.json](recovery_results.json).
Corrected prepared-packet plan: [plan_v2.json](plan_v2.json).
Runtime: `fama_style.evaluation.preference_token_preflight`.
Offline artifact review: `fama_style.research.preference_preflight_review`.

The canonical `kaggle/preference_token_preflight.ipynb` is regenerated with the
fix and explicitly includes Jinja2. Optional future Kaggle replays must use
the two files in `data/actual_error_preflight_v2/upload/` because the config hash
changed; the Dataset is unnecessary for the already completed local recovery.

To replay locally in the isolated tokenizer environment, use a fresh output:

```sh
PYTHONPATH=src HF_HUB_OFFLINE=1 python3 -m fama_style.evaluation.preference_token_preflight \
  --data data/actual_error_preflight_v2/upload/actual_error_preflight_pairs.jsonl \
  --config data/actual_error_preflight_v2/upload/actual_error_preflight_config.json \
  --tokenizer-dir research_runs/preference_token_preflight_v1/kaggle_v2/actual_error_preflight_v1/tokenizer_assets \
  --run-tokenizer-check --output FRESH_PRIVATE_OUTPUT
```

No credential value is needed or written. Original logs, snapshots, failed runs,
datasets and hashes are retained. No Kaggle job was submitted by the agent.

Validation: 647 CPU tests passed, including a regression for the structured
default return and corruption checks for recovered token hashes, masks, shared
prefixes, archive identities and coverage. The original training CLI dry run
passed before the change; its implementation and the candidate dataset are
unchanged.
