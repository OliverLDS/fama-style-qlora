# External-realizer check of four frozen packing prompts

**Prepared, not executed.** The [readable-format run](READABLE_PACKING_ABLATION_V1_RESULTS.md) reached the requested sentence counts but missed the claim-bound merge/split. This package tests whether another instruction-following realizer can execute the same tasks before further prompt changes or training.

## Inputs and design

The four inner prompts are copied byte-for-byte from the verified Kaggle run: offset/JSON and readable-binding/direct-plan prompts for the hierarchy merge and market-efficiency split. No source, claim evidence, operation, style instruction or target slot is regenerated. Source excerpts are private; reference paragraphs, annotations for evaluation and old outputs are kept in a separate local key and excluded from generation files.

Each file contains one job with only `job_id`, `prompt_sha256` and `prompt`. One job per chat reduces direct cross-task answer context. Job IDs are opaque, but the prompt formats themselves are visible; this is not a blinded design. Use the same chosen model and visible settings for all four tasks. Model identity and strength are not verified by a product label.

Both cases are previously inspected training examples chosen after earlier outputs. The claim inventory and grouping remain reference-informed oracle information. There is no development/test data or generalization evaluation here. New conversations do not prove independence or eliminate memory effects; no chat deletion or memory-clearing procedure is required.

The frozen inner prompts match Kaggle; the external delivery wrapper, system context, possible source familiarity and hidden model settings do not. Greedy decoding, token budgets and normal termination cannot be assumed. Results would be evidence about this realization setup, not a clean causal test of model size.

## Manual generation

Active package: `research_runs/readable_strong_realizer_v1/package_v2/`.

For each number 01–04, open a separate new chat with the same model/settings:

1. Attach **only** the matching `batches/jobs_XX.jsonl` and `response.schema.json`.
2. Paste `TASK_PROMPT.md` as the request.
3. Save the first returned `responses_XX.jsonl` unchanged in `research_runs/readable_strong_realizer_v1/responses/`.

Record the visible model label/settings and whether separate chats were used in the optional `responses/notes.txt`. Unknown is acceptable; do not ask the model to invent its hidden version or seed. After saving the four files, tell Codex they are ready. No quality-rating form or human review is required.

Do not attach `private_key.private.json`, manifests, reports, reference paragraphs, old outputs, the full directory or earlier chat transcripts. No API key, HF_TOKEN, Kaggle access or GPU is needed for these manual chats. Keep source files and returned paragraphs private. `package_v1` is a preserved local preparation draft; use **package_v2**.

The response schema permits exactly `job_id`, `prompt_sha256` and `output`. The inner paragraph-only instruction applies inside `output`; the outer JSONL wrapper only provides transport. Preserve first attempts, including partial/invalid responses. Do not select a better retry or silently repair JSON or prose.

## CPU import and assessment

```bash
PYTHONPATH=src python3 -m fama_style.rhetoric.readable_strong_check \
  --audit-responses \
  --package research_runs/readable_strong_realizer_v1/package_v2 \
  --responses-dir research_runs/readable_strong_realizer_v1/responses \
  --output-dir research_runs/readable_strong_realizer_v1/response_audit_v1
```

Use a fresh audit directory each time. The importer verifies the package and reviewed parent hashes, exact prompts, job/file identities and full coverage. It rejects invalid UTF-8/JSON, duplicate JSON keys, unknown jobs, wrong response files/hashes, extra fields and empty/nontext outputs. All copies of a repeated job are excluded, without choosing a first/best response. Raw files and optional notes are copied byte-for-byte and hashed. Valid partial coverage remains explicit.

The CPU audit records copying, sentence counts, source/reference surface metrics, style proxies and bounded relation warnings. It does not load a model or perform NLI, infer a hidden cutoff, grade claim drift, certify operation realization or establish author style. Those judgments remain pending a subsequent evidence audit. Count compliance alone is already known to be insufficient.

The decisive comparisons remain:

- **Hierarchy:** Are the organizational examples and informal-monitoring claims kept in their intended separate sentences, with the initial merge and no replacement filler conclusion? The initial merge alone was already present in Gemma's controls.
- **Market efficiency:** Is the cited stock-return finding separated from the author's earnings presumption, with correct attribution, the separate IPO presumption and correction conditions preserved?
- **Tradeoff:** Does actual expression/style improve enough to justify any minor drift? Distinguish worthwhile changes from invented mechanisms, lost conditions, attribution errors or generic expansion. Do not turn NLI or copying into a binary acceptance rule.

If another realizer executes both tasks, the representation is actionable in at least that setup and more varied realization evidence becomes justified. It does not prove model capacity is the sole bottleneck or establish enough data for fine-tuning. If it also fails, revisit the representation before training. No external-realizer outputs or new quality results are claimed yet.

## Reproduction and checks

```bash
PYTHONPATH=src python3 -m fama_style.rhetoric.readable_strong_check \
  --prepare --output-dir research_runs/readable_strong_realizer_v1/package_v2
PYTHONPATH=src python3 -m fama_style.rhetoric.readable_strong_check \
  --check --package research_runs/readable_strong_realizer_v1/package_v2
```

Preparation requires a new directory. It checks the reviewed parent run, log-review flags, all raw artifact hashes and frozen prompt identities. Package hashes also bind the response schema, task prompt and private key. The response parser reuses the existing strict manual-import implementation; the new experiment remains separate.

All **318 CPU tests passed**, including exact one-job prompt transport, source/reference separation, parent identity, raw-response preservation, partial/duplicate/wrong-hash handling, tamper rejection and absence of model/network/training actions. The original `train.py --check` passed. No credentials, model weights, external requests, GPU submissions or training were used. Existing experiments, annotations and all reserved test examples remain unchanged.

Implementation: `src/fama_style/rhetoric/readable_strong_check.py`; tests: `tests/test_readable_strong_check.py`.
