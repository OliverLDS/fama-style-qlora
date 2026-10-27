# External-realizer mechanism check v1

**Prepared, not executed.** Ten requested rewrites compare ordinary and operation-guided realization on the same four transformation examples plus one preservation control. This is the next diagnostic after [operation realization failed to establish a style gain](OPERATION_REALIZATION_RESULTS.md), not another training run.

## Fixed inputs and purpose

The two batches contain the **exact inner prompts** from A and C of the completed Kaggle experiment, using the same source evidence, source-supported relationships and oracle operation programs. Source text and plans are not regenerated. No reference paragraph, target annotation, target cue list, prior model output, case role or condition label is included as a separate generator field. The structural programs still carry reference-informed oracle information.

There are five jobs per batch, in identical deterministic source order. One batch is ordinary rewriting with shared source evidence; the other adds rhetorical operations. Opaque job IDs and a private mapping keep assessment metadata out of the generation files. This is not genuine blinding: the programs are visible in the prompts and the texts may be recognizable.

Use the same chosen external model and visible settings for both batches. Its exact identity is not known or verified by this package. Do not infer a version or claim it is necessarily stronger than Gemma solely from a product name. The purpose is to find whether another capable realizer can follow the fixed operations at all.

## Manual generation files

Active private package:

`research_runs/strong_realizer_check_v1/package_v1/`

- `TASK_PROMPT.md`: the shared task/delivery request to paste into each conversation.
- `batches/jobs_01.jsonl`: five prompt-only tasks for the first conversation.
- `batches/jobs_02.jsonl`: five prompt-only tasks for the second conversation.
- `response.schema.json`: the strict output contract; also available as a [public schema](strong_realizer_check_v1/response.schema.json).
- `START_HERE.md`: operator instructions; do not upload it.
- `private_key.private.json`: assessment mapping and reference data; **never upload it to the generator**.
- `manifest.json`: provenance/hashes; keep it local.

In a new conversation attach only `jobs_01.jsonl` and the schema, then paste the task prompt. In a separate new conversation with the same chosen model/settings, attach only `jobs_02.jsonl` and the schema and paste the same request. No PDF, reference paragraphs, old chat transcript, Kaggle outputs or adapter is needed. Do not ask the generator to critique/select/rerun its answers.

Save the returned files unchanged as:

`research_runs/strong_realizer_check_v1/responses/responses_01.jsonl`

`research_runs/strong_realizer_check_v1/responses/responses_02.jsonl`

An optional `notes.txt` can record the visible model label, known settings and conversation arrangement. Unknown is acceptable; do not ask a model to invent its hidden version. No credential, API key or `HF_TOKEN` belongs in any of these files.

New conversations reduce direct sharing of generated answers, but do not establish independence or eliminate memory effects. No chat deletion or memory-clearing procedure is required.

## Output and CPU import

Each response line contains only `job_id`, the exact copied `prompt_sha256`, and `output`. The paragraph-only instruction applies inside `output`; the outer JSONL wrapper is a delivery convention. This wrapper and five-job batch context differ from the single-job greedy Kaggle calls. Decoding, tokenizer budgets, hidden context and possible pretraining familiarity are uncontrolled. This is not a clean causal model-size comparison or held-out generalization test.

The importer rejects malformed JSON/duplicate JSON keys, extra fields, wrong batches/hashes, unknown jobs and empty/nontext outputs. Every copy of a duplicated job is excluded; it never chooses the first/best response. It does not strip fences, repair JSON, rewrite answers or fabricate missing jobs. Raw response files are copied byte-for-byte into the audit directory, with hashes, and partial/invalid coverage remains explicit. A syntactically complete import does not establish that generation terminated normally; termination and model identity remain unverified.

CPU-only response audit:

```sh
PYTHONPATH=src python3 -m fama_style.rhetoric.strong_realizer_check \
  --audit-responses \
  --package research_runs/strong_realizer_check_v1/package_v1 \
  --responses-dir research_runs/strong_realizer_check_v1/responses \
  --output-dir research_runs/strong_realizer_check_v1/response_audit_v1
```

Use a fresh output directory for each import. Files remain private. No external request, ML dependency, downloaded model weights or GPU is needed for preparation/checking/import.

The audit reports copying, basic reference/source surface measurements and bounded relation/citation-context warnings. A Roll/Mitchell citation appearing in a proxy-contest sentence is only a co-occurrence warning, not a semantic verdict; a legitimate contrast could also trigger it. Semantic fidelity, rhetorical-operation realization and author style remain pending assessment. No numeric-token match or low copying score is promoted into correctness or a successful style transformation.

## Interpretation

Keep the four transformations and preservation control separate. Compare ordinary versus operation-guided outputs within the external model, then use the saved Gemma outputs as contextual evidence with the transport/decoding caveats above. Inspect claim binding, citation/quantity roles, actual requested reorganization, meaningful style change, copying and verbosity separately. Retain the existing allowance for minor drift when substantial style gains justify it.

A positive signal requires the core requested reorganization and substantial reference-grounded style improvement on at least two transformation cases over the ordinary condition, without more high-drift cases; the preservation control must not introduce attribution/explanation errors. This is a provisional follow-up trigger on reused training examples, not statistical significance. The same user/assistant assessment process is not an independent blinded domain evaluation.

Success would establish that the representation can be actionable for at least one different realization setup; it would motivate investigation of capacity or operation-conditioned supervision. Failure would motivate revising the representation or prompt rather than training a mapper from these five cases. Neither outcome alone proves that model capacity is the sole bottleneck. No such outcome is claimed before responses arrive.

Implementation: `src/fama_style/rhetoric/strong_realizer_check.py`; tests: `tests/test_strong_realizer_check.py`. Package preparation validates the reviewed parent execution and reproduces all ten frozen prompts. No existing dataset, experiment, adapter, development example or reserved test example is modified.
