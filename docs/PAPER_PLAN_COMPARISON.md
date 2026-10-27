# Six-paper ordinary-versus-plan comparison

**Prepared, not executed.** Twelve manual external-model outputs will compare ordinary rewriting with readable edit guidance. This supplies the missing no-plan baseline after [both prompt formats succeeded on two elementary edit tasks](READABLE_STRONG_REALIZER_V1_RESULTS.md).

## Matched inputs

Both conditions receive byte-identical concrete style instructions, full source text, readable sentence/claim evidence, protected evidence, relation annotations/certainty and common composite-claim/pronoun instructions. B alone appends the existing MERGE/SPLIT/KEEP program and its numbered output slots.

| Condition | Input |
| --- | --- |
| A | Ordinary rewriting with shared readable source evidence; no edit program, desired sentence count or target grouping |
| B | The same input plus the readable edit program |

The ordinary arm is **not a bare-text baseline**. Its claim/evidence inventory comes from paired annotations and can itself help a capable model. This check measures the incremental value of the target edit plan beyond that shared scaffold. Original author wording and private evaluation metadata are excluded from generation files. The source itself may naturally contain citations; those remain unchanged.

## Six-paper selection

| Paper key | Role | Planned local operation |
| --- | --- | --- |
| FJ1983 | Prior positive anchor | Hierarchy/process merge |
| ME1997 | Prior positive anchor | Cited finding versus author-presumption split |
| FF2000 | Cross-paper extension | Merge already distinct claims |
| NL2003 | Cross-paper extension | Merge difference and interpretation |
| F1980 | Cross-paper extension | Two local merges with other claims retained |
| VL2018 | Cross-paper extension | Merge volatility and horizon-risk claims |

All are previously inspected **training** paragraphs. The two anchors are forced. For each remaining paper, selection minimizes existing compiled-prompt length, with case-ID ties; the two batches share the same opaque hash order. The diagnostic dividend probe is excluded. Selection is not based on ranking these new outputs by quality; anchor selection is explicitly retrospective. The four extensions have not appeared in the immediately preceding external-model comparison, but are not unseen or held-out data.

Programs are rebuilt through the unchanged evidence validator and checked against their compiled originals. Source/reference paragraphs, annotations, claim IDs, semantic protections, original author targets and training gates remain unchanged. No new source extraction, taxonomy annotation, neutral paraphrase or synthetic author-style target is created. Only one selected case is a split; this is not balanced operation coverage or sufficient mapper-training data.

## Manual generation: two separate chats

Active package: `research_runs/paper_plan_comparison_v1/package_v1/`.

1. In a **new chat**, select **GPT-6, High thinking effort**. Attach only `batches/jobs_01.jsonl` and `response.schema.json`, then paste `TASK_PROMPT.md`.
2. In a **different new chat** with the same model/visible setting, attach only `batches/jobs_02.jsonl` and the schema and paste the same task prompt.
3. Keep the first returned `responses_01.jsonl` and `responses_02.jsonl` unchanged. Save them in `research_runs/paper_plan_comparison_v1/responses/` and tell Codex they are ready.

Each batch has six jobs and requires six JSONL records. Do not show either batch's answers to the other conversation. Do not combine both conditions into one chat. This separates direct cross-condition answer context, but the six jobs within each chat still share context. Chat memory, hidden system instructions/decoding, source familiarity and within-batch interactions remain uncontrolled. Separate chats do not prove independence; no deletion or memory clearing is required. This is not twelve independent single-job trials or a causal model-size study.

Record actual model/settings/chat arrangement in optional `responses/notes.txt`; the planned GPT-6 High label is not a claim about already generated results. No API key, HF_TOKEN, Kaggle notebook, adapter or GPU is needed. Keep all source and response files private. Do not upload the private key, manifest, reports, reference paragraphs, old outputs, whole package or chat transcripts. No human scoring form is required.

## Assessment and decision

The primary question is whether **guidance adds useful organization or expression over ordinary rewriting**, with an acceptable drift tradeoff. Compare actual outputs in pairs, not just whether B obeys a plan. A is not an instruction-following failure because it does not reproduce an edit it was never given.

Report the two anchors separately from the four cross-paper extensions. A provisional useful signal is worthwhile reference-grounded organization/expression improvement beyond count compliance on at least **two of four extension cases from different papers**, without more moderate/high-drift cases than A. This is an exploratory follow-up criterion, not statistical significance or a validated utility metric. Minor drift can be acceptable for substantial benefit. Report the concrete changes and how the decision depends on that judgment.

Check preserved actors, citation roles, conditions, quantities, modality and causal direction; actual claim-to-sentence organization; unnecessary additions; copying; and style separately. Sentence counts, lexical overlap and distance to the training style profile are diagnostics, not correctness or author-identification scores. Faithful local edits alone do not establish distinctive Fama/coauthor style. The independent/shared-context limitations of generation and any unblinded assistant assessment must remain explicit.

If ordinary rewriting performs similarly, this provides no reason to learn a mapper merely to reproduce the target grouping. If plans add useful benefit across papers, expand the eligible elementary-edit collection before any operation-conditioned training. Original author paragraphs remain the only author-style targets. No development or reserved test examples are used in preparation or selection.

## CPU import and reproducibility

```bash
PYTHONPATH=src python3 -m fama_style.rhetoric.paper_plan_comparison \
  --prepare --output-dir research_runs/paper_plan_comparison_v1/package_v1
PYTHONPATH=src python3 -m fama_style.rhetoric.paper_plan_comparison \
  --check --package research_runs/paper_plan_comparison_v1/package_v1
PYTHONPATH=src python3 -m fama_style.rhetoric.paper_plan_comparison \
  --audit-responses \
  --package research_runs/paper_plan_comparison_v1/package_v1 \
  --responses-dir research_runs/paper_plan_comparison_v1/responses \
  --output-dir research_runs/paper_plan_comparison_v1/response_audit_v1
```

Use fresh output directories. The importer verifies package/parent hashes, job IDs, prompt hashes, response files and coverage. It preserves raw bytes and optional notes, excludes all copies of duplicated jobs, rejects wrong hashes/files/extra fields and reports partial/invalid coverage. It does not strip fences, repair JSON, select a better attempt, run NLI or infer a hidden cutoff. Surface metrics and the evaluation-only target sentence count are saved with actual per-condition/anchor/extension denominators; semantic/operation/style quality remains pending an evidence audit.

All **325 CPU tests passed** and the original CLI check passed. Tests cover the no-plan information boundary, identical shared evidence, six-paper nonprobe selection, exact prompt/source provenance, raw-response preservation, partial/duplicate handling, tamper rejection and absence of network/model/training actions. No external request, model download, Kaggle submission or training was performed. No results from unexecuted tasks are claimed.

Implementation: `src/fama_style/rhetoric/paper_plan_comparison.py`; tests: `tests/test_paper_plan_comparison.py`.
