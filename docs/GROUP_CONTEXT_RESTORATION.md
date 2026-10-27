# Read-only source-context restoration v1

Status: prepared, not executed. This is an inference-only mechanism check following groupwise realization v2. That run produced 33/33 single-sentence pieces, but an exploratory, unblinded assessment found high drift in 8/9 assembled paragraphs. Correct format alone did not yield faithful or distinctive style.

This experiment asks whether missing source referents contributed to the drift. It does not train a rhetorical mapper or adapter, and does not replace earlier results.

## Matched design

Three source groups, three seeds, two conditions: **18 calls**. There is no paragraph assembly.

| Selected group | Reason |
| --- | --- |
| Managerial supervision, group 2 | The isolated possessive lacks a local antecedent. |
| Dividend initiation, group 4 | Rates and “starting” lack a local firm/dividend subject. |
| Dividend initiation, group 5 | Both expected and actual quantities are already present; a prior output invented a derived percentage. A context-complete numeric control. |

A uses the exact v2 local-only prompts, sampling seeds and proportional group token limits. B appends the full plain-English source as clearly marked read-only context, with instructions to resolve referents while expressing only the assigned group's claims. B also says not to add derived quantities; the intervention therefore includes context-use instructions, not just additional tokens. Alternate pair order; no calls depend on previous responses.

Three samples per condition/group are matched by seed, not identical token-by-token random trajectories. All selected groups are from training papers; none of the 21 validation or 45 reserved test rows is read. The private pack retains all three historical source cases for frozen constructors, but only the three selected groups from two cases generate calls. Oracle grouping is retained. Reference wording and evaluation notes never enter generation prompts.

Pinned base Gemma 3 4B, same NF4/FP32 setup, sampling temperature 0.6/top-p 0.9, context limit 4096 and no prompt truncation. Only GPU 0 is visible. No adapters, optimizer, training, retries, output repairs or source fallback. Empty/cutoff responses remain failed samples. Format errors remain verbatim. This is smaller than the preceding 42-call run, but loading and NLI overhead remain; record cost rather than promise a runtime.

## Reading results

Compare missing/reassigned actors, changed actual-versus-expected relations, quantities and invented mechanisms within each matched group/seed. Separately judge style change, copying and whether the model imported other source-group claims. Retain the established allowance for minor drift; no perfect-preservation gate or automatic NLI acceptance filter is imposed.

Save full local-group verifier verdicts and full-source-to-output NLI, not a full-paragraph equivalence verdict for a one-sentence assignment. Local checks can flag valid antecedent resolution. Full-context entailment can accept an unrelated source claim and does not establish correct group binding or omission-free realization. Neither is calibrated semantic accuracy. Surface features and source overlap are diagnostics, not validated author-style scores.

Historical A output hashes should reproduce. Investigate mismatches before attributing changes to context. If context resolves actors but numeric/relation errors remain, do not treat that as readiness for fine-tuning. Even a positive result is evidence about three selected groups, not generalization or superiority of the structural approach.

## Manual Kaggle run

1. Upload `kaggle/group_context_restoration.ipynb` as a **new private notebook**.
2. Create/attach a **private Dataset containing only** `research_runs/group_context_restoration_v1/input_pack/context_pack.json`. The pack includes copyrighted source/reference text for local evaluation; keep it private.
3. Enable internet, choose GPU T4 x2, and enable the existing `HF_TOKEN` Kaggle Secret. The first cell exposes only GPU 0; credential values are never printed.
4. Save & Run All. Share the notebook slug and version after completion.

Download `/kaggle/working/group_context_restoration/`: `evaluation.json`, raw generations, metrics, full verifier verdicts, jobs, token audit, runtime, status and hash manifest. An execution can finish with failed samples; all attempts remain in the denominator.

## Reproducibility

Implementation: `src/fama_style/rhetoric/group_context_restoration.py`.
Builder: `scripts/build_group_context_notebook.py`.
CPU tests: `tests/test_group_context_restoration.py`.

The private pack binds parent pack hash, selected groups, model revision, runtime source hashes, jobs, historical controls and decoding. The notebook embeds runtime code and public verifier fixtures, without paragraph data or credentials. Its preflight works without a repository checkout. The pack preflight and synthetic dry run require no ML imports, Secret or model weights. Synthetic outputs are labeled and never reported as model results. Existing experiment files remain unchanged.
