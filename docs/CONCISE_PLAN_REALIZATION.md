# Concise target-plan realization diagnostic

Status: prepared and CPU-checked, **not executed on Kaggle**. This is inference
with the pinned base Gemma 3 4B used in the earlier corpus/oracle experiments.
The original Gemma 3 1B training workflow and all earlier experiments are unchanged.

## Run through Kaggle's UI

1. Upload `kaggle/concise_plan_realization.ipynb` into a **new private notebook**.
2. Create a **private Kaggle Dataset** containing ONLY
   `research_runs/concise_plan_v1/input_pack/concise_plan_pack.json` and attach it.
   A suggested dataset title is **Concise Plan Realization Inputs**. Preserve the
   exact filename and file bytes. It contains copyrighted reference text for
   evaluation and must remain private.
3. Enable the notebook's `HF_TOKEN` Secret, internet, and **GPU T4 x2**. The first
   code cell exposes GPU 0 only. No CLI push/submission is necessary.
4. Use Save & Run All. The preflight cell validates the approved input hash before
   Secret access or model loading; the final cell runs inference only.
5. When finished, share the notebook slug and version. Keep the notebook and output
   files private. The main result is
   `/kaggle/working/concise_plan_realization_v1/evaluation.json`.

Use a fresh session/output path for a rerun; previous results are not overwritten.
The notebook has not been uploaded or submitted by the preparation scripts.

## Frozen comparison

Four existing training pairs were selected before model generation, one per paper,
with two sentence-count increases and two decreases:

| Training paper key | Source sentences | Reference-plan sentences |
|---|---:|---:|
| F1965 | 2 | 3 |
| FF2000 | 3 | 2 |
| NL2003 | 3 | 2 |
| VL2018 | 5 | 6 |

Only references retained unchanged by the earlier ChatGPT review are eligible.
Their faithfulness is still unverified. The original text pairs remain untouched;
none of the proposed neutral repairs is substituted. Selection minimizes word
budget subject to four-paper and direction balance. It uses no model outputs,
development-validation rows or reserved-test rows.

- **A:** frozen ordinary Fama/coauthor rewrite instruction plus the source.
- **B:** exactly the same instruction/source, followed by a concise presentation plan.

Eight greedy generations use the same revision, quantization, seed and output
budget; order alternates by pair. There is no adapter, optimizer or training.
Both conditions receive the same substantive information. A semantic ledger is
not added to B alone.

## What the plan contains

The existing LLM target annotation supplies verified text-coverage boundaries.
The compiler transmits only sentence count, per-sentence word-length bands,
positive coarse clause hints and a semicolon budget. A single annotated spanning
clause does not become an assertion of simple syntax. Clause hints remain fallible
and conditional on the relationships actually present in the source.

The plan contains no reference words, entities, citations, quantities, claim
descriptions or inferred source/target claim-ID alignment. Its renderer accepts
only closed fields and labels; the generator's job object contains the prompt
and hashes, not reference text. Exact target structure is oracle information,
so this is not a deployable mapping or held-out performance result.

This focused subset tests count/length/grammatical realization. It does not test
the entire 132-category vocabulary or multi-paragraph issue structure, and it does
not depend on the disputed F03/F04/F12 labeling decisions from the repeat check.

## Automated evaluation

- Exact sentence-count adherence, absolute count error, word-band adherence,
  semicolon budget and a separately labeled surface proxy for requested shapes.
- Source copying and change in distance to the frozen training-reference feature
  profile. Reference lexical similarity is diagnostic only, not a selection goal.
- Independent bidirectional document/sentence-unit NLI, with synthetic sanity
  controls and context-overflow abstention. These remain imperfect proxies.
- Source/reference NLI for each selected pair, to reveal potentially unsuitable
  targets or source information loss. All four pairs remain reported; there is no
  post-hoc removal of inconvenient examples.
- Input/output tokens, generation/NLI/wall times and peak memory.

Gemma is unloaded before NLI runs on the same T4, with cached duplicate NLI
requests. This addresses the earlier CPU evaluation bottleneck; no new runtime
speed or memory-fit result is claimed before actual execution. Progress heartbeats
run during slow loads and generation. Empty or cutoff outputs stop the run and
remain incomplete; no sequence is silently truncated.

Before further mapping work, inspect whether planned outputs follow the intended
structure while retaining claims and improving the style proxy relative to A.
Four examples can diagnose a realization problem; they cannot validate a mapper,
establish an author-style signature or justify full training by themselves.

## Local reproducibility

```sh
PYTHONPATH=src python3 -m fama_style.rhetoric.concise_plan \
  --dry-run --output-dir /private/tmp/concise-plan-new-dry
# With private source artifacts, prepare an immutable input pack:
PYTHONPATH=src python3 -m fama_style.rhetoric.concise_plan \
  --prepare-pack --output-dir research_runs/concise_plan_v1/new_pack
python3 scripts/build_concise_plan_notebook.py
python3 -m unittest discover -s tests -v
```

The notebook builder uses the approved `input_pack/concise_plan_pack.json`, embeds
its byte hash and source code, and includes no paragraph data or saved outputs.
The input preparation manifest records parent-corpus/annotation hashes. CPU dry
runs use synthetic identity outputs labeled as wiring checks, never model results.
Local preflight does not access credentials, download weights or run inference.
