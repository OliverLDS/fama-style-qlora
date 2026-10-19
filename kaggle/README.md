# Kaggle notebooks

Build from Python sources with `python3 scripts/build_notebooks.py`. The exporter creates local notebooks only: it never uploads files, enables Secrets or starts execution.

## Current corpus experiment

| Notebook | Purpose and required private inputs |
|---|---|
| `corpus_v1_baseline.ipynb` | Token audit and base inference; exact curated corpus |
| `corpus_v1_smoke.ipynb` | Three examples/five updates; corpus and baseline v4 output |
| `corpus_v1_pilot.ipynb` | One epoch/247 examples; corpus, baseline v4 and smoke v1 outputs |

Import through Kaggle's UI. Attach the specified Dataset and notebook outputs, select **GPU T4 x2**, enable internet and **HF_TOKEN** in Kaggle Secrets, then Save & Run All. Only the first T4 is exposed. Configure Gemma model access separately. Never paste credentials into notebook cells or commit them. Do not push with the CLI when you intend only to update content: the established project workflow uses manual UI upload and explicit execution.

The current notebooks enforce frozen corpus/prerequisite hashes. Data and checkpoints are not in this public repository. [Reproduction guide](../docs/REPRODUCIBILITY.md) and [recorded results](../docs/RESULTS.md).

## Historical notebooks

The remaining notebooks preserve earlier experiments: instructed 1B training/comparison, supervised-label and copying diagnostics, two-example overfit, epoch checks, few-shot 1B/4B/12B comparisons and reviewed 4B smoke/pilot runs. Each requires its own historical private artifacts. Do not attach current-corpus data to them merely because their model names overlap.

The original `train.ipynb` uses `google/gemma-3-1b-it`; current corpus notebooks use `google/gemma-3-4b-it`. Three examples/five steps are only a pipeline smoke test. Full training is not automatically authorized by notebook generation or publication.

All committed notebooks have saved outputs cleared. Tests compile their code cells and check exported helpers without requiring this package to be installed in Kaggle.
