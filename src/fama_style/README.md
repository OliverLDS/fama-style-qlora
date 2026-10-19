# Code layout and commands

All ML imports remain inside runtime functions. Importing the package, running tests or using `--check` does not download weights or require a GPU. Run the following commands from the repository root.

| Package | Contents |
|---|---|
| `training` | Original 1B pipeline, reviewed 4B smoke and reviewed 4B epoch tracking |
| `evaluation` | Saved-adapter style comparison and base-model few-shot comparisons |
| `diagnostics` | Copying, supervised-label, memorization and epoch-adapter checks |
| `prompting` | Frozen rewrite and few-shot instructions/formatters |

The original entry point remains:

```sh
python3 train.py --check
```

Other module commands use the source path, for example:

```sh
PYTHONPATH=src python3 -m fama_style.evaluation.compare_few_shot --dataset data/historical/fama_jensen_1983_pilot_instructed.jsonl --check
PYTHONPATH=src python3 -m fama_style.diagnostics.overfit_two --dataset data/historical/fama_jensen_1983_pilot_instructed.jsonl --check
```

An editable installation of this package is optional; package metadata does not install ML dependencies. Tests add `src/` to their import path and need no installation.

Data-preparation entry points are under `scripts/data/`:

```sh
python3 scripts/data/prepare_fama_pilot.py --help
python3 scripts/data/review_fama_training.py --help
python3 scripts/data/build_curated_corpus.py
```

The PDF screen (`scripts/data/prepare_curation.py`) additionally needs pypdf. Source-review scripts retained next to historical audit artifacts in `reviews/` are specific to those audit snapshots.

`scripts/build_notebooks.py` reads these package modules and embeds their dependencies into standalone notebook cells or helper files. The generated Kaggle helpers use local imports and do not require installing this repository on Kaggle. Edit the Python sources, then regenerate the notebooks.
