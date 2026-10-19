# Reproduction and local checks

## Public checkout

```sh
python3 -m unittest discover -s tests -v
python3 train.py --check
python3 scripts/build_notebooks.py
```

Python 3.9+ is sufficient. The default dry-run uses three synthetic educational examples and does not access credentials, import ML libraries, download weights or train. Three rows/five GPU steps are a pipeline smoke test, not a quality experiment. Tests needing private artifacts skip explicitly; synthetic checks remain active. Optional Node-based DOM checks do not assert live visual browser rendering.

The public code can be inspected, tested and exported to notebooks without the corpus. It does **not** supply all data/weights needed to reproduce reported model results. ML imports stay inside runtime functions; package installation does not install ML dependencies.

## Restore exact local artifacts

```text
data/fama_corpus_v1/curated.jsonl
data/fama_corpus_v1/{train,validation,test}.jsonl
data/historical/fama_jensen_1983_pilot_instructed.jsonl
data/historical/reviewed_training_decisions.private.json
experiments/corpus_v1_baseline/outputs/version_4/...
archive/corpus_v1_smoke_version_1/...
archive/corpus_v1_pilot_version_1/...
reviews/corpus_curation_20261019/...
archive/fama_corpus_20261019/...
```

Restore only the experiments needed. Curation rebuilds need original sources and recorded semantic decisions. The legacy reviewed-pair builder reads detailed paragraph rewrites from the local-only decisions JSON instead of embedding them in public Python.

Corpus SHA-256: `c24e2d708884877ce61ea212c4d6e442261728be499ff2d15ee7826887a4299f`. Do not relax hash guards to substitute data while claiming the same experiment. New data calls for a new reviewed experiment definition.

## Kaggle sequence

1. `corpus_v1_baseline.ipynb`: private corpus; token audits and base inference.
2. `corpus_v1_smoke.ipynb`: corpus plus successful baseline v4 output.
3. `corpus_v1_pilot.ipynb`: corpus, baseline v4 and smoke v1 outputs. Starts fresh, ends after one epoch.

Import through the UI, verify inputs, select T4 x2, enable internet and the `HF_TOKEN` Secret, then Save & Run All. Only GPU 0 is exposed. Configure model access with the provider outside this repository. No tool here automatically submits Kaggle runs or enables Secrets.

## Offline review

With the exact private pilot output restored:

```sh
PYTHONPATH=src python3 -m fama_style.evaluation.source_control_review build --output-dir /private/tmp/fama_three_way_review
```

Open `review.html`, rate all 63 candidates and export JSON. Browser storage is a convenience; exported JSON is the portable record. Claims are judged against the input, style against the reference. Labels are masked, not independently blinded.

```sh
PYTHONPATH=src python3 -m fama_style.evaluation.source_control_review analyze --review-dir /private/tmp/fama_three_way_review --scores /path/to/exported_scores.json --output-dir /private/tmp/fama_review_results
```

Analysis checks hashes, completeness, ranges and identical-text ties. Claim/style scores and paired counts remain separate. It neither starts training nor invents missing ratings.
