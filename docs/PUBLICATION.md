# Public file selection

The public repository is a code and experiment-documentation release. Full paper texts and model artifacts remain in the author's workspace, outside tracking. Publication does not change the private Kaggle notebooks or Dataset visibility.

## Included

- Root README, MIT license, project instructions, package/dependency metadata and original CLI.
- Python source, reusable HTML review template, curation tools and notebook builder.
- Standalone Kaggle notebooks with outputs, attachments and execution counts cleared.
- CPU tests, including synthetic contract tests; private-artifact cases explicitly skip without those files.
- Frozen rewrite instruction, three synthetic educational example rows and data-availability documentation.
- Selected aggregate results, methods, limitations and reproduction instructions under `docs/`.
- A CPU-only GitHub Actions workflow; it never installs model weights or invokes GPU training.

## Excluded

- `archive/`: PDFs, raw datasets/reports, adapters and tokenizer files.
- Real JSONL corpus/splits and detailed paragraph-level curation decisions/repairs.
- `reviews/`: real-data HTML/Markdown packets, candidate mappings, score exports, source text and raw logs.
- `experiments/`: local execution outputs and historical documents with private artifact links or excerpts. Public summaries are in `docs/`.
- Historical Kaggle kernel metadata, machine-specific paths, caches, local settings and credentials.

The root `.gitignore` uses an allowlist. Detailed legacy rewrite decisions have been moved from Python constants into the ignored `data/historical/reviewed_training_decisions.private.json`. The local curation workflow still reads them; their contents are preserved. Public docs make no redistribution claim for papers, Gemma weights or third-party datasets. The MIT license covers project code/original documentation only.

## Review before publishing

```sh
python3 scripts/review_public_files.py --private-scan
python3 -m unittest discover -s tests -v
python3 train.py --check
python3 scripts/build_notebooks.py
```

The publication checker inspects **Git index bytes**, not just working files. It checks the file allowlist, sizes, credential-shaped strings, local home paths and notebook outputs. The optional local scan compares staged text against long paper-derived paragraphs retained in local data/archives; it prints only filenames and finding categories, never matched values. It is a targeted automated check, not proof that every possible credential format or paraphrased source text is detected.

A separate export of the staged public tree is tested without local artifacts before the initial push. This catches accidental dependence on ignored files. Notebook regeneration must leave committed notebooks unchanged.

Git publication and GPU execution are separate operations. No Kaggle push, training, message to another person or credential upload is part of this workflow.
