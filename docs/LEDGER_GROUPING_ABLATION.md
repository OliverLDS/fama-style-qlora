# Ledger format and grouping: eight-generation diagnostic

Status: prepared and CPU-tested, **not executed on Kaggle**. Previous code,
notebooks, datasets and results remain unchanged. This follows the observed
sentence inflation and annotation-cue leakage in the 21-example comparison.

## Upload and run

1. Upload `kaggle/ledger_grouping_ablation.ipynb` into a **new private notebook**.
   Suggested title: **Ledger Grouping Ablation**.
2. Create a new **private Dataset** containing only
   `research_runs/ledger_grouping_ablation_v1/input_pack/ledger_ablation_pack.json`
   and attach it. Suggested title: **Ledger Ablation Inputs**. Older packs are
   not inputs to this notebook.
3. Enable internet, **GPU T4 x2**, and the notebook's **HF_TOKEN** Secret.
   Only GPU 0 is visible. Credentials never belong in code or input files.
4. Save & Run All. Share the notebook slug and version after completion.

Keep inputs and outputs private because the package contains source/reference
paragraphs and original prompts. Neither the builder nor preparation submits a
Kaggle job. The approved pack SHA-256 is
`27fb70dc4a75c8b214dfa38fec9f4b42f96ac255033f70d0bdaffbbff8fbbfee`.

## Controlled factors

Two already-observed development failures receive four first attempts each:

| Condition | Evidence format | Existing soft-plan block |
|---|---|---|
| A | Source sentence evidence only | Off |
| B | Source sentence evidence only | On |
| C | Cue-rich ledger | Off |
| D | Cue-rich ledger | On |

Sentence evidence, claim IDs, ledger status, full source passage, master rewrite
instruction and common preservation instructions are identical. The simple
ledger removes **only** each claim's `protected_evidence` field, which lists
isolated protected words/numbers and their kind labels. Every source sentence
still contains its original quantities, citations and qualifiers. Nothing is
rewritten or repaired in the data.

C and D use the original 21-example A and B prompts **byte for byte**. The on/off
factor retains the whole existing optional plan block, including source-order
labels and semantic reminders; it does not isolate a single grouping sentence
from every other token in that block. Prompt length changes with the interventions
and is recorded; no length-matched placebo is included.

The same pinned base Gemma 3 4B, NF4/FP32 compute, greedy decoding, seed 42,
768-new-token budget and context guard apply to all conditions. There is no
adapter, optimizer, training, reranking, repair or hard sentence quota. Order is
A/B/C/D on one case and reversed on the other. The original 1B CLI is untouched.

## Case selection and hypotheses

Selection is explicitly **post-hoc**: the two B outputs with the lowest frozen
style-gain proxy in the reviewed 21-example run, breaking ties by ID. One is the
known 213-word sentence with metalinguistic cue leakage; the other is a separate
single-sentence failure. The selection rule and parent artifact hashes are
recorded before new generation. Neither source is a fresh validation case.
The 45 reserved test examples are never read.

The diagnostic asks whether removing isolated cue fields reduces annotation
leakage, whether grouping increases sentence inflation, and whether the factors
interact. It does not measure generalization, prove a better taxonomy or authorize
new fine-tuning. With two cases, report each within-case contrast, not just an
aggregate winner.

## Evaluation

The predeclared artifact screen records:

- Protected cue words/short phrases newly placed in quotes, excluding quantities
  and cues already quoted in the source.
- New metalinguistic terms such as modality, ledger or annotation, absent from
  the source.
- Newly introduced numerical-evidence phrases referring to a year.

These are surface warnings developed from the observed failure, not a validated
semantic-error detector. A legitimate paraphrase can trigger a warning, and an
unsupported claim can evade all marker patterns. The existing v3 verifier remains
an independent diagnostic alongside these warnings; its synthetic success is
not economics-domain accuracy.

Report all eight outputs: artifact warnings, sentence counts and words per
sentence, format, verifier status/abstentions, copying, style-distance gain,
tokens and time. Reference features are diagnostic only and never supplied to
Gemma. Unknowns remain in two-example denominators. Empty/cutoff generations stop
the run with preserved partial files.

The cue-rich controls also compare their output hashes with the prior run.
If the old artifact does not reproduce, do not claim that a simple ledger cured
it; examine control replication and remaining within-run contrasts. No hash
mismatch is silently repaired or selected away.

## Execution and artifacts

Before Secret access, the notebook validates pack/job/source hashes and repeats
the passed synthetic v3 verifier calibration as a runtime canary. If the canary
fails, generation stops. NLI is unloaded before Gemma loads, and Gemma is unloaded
before output verification. Token/context overflows abstain or stop rather than
truncate. At most one model's weights occupy GPU 0 at a time.

Output directory: `/kaggle/working/ledger_grouping_ablation_v1/`.
Main files: `evaluation.json`, `metrics.private.jsonl`, `generations.private.jsonl`,
`jobs.private.jsonl`, `nli_probability_cache.private.jsonl`, token audit, run status
and recursive hash manifest. The synthetic canary is retained in its own folder.
The runner measures runtime and peak memory. Roughly 3–8 runner minutes is a
planning estimate only; queue, installation and cold downloads add time.

Implementation is isolated in `src/fama_style/rhetoric/ledger_ablation.py` to
preserve the earlier frozen experiment. It uses the same inference lifecycle and
verifier. ML imports remain inside `run_training`, which calls no trainer.

```sh
python3 scripts/build_ledger_ablation_notebook.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m fama_style.rhetoric.ledger_ablation \
  --dry-run --output-dir /private/tmp/ledger-ablation-new-dry
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

CPU checks cover exact control prompts, factor isolation, reference exclusion,
marker positives/negatives, output identity/cutoffs, unknown denominators and a
self-contained notebook dry run outside the checkout. Synthetic runs are labeled
as wiring checks and make no model calls. The notebook embeds code, hashes and
synthetic calibration controls, not real paragraphs.
