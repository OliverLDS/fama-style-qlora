# Expanded teacher-pair baseline: completed

Downloaded the explicitly requested version 1 of
`oliverlds/teacher-pair-expanded-base-4b`, including its notebook log, runtime
source and complete `teacher_pair_base_v2/` directory. The user supplied the
underscore form of the notebook name; Kaggle's actual slug uses hyphens.
The run completed in **207.4 seconds on one Tesla T4**, with zero optimizer
updates and no training. This review made no model calls or GPU submissions.

| Check | Observed result |
| --- | ---: |
| Training label audits | 46/46 |
| Development label audits | 8/8 |
| Largest full sequence | 550 tokens / 1,024 allowed |
| Largest prompt | 311 tokens |
| Largest prompt plus generation budget | 823 tokens / 1,024 allowed |
| Development generations | 8/8 |
| Training probe generations | 2/2 |
| Empty or length-limited outputs | 0/10 |
| Target-token-weighted development loss | 2.9694209101813636 |
| Mean per-example development loss | 3.2463565915822983 |
| Outputs identical to the original base run | 10/10 |
| Development loss rows identical to the original base run | 8/8 |

The frozen data/config hashes, downloaded runtime hash, pinned Gemma revision,
one-T4 device setting and recorded ML package versions were verified. The
runtime's token audit reports target-only supervision, complete coverage and no
truncation. CPU review additionally validates exact IDs, split assignments,
positive supervised-token counts, token hashes, prefix masking and generation
context budgets. Both loss aggregates were recomputed from the eight loss rows.

## Generation findings

All ten outputs reproduce the earlier base outputs exactly, with the same
source, reference, prompt hash and split. Their earlier text-bound qualitative
assessments are therefore reused unchanged; this is **not a new independent
semantic assessment**. Hash receipts bind the original report, original review
and reused judgments. The detailed findings remain in the
[original baseline review](../teacher_pair_pilot_v1/BASELINE_RESULTS.md).

| Development outcome | Result |
| --- | ---: |
| Material claim issues identified | 7/8 |
| Exact source copies | 0/8 |
| Median source word-sequence overlap | 0.288 |
| Median reference word-sequence overlap | 0.226 |

Both training probes also retain the same material issues, reported separately
from development. Unsupported elaboration, omitted qualifications and changed
economic scope remain the concern; these outputs do not establish distinctive
Fama style. Exact copying is absent, but low copying does not certify fidelity.

The source data expansion changes only future training examples. Because no
updates occurred here, unchanged development results are expected and establish
a reproducible starting point, not an effect of the additional pairs.

## Log and credential check

The downloaded log contains the model-access, tokenizer-preflight, model-load
and base-generation phases, with no Python traceback. Kaggle includes duplicate
renderings of stream events; those are not treated as additional generations.
The pip resolver reports that installed Pillow 13 conflicts with the preinstalled
Gradio requirement of Pillow below 13. This runtime does not use Gradio, and
model loading, losses and inference completed. No dependency changes or rerun
were needed to review this result; do not generalize compatibility to Gradio.

Kaggle token retrieval remains through `UserSecretsClient().get_secret('HF_TOKEN')`.
The downloaded runtime matches the frozen source. Its secret-loading failures
hide underlying exception details; it does not print tokens or write them into
reports. No credential patterns were found in the downloaded source, JSON or
log. Credentials remain outside repository artifacts.

## Decision

The baseline, dataset/tokenizer and reproducibility gates for the
[proposed two-epoch / 46-update pilot](EXPERIMENT.md) pass. The next experiment
can now test whether the expanded teacher pairs improve claim retention and
expression on the same eight development cases. More data and more optimizer
updates both change relative to v1; retain that confound and report both epochs.

No pilot was enabled, submitted or run during this baseline review. The user
subsequently authorized the bounded 46-update scope; the enabled notebook and
exact reviewed baseline hash are documented in [PILOT_RUN.md](PILOT_RUN.md).
No job has been submitted by these tools. These results do not justify full
training, a hyperparameter search, reserved-test use, or Fama-style/human-
authorship/detector claims.

Private downloads: `research_runs/teacher_pair_reverse_pilot_v2/kaggle_base_v1/`.
Private review: `research_runs/teacher_pair_reverse_pilot_v2/base_review_v1/`.
Public aggregates: [baseline_results.json](baseline_results.json).
CPU validation is implemented by
`fama_style.research.teacher_pair_expanded_baseline_review`.
