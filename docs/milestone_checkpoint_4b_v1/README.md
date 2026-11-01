# Matched multi-author 4B milestone checkpoint

Status: [version 1 completed and reviewed](RESULTS.md). The user executed the
prepared inference notebook; all fourteen missing outputs passed the integrity
checks. [Completed first-round detector results](DETECTOR_RESULTS_V2.md) cover
twenty texts with each provider. Both flag all twelve Gemma outputs as AI-related
and label all four historical references human. No new training was performed.
Existing experiments and frozen inputs are unchanged. The preparation and manual
execution instructions below document the frozen protocol.

This checkpoint returns to the agreed multi-author milestone before considering
preference optimization. It measures meaning, readability, copying and detector
outcomes separately. It cannot establish a distinctive Fama/coauthor signature.

## Frozen comparison

Use the original sparse-detector subset: eight development cases, two from each
of four papers. Each of two rounds contains one case from every paper. Five cases
overlap the recent eight-case teacher-pair evaluation; the other three are
reintroduced without repairs or substitution. Original context and uncertainty
flags remain attached. Assessment starts from the supplied source; disagreement
with an uncertain historical reference is not automatically a generation error.
Reserved test examples are not used.

| Arm | Reused outputs | New outputs |
| --- | ---: | ---: |
| Base, ordinary rewrite instruction | 5 | 3 |
| Base, source-only structural prior | 0 | 8 |
| Original expanded-pilot epoch-2 adapter, ordinary instruction | 5 | 3 |
| Total | 10 | 14 |

The adapter is from `oliverlds/teacher-pair-expanded-pilot-4b`, version 1,
`teacher_pair_pilot_v2/epoch_2/adapter`. The input-replacement adapter failed its
matched gate and is not used here. Archived outputs retain their exact text and
origin. Prompt/source/reference bindings, package versions, generation settings
and archived prefix-token counts must match. No extra reproduction generations
are scheduled; archive/new-session equivalence remains a limitation.

## Version and structural limits

This is a new 4B checkpoint, not an unchanged execution of the earlier 1B package.
It pins `google/gemma-3-4b-it` revision
`093f9f388b31de276ce2de164bdc2081324b9767`, seed 1729, NF4 with FP32 compute,
greedy decoding and 512 maximum new tokens, matching the saved 4B runs. The older
milestone used 1B and seed 42. The repository's original training model remains
`google/gemma-3-1b-it`.

The structural prior retains the existing optional adjacent-sentence merge rule
(at most 30 combined words), source-sentence display and descriptive style
profile. The profile is refitted on the 46 actual training targets rather than
the older 58-pair package. This is a static heuristic, not a learned rhetorical
mapper or reference-informed oracle. Descriptive sentence-length and punctuation
statistics are not evidence of author-specific style.

Structural prompts repeat source sentences and use a 4,096-token safety guard;
ordinary prompts retain the earlier 1,024-token guard. Neither truncation nor a
RoPE change is allowed. Prompt length and repetition are confounds, so a gain
would not isolate abstract rhetorical structure. All 24 prompts are token-audited,
including the ten archived slots. References and author labels stay outside the
generation upload; only aggregate training-reference statistics inform the prior.

## Quality and sparse measurement

The frozen `quality_rubric.json` distinguishes equivalent wording, minor shifts,
material claim changes and unresolved judgments. An individually eligible rewrite
allows at most one minor shift and no material change. This is an explicit local
research budget, not a semantic gold standard. Changed actors, quantities, causal
direction, central conditions or negation count as material. Aggregate comparison
also requires no increase in material-error rate or exact copying, and more
readability wins than losses. Unresolved judgments are not counted as successes.
The rubric precedes the fourteen new outputs; five cases already have known
ordinary outputs, so this is not an entirely unseen evaluation.

After all outputs finish, the exporter freezes forty detector slots: GPT-6 input,
historical human reference and three model outputs per case. Identical texts are
scanned once while preserving every matched slot. Blinded text files contain only
the exact original text. No trimming, selective removal, output regeneration or
scan-driven editing is allowed. Empty, short or unsupported texts retain explicit
unavailable statuses rather than disappearing from the comparison.

The saved policy uses a single free account and an 8,000-word project ceiling,
based on the user-reported 10,000-word allowance. The current estimate is 5,604
words including the initial calibration pair; generated lengths, provider charges
and remaining balance are unknown. The exporter recomputes actual text counts
and recommends complete four-paper rounds only. Check remaining allowance before
scanning. No paid plan, additional accounts or automatic provider calls are needed.

Record native labels, raw scores, score scale, provider/version and date without
treating scores as calibrated probabilities. Do not pool different provider
snapshots as one matched evaluation. The initial human-control result is still
pending locally; one unflagged GPT-6 rewrite does not establish a floor for Gemma
outputs. A detector with no useful discrimination cannot establish improvement.
Quality judgments and detector outcomes must be reported independently.

## Manual Kaggle execution

1. Create a new private Dataset, for example **Multi-author Checkpoint 4B V1**,
   containing only these two files from `data/milestone_checkpoint_4b_v1/upload/`:
   `milestone_checkpoint_jobs.jsonl` and `milestone_checkpoint_config.json`.
2. Upload `kaggle/milestone_checkpoint_4b.ipynb` into a new private notebook.
   Attach that Dataset and the **version-1 outputs** of
   `oliverlds/teacher-pair-expanded-pilot-4b`, including its epoch-2 adapter.
3. Enable Internet, the existing `HF_TOKEN` Secret and **GPU T4 x2**. The first
   cell selects GPU 0. The token is retrieved through Kaggle Secrets; its value
   is never printed or stored in the package.
4. Save & Run All. This generates fourteen missing outputs with zero optimizer
   updates. Package or compatibility mismatches fail rather than silently
   replacing archived results. Share the actual notebook slug and version.

Retrieve the notebook log, emitted `milestone_checkpoint_4b.py`, and output folder
`milestone_checkpoint_4b_v1/` containing `evaluation.json`, `state.json`,
`token_audit.json` and `generation_config.json`. Do not start checkpoint detector
scans before all fourteen missing outputs complete.

## Local preparation and completion

The packet builder is `fama_style.research.milestone_checkpoint_packet`; the
self-contained notebook builder is `scripts/build_milestone_checkpoint_4b.py`.
The private packet's manifest records input lineage, hashes, exact adapter and
generation settings. Its public plan and rubric are saved beside this document.
CPU tests use synthetic outputs only; they do not report model results.

After downloading a completed run, prepare blinded quality review and detector
files in a fresh private directory:

```sh
PYTHONPATH=src python3 -m fama_style.research.milestone_checkpoint_export \
  --package data/milestone_checkpoint_4b_v1 \
  --download research_runs/milestone_checkpoint_4b_v1/kaggle_v1 \
  --output research_runs/milestone_checkpoint_4b_v1/export_v1
```

The exporter refuses incomplete runs or altered bindings. It creates no detector
scores and makes no external calls. Automated quality assessment and manual sparse
scans remain pending until genuine model outputs are available.
