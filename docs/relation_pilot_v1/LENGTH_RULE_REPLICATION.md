# Additional-paper replication: fixed 30-word rule

**Current status:** [verification audited and fixed policies scored](LENGTH_RULE_RESULTS.md).
The generation-preparation description below records the original protocol.

**Prepared, not executed:** 32 fixed adjacent pairs from 17 existing neutral
paragraphs, in eight generation batches of four. The 30-word cutoff is unchanged.
There are no new generation or verification results and no refitted classifier.

The four papers differ from all four used in the cutoff experiment, with distinct
authors. All are in the earlier **training** pool, not development or reserved
splits. They have appeared in earlier project work; the paragraphs are reused,
not project-unseen. Fresh candidates will test transfer to additional papers,
without claiming a pristine independent benchmark.

| Paper | Uploaded version / publication year | Pairs |
|---|---|---:|
| Holmstrom, *Moral Hazard and Observability* | 1979 / 1979 | 8 |
| Pastor–Stambaugh, *Are Stocks Really Less Volatile in the Long Run?* | 2011 / 2012 | 8 |
| Shiller, *Human Behavior and the Efficiency of the Financial System* | 1998 / 1999 | 8 |
| Boudoukh–Israel–Richardson, *Long-Horizon Predictability: A Cautionary Tale* | 2019 / 2019 | 8 |

Dates and source hashes reuse the previously checked
[source metadata](length_replication_sources.json); no new PDF extraction is
performed. Publication/version dates precede 2020, but these papers may be known
through model pretraining and earlier annotations.

## Sampling and inputs

Each paper contributes four pairs at or below 30 words and four above it: **16
per stratum** overall. Pair lengths range from 16 to 59 under the original
ASCII-letter word regex. Numerals are not counted by that legacy function.
Strata balance is a sampling design, not a forced quality-label balance or an
estimate of natural prevalence.

Papers require capacity in both strata. A fixed hash order then chooses four,
excluding prior cutoff papers/authors and repeated selected authors. Within each
paper/stratum, four length-quantile groups provide one pair each, with fixed hash
tie-breaking. Selection uses source length and metadata, not earlier reference
layouts, grouping labels, generated candidates or verifier outcomes. Some pairs
overlap or share a paragraph; no paragraph repeats within a batch and every
batch contains four papers. No chained paragraph edit is licensed.

Source eligibility requires mechanical integrity, an available neutral paragraph,
complete source-sentence reconstruction and neither pending nor unusable source
status. Of 32 selected pairs, ten inherit usable source status and 22 uncertain;
23 inherit review flags and 17 context-dependence flags. These pair counts include
repeated paragraphs. Typical cautions concern missing surrounding definitions,
figures or document context. They remain pinned privately and must not be
silently treated as resolved. The local comparison is relative to the supplied
neutral text, not independent verification against the original author's paper.

## Two stages

Private entry point:
`research_runs/length_rule_replication_v1/package_v1/START_HERE.md`.

**Run candidate generation first.** For each `generation/batch_01` through
`batch_08`, paste `TASK.md` and attach its `items.jsonl` plus
`common/generation_output.schema.json`. Use GPT-6, request High effort and report
effective effort as uncertain if necessary. Save first complete unedited
`responses_01.jsonl` through `responses_08.jsonl` and coverage reports in
`responses/generation/`, then notify Codex. Do not regenerate to obtain a desired
outcome.

Each payload contains only an opaque item ID, neutral paragraph and fixed left/
right sentence. The generator attempts one faithful, readable single-sentence
replacement or abstains. It does not select pairs or judge merge preference.
No cutoff, rule prediction, sampling stratum, author metadata, original reference,
earlier taxonomy, claims, warnings or judgment is exported. The merge operation,
source wording/length and broad study/version naming remain visible; this is
concealment of the cutoff and prior judgments, not full experimental blinding.

After generation passes audit, all actual available candidates from **both**
strata will receive the existing fixed-candidate verification procedure, with
fresh source-unit extraction, fidelity and readability judged separately.
The [verification specification](verification_prompt.md) and
[schema](verification_output.schema.json) are reused to limit task drift.
Actual verification inputs do not exist yet; never substitute originals or
placeholder candidates. Generation notes/context explanations and rule decisions
will not enter those verification inputs. Same-model self-review, possible chat
retrieval and pretraining remain uncontrolled.

## Locked analysis plan

Compare KEEP, MERGE-all-available and the fixed source-pair ≤30 rule without
threshold selection or classifier fitting. An abstained generation is unavailable
and forces KEEP for all executable policies; report that coverage separately,
without inventing quality labels for a nonexistent candidate.

As in the earlier comparison, primary decision scoring uses faithful candidates
with decisive readability judgments. Faithful ties, local/material shifts,
unresolved judgments and generator abstentions are reported separately, never
silently folded into negative training labels. Report per-paper/stratum counts
and macro paper scores; balanced sampling and faithfulness/availability selection
prevent treating them as corpus-wide accuracy. Keep selection/drift counts
visible even when they do not enter the primary score. There is no automatic
approval of examples for fine-tuning and no evidence of distinctive author style
from this preparation.

## Artifacts and checks

Public: [generation prompt](length_replication_generation_prompt.md),
[generation schema](length_replication_generation.schema.json),
[locked protocol](length_replication_protocol.json),
[counts](length_replication_preparation.json) and metadata linked above.
Source paragraphs, exact pair selection, earlier cautions, hidden rule decisions
and future responses remain Git-ignored. Existing experiments, hashes, source
texts and reserved data stay unchanged.

CPU builder: `fama_style.research.length_rule_replication`, with `PYTHONPATH=src`,
`--audit`, `--registry`, `--selector`, `--prior-papers` and a fresh `--output`.
Manifests pin upstream audit/selector, metadata, code, protocol and packet bytes.
The session-notes form and future responses are outside the preparation manifest.
Tests cover reserved/development-body exclusion, paper/author separation,
deterministic balanced sampling, cutoff tampering, prior-judgment exclusion,
exact-text/abstention/context checks, future verification projection and immutable
outputs. No external model calls, threshold tuning, GPU or training runs occur.
