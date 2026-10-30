# Multi-author milestone v1: preparation and evaluation specification

This implements the first step of the agreed [development plan](../DEVELOPMENT_PLAN.md).
The preparation below records v1. A [sparse detector policy v2 and first
user-supplied report](SPARSE_DETECTOR_V2.md) now supersede its upfront detector
usage schedule; the frozen v1 package is unchanged. No agent detector calls,
model inference, training or GPU execution have occurred.

## Provisional dataset

The frozen earlier corpus contains 60 training, 20 development and 20 reserved
paragraph pairs from 20 non-Fama papers published before 2020. This package uses
only training/development annotation bodies. Reserved records are not deserialized
or analyzed; registry counts and byte-level input hashes supply provenance only.
The separate original Fama 45-example reserved split is not opened or repurposed.

| Selection | Training | Development |
|---|---:|---:|
| Earlier nonreserved rows | 60 | 20 |
| Quarantined uncertain claim alignment | 2 | 0 |
| Quarantined known epistemic-strength shift | 0 | 1 |
| Quarantined identical source/target | 0 | 1 |
| Provisional candidates | **58** | **18** |

Rows retain exact source/target text, hashes, original paragraph ID, paper/version,
publication evidence, PDF checksum, authors and stable author-group identity.
The upstream audit manifest pins annotation/extraction history; no wording or
prior annotation is repaired. Duplicate normalized sources/targets are checked;
none adds an exclusion in this build. Whole-paper and author split separation
is enforced. Current authors are metadata, not conditional labels learned by
the baseline. Current author-separated development cannot evaluate learning a
particular unseen author's label; that future question needs paper-held-out data
for authors actually represented in training.

The candidate set still carries review flags: **42/58 training and 16/18
development examples**. Many concern context, not demonstrated meaning errors.
All-preserved self-labels and exact evidence do not certify completeness or
equivalence. A 58-pair collection supports at most a small pilot; it is not a
full-scale author-style dataset. The files are named provisional and are not
automatically consumed by the existing training CLI. No training is approved.

## Step 1: detector-control feasibility

Private entry point:
`research_runs/multi_author_milestone_v1/package_v1/START_HERE.md`.

Start with `detector_controls/INSTRUCTIONS.md`. Twelve blind-ID text files cover
six matched training-paper pairs: historical human original and GPT-6 rewrite.
Paper choice is deterministic; within each paper the pair with the longest
shorter text is used to reduce obvious length-related missing results. It is a
length-selected calibration sample, not a representative detector benchmark.
Source/target identities are in `PRIVATE_KEY.jsonl`, never in the detector item.

If access/quota permits, submit each item unchanged to GPTZero manually and fill
`results_template.csv` with actual label, score/scale if supplied, date, visible
provider version and decision rule. Keep `too_short`, `unavailable` and
`provider_error` statuses; never fabricate scores, silently drop items, concatenate
paragraphs, edit inputs or retry for a preferred label. Do not upload keys,
references, datasets or credentials. No provider calls have been made here.

The initial binary endpoint counts AI, Mixed and AI Paraphrased labels as flagged,
Human as unflagged; unsupported labels stay unmapped. Provider-native flags are
primary and numerical scores secondary; percentages do not establish human
authorship. A fixed small-pilot feasibility screen requires all six matched
pairs scored, at most one of six human texts flagged, and at least three of six
GPT-6 rewrites flagged. These thresholds are planning choices, not estimates of
population accuracy. Failure means the endpoint is uninformative under this
protocol; do not optimize a model or pick easier examples to repair the result.
Any revised endpoint or corpus needs a new version.

GPTZero explicitly describes errors and weaker performance at paragraph than
document level in its [limitations](https://support.gptzero.me/articles/1787085022-what-are-the-limitations-of-gptzero-s-ai-classifier).
The corpus consists of paragraphs, so coverage and human false positives are
essential. No promise is made about current free quotas or access. A future
second detector must be specified/frozen before cross-detector claims are made.

## Step 2: prompted-base comparison, awaiting authorization

Prepared notebook: `kaggle/multi_author_baseline.ipynb`.
Upload **only** the two files in the private packet's `upload/` folder as a private
Kaggle Dataset. Training candidates, development references and detector keys
must remain outside that upload. Enable `HF_TOKEN` through Kaggle Secrets before
any authorized run. Do not use CLI push/submission. Selecting T4 x2 is compatible
with the first cell exposing GPU 0 alone. Do not execute the notebook yet.

The baseline requests **36 first-attempt outputs**: two matched conditions for
each of 18 development sources, using `google/gemma-3-1b-it`:

1. **Ordinary:** one fixed natural-academic-economics rewrite instruction.
2. **Source structural prior:** the identical instruction plus a three-level
   source-only prior. It retains reasoning order, optionally proposes at most
   one ≤30-word source pair, and supplies numerical training-reference sentence/
   semicolon distributions as descriptive context. The shortest eligible pair
   is proposed, with an explicit instruction to keep separation if clearer.

The prior is a static baseline, **not the learned taxonomy mapping**, a full
132-category execution engine or an oracle author plan. Its descriptive profile
is fitted on training targets only. Development targets, original-derived claim
propositions, taxonomy labels, author metadata and target vocabulary are excluded
from jobs. Source citations can still reveal provenance. The structured prompt
is longer and repeats source sentences; resulting input-token costs are recorded,
and effects cannot be attributed to structure alone without a future matched-
length control. The mixed outcome of earlier ≤30 checks remains a limitation.

Both arms use NF4 double quantization with FP32 compute, one visible GPU, seed
42, greedy decoding, 512 new tokens and a 4096-token total budget. Tokenization
never truncates; over-budget inputs fail. No reference is loaded into the GPU
runtime. The first run resolves and records an immutable model commit; future
adapter comparisons must reuse that SHA and these input/quantization settings.
Installed package versions, token hashes, output-limit flags, runtime and GPU
memory are saved. Transformers 5.20.1 reuses an existing repository dependency
pin; this new runtime has not yet been executed on Kaggle. The official
[model card](https://huggingface.co/google/gemma-3-1b-it) supports the text-only
AutoTokenizer/AutoModelForCausalLM route used here.

The function named `run_training` follows the existing GPU-import convention
but performs frozen inference only: model parameters are nontrainable and there
is no trainer, optimizer or backward pass. All ML imports stay inside it.
CPU `--check` validates bytes, job coverage, matched sources and decoding without
loading libraries, tokenizers or weights. Saved notebook outputs are empty and
the notebook source is self-contained and below Kaggle's 1 MB limit.

## Evaluation after outputs exist

Keep four evidence families separate:

- **Meaning:** source-first automated units/constraints, then preserved, local
  shift, material shift or unresolved findings. Report added/lost claims, actors,
  quantities, qualifiers, conditions, attribution, negation and causal direction.
  Assess outputs against the given input first; do not blame inherited input/
  reference drift on the model. Local shifts remain explicit under a drift budget.
- **Readability and style:** blind ordinary-versus-structured comparisons of
  naturalness, organization, precision and density using a frozen rubric.
  Same-model judges are not independent gold. Reference similarity is secondary;
  one human paragraph is not the only acceptable realization.
- **Copying/reconstruction:** exact and normalized input copies, source/reference
  lexical overlap, output length, repeated spans and target reconstruction.
  Input copies do not demonstrate style learning; matching public historical
  references can reflect memorization. Paper/author splits do not establish
  absence from pretraining.
- **Detectability:** freeze output bytes before external measurement. Compare
  unchanged GPT-6 input, historical-human reference, ordinary base and structured
  base controls under the same provider/date/version/rule, retaining missing
  cases and recording scores separately from labels. No repeated editing or
  generation to optimize a detector. A later adapter uses those same controls.

Future pilot progress requires a lower declared detector flag rate than the
prompted base, no increase in material drift, readability wins exceeding losses
and no increase in exact input copying. All outcomes remain visible when the
gate fails. Small development counts are exploratory; reserved evaluations stay
sealed until model/prompt selection and measurement rules are frozen. Future
QLoRA needs separate authorization, complete-target token/label checks and a
versioned training configuration. A pipeline smoke test is not fine-tuning evidence.

## External benchmarks and public artifacts

[BUST](https://aclanthology.org/2024.naacl-long.444/),
[NIST GenAI](https://www.nist.gov/publications/nist-genai-pilot-overview-text-text-evaluation-results)
and [WATTLE](https://aclanthology.org/2026.wac-1.25/) evaluate detection tasks and
robustness; they are not automatic paired-rewrite quality scores. No benchmark
content was downloaded or mixed into training/development here. Any later use
needs access/scope/licensing checks and an evaluation split separate from training.

Public files contain [protocol](protocol.json), [preparation counts](preparation.json)
and [pooled numerical profile](training_style_profile.json). Private data, source
text, targets, detector keys/results and future model outputs stay Git-ignored.
The original Fama pipeline, earlier results/hashes and both reserved splits are
unchanged. No credentials are stored or printed. CPU tests cover quarantine,
author boundaries, duplicate screening, reserved-body exclusion, source-only
guidance, train-only profile fitting, detector-control origins, input hashes,
lazy imports, Secret retrieval and error redaction, immutable saves and notebook
structure. No model/detector call, GPU job or training was started.
