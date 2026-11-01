# Teacher-pair reverse-direction pilot v1

The [base run has reviewed version 1 results](BASELINE_RESULTS.md): all ten
outputs and 24 label audits completed. The user subsequently authorized the
limited two-epoch/16-update pilot on 31 October 2026. The
[pilot has now completed](PILOT_RESULTS.md): lower loss and modest expression
preferences, but persistent claim drift prevents scaling. No additional training
has been authorized by the review.

The specification below records the fixed experiment. It uses the intended learning direction:
**GPT-6 neutral rewrite → archived human academic paragraph**. It tests a small
pooled multi-author QLoRA adaptation against prompted base Gemma 3 4B. It does not
ask Gemma to reconstruct neutral inputs from a teacher's brief or use taxonomy
plans, reference demonstrations or named-author prompts. All earlier experiments
and datasets remain unchanged.

## Data audit and frozen selection

The parent milestone has 58 training and 18 development candidates after four
prior quarantines. This new audit accounts for all 76 candidates and selects:

| Selection | Training | Development |
| --- | ---: | ---: |
| Pairs | 16 | 8 |
| Papers | 10 | 4 |
| Author groups | 10 | 4 |
| Median input/reference word-sequence overlap | 0.764 | 0.780 |

The 16 training pairs are all of the previously flag-clear training candidates,
now checked qualitatively against their archived human paragraphs for actors,
claims, scope, comparisons, quantities, conditions, negation and causal direction.
Forty-two flagged training candidates remain held rather than being promoted on
the basis of the teacher's own alignment labels. No source or target is rewritten.
The near-copy fee/information example remains as an explicit preservation anchor;
no artificial edit-distance target or detector score is used to select pairs.

Development retains the original paper/author separation: two examples each from
Hayek 1945, Kahneman–Tversky 1979, Hansmann 1988 and Pagano–Panetta–Zingales 1998.
Seven selected development examples carry original review flags; each has an
explicit, local context disposition and alignment note in the new audit. Those
original flags are retained, not overwritten. Two other development cases are
held for an unbound systems reference and an ambiguous people/quantity referent;
eight are not selected for this small design, without being declared erroneous.
No reserved examples are opened, analyzed or repurposed, including the original
45-example Fama reserve and the separate 20-example multi-author reserve.

The assessment is an **automated source-alignment review**, not independent
semantic gold or a new PDF/OCR certification. Source checks compare the archived
human paragraphs with their teacher rewrites and preserve original extraction
uncertainty and provenance. All 24 selected pairs are provisionally eligible for
this isolated pilot, not certified for full-scale training. No new synthetic
inputs, reference corrections or silent context reconstruction are introduced.

Papers, authors and PDF hashes are separated across training and development.
These development examples were already exposed to earlier project work; they
are held out from gradient updates, not a fresh blind benchmark. An author absent
from training cannot be treated as a learned style label. This is pooled academic
rewriting, not a test of learning Fama or a specific held-out author's signature.
The high overlap limits the stylistic training signal and may reinforce copying.

The frozen private package is `data/teacher_pair_pilot_v1/`. Its audit accounts
for every candidate and preserves before-text hashes, decisions, per-pair notes,
source/target text and original flags. The decision input is under
`research_runs/teacher_pair_reverse_pilot_v1/selection_v1/`. Public counts are in
[preparation.json](preparation.json); copyrighted paragraphs and responses remain
private and excluded from Git. The source corpus and parent manifest are unchanged.

## Fixed comparison

Both modes use `google/gemma-3-4b-it` at revision
`093f9f388b31de276ce2de164bdc2081324b9767`, NF4 double-quantized weights with FP32
compute, the same ordinary rewrite instruction, native chat template, greedy
one-beam decoding, seed 1729 and at most 512 generated tokens. No target author
metadata or reference text is included in a generation prompt. Teacher-forced
loss uses the reference as a label, not as prompt content.

Before weights are loaded, all 24 tokenized source/target pairs must fit the
1,024-token training context. Prefix masking and decoded target coverage are
checked exactly apart from whitespace normalization. Every user/prompt token is
masked with -100; only the assistant target and its terminating tokens receive
supervision. No source truncation, target truncation or dynamic row dropping is
allowed. The generation prompt plus output budget must also fit the same limit.
The base run confirmed all 24 pairs fit (maximum 508 full tokens). The training
path repeats these checks. CPU checks do not download processors or weights.

The base notebook computes finite teacher-forced losses for all eight development
examples and generates ten first outputs: eight development paragraphs and two
clearly labelled training probes from different papers. It creates no adapter,
optimizer or parameter updates. The primary numeric loss is weighted by supervised
target-token count; the mean per-example loss is also recorded. Loss is not a
claim-preservation or style score.

The prepared QLoRA path uses a fresh adapter and only the 16 training pairs:

- Two epochs, batch size one, gradient accumulation two: **16 optimizer updates**.
- Learning rate 2e-5, linear decay, no warmup; rank 8, alpha 16, dropout 0.05.
- q/v projections only in every language layer. Base, vision and other parameters
  must remain frozen; matched-module counts and trainable names are validated.
- Non-reentrant gradient checkpointing, FP16/BF16 disabled on T4, one exposed GPU.
- Finite loss and positive finite adapter-gradient checks for every update;
  changed adapter weights required at completion.

The pilot requires a complete baseline artifact bound to the exact configuration
and development IDs. It recomputes a base snapshot in the new process and records
how many outputs match the external baseline rather than assuming cross-run
numerical identity. It saves both epoch adapters and snapshots after updates 8
and 16, using the same eight development examples and two training probes. Each
snapshot retains empty, copied or length-limited outputs for later assessment.
Epoch 2 is the predeclared final endpoint; do not select an epoch on these eight
cases and then report the result as fresh validation.

There are 30 pilot snapshot outputs (base, epoch 1, epoch 2) and one final reload
probe. The final adapter is unloaded/reloaded from disk; every adapter tensor
must match, and one development generation is saved after reload. This checks
saved weights and one probe, not all regenerated outputs. `evaluation.json`, label
audits, runtime/package/model hashes, gradients, training logs, checkpoints and
status preserve what actually ran. No result is claimed before execution.

The implementation follows the repository's previously exercised T4/FP32/QLoRA
pattern and the official [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization)
and [Transformers Trainer API](https://huggingface.co/docs/transformers/main_classes/trainer).
The new pilot now has [completed GPU results](PILOT_RESULTS.md). Dependencies are pinned to the
existing project set: Transformers 5.20.1, PEFT 0.21.2, Accelerate 1.15.0,
bitsandbytes 0.50.2 and huggingface-hub 1.33.0. Mac checks need none of these.

## Run the baseline first

1. Create a **private Kaggle Dataset**, titled **Teacher Pair Reverse Pilot**, with
   exactly the two files in `data/teacher_pair_pilot_v1/upload/`.
2. Import `kaggle/teacher_pair_base_4b.ipynb` as a **new private notebook**, attach
   that Dataset, enable `HF_TOKEN` in Secrets and Internet, and select **GPU T4 x2**.
   The first cell exposes GPU 0 only; use a fresh session.
3. Save & Run All, then provide the notebook's actual URL slug and version or
   download the complete `teacher_pair_base_v1` folder.

The training notebook `kaggle/teacher_pair_pilot_4b.ipynb` now uses
**`RUN_PILOT = True`** for the explicitly authorized limited pilot. Attach both
the original Dataset and reviewed baseline version 1 outputs, then follow
[PILOT_RUN.md](PILOT_RUN.md). The notebook pins the data/configuration/baseline
checksums before invoking the CLI's `--train-authorized` path. The frozen
configuration records preparation time; session authorization is separate.
Nothing is pushed, submitted or trained by the preparation tools.

GPU scope is one T4, ten base outputs, and later a capped 16-update/31-output
pilot. The observed base and pilot process times were 218.7 and 607.4 seconds
in their Kaggle runs, respectively. These do not establish future runtime or
dollar costs under different cache/hardware conditions. The step cap bounds
this as a pilot, not full training.

## Evaluation and continuation criteria

Compare base, epoch 1 and epoch 2 on identical development inputs. Assess claims
against the input and reference, and style separately. New actor reversals,
unsupported economic relationships and changed negation, conditions or causal
force are material errors; minor expression shifts may be acceptable but must
be recorded. Report exact copying and source overlap separately from target loss.
Blind the condition labels in the qualitative comparison where practical and
state automated-review limitations. No human review panel or detector API call
is required. Detector scores, if later measured sparsely, are not training rewards
or pair-selection criteria.

A lower loss alone does not justify scaling. Before continuing, look for no
increase in material-error or exact-copy counts versus base, plus meaningful
style/organization gains spanning more than one development paper. Persistent
material drift prevents scaling even if these relative criteria pass. These are
small-pilot continuation rules, not statistically powered claims of human style,
Fama style or detector evasion. Report all snapshots and negative results; do not
silently extend epochs or select better-looking cases after generation.

Reproduce the packet into a fresh directory with
`fama_style.research.teacher_pair_selection --parent --decisions --output`.
Regenerate both self-contained notebooks with `python3 scripts/build_teacher_pair_pilot.py`.
The root `train.py` and original `google/gemma-3-1b-it` training entry point remain
unchanged. Their historical three-row/five-step smoke remains a pipeline test,
not this new 16-pair/two-epoch experiment.
