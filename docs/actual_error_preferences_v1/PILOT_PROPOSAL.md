# Bounded proposal: preference feedback versus further chosen-only SFT

**Status: design only.** The [real-tokenizer CPU gate passed locally after a
code fix](../preference_preflight_v1/RUN_V2_RECOVERY.md).
No GPU training code, training notebook or execution approval is supplied here.
The proposal tests fidelity-oriented feedback from seventeen existing actual
errors; it does not presume preference learning will solve human/Fama style.

## Question and controls

Does explicitly contrasting a human reference with an actual erroneous adapter
response improve free-running rewriting more than showing the same chosen
references again? Compare three arms, all starting from the exact reviewed
expanded epoch-2 adapter:

| Arm | Additional optimization |
| --- | --- |
| Existing adapter | None; frozen comparison |
| Chosen-only SFT continuation | Seventeen matched prompts/references |
| DPO continuation | Same chosen exposure plus the actual rejected responses |

Both trained arms start independently from the same adapter; SFT must not start
from DPO or vice versa. The plain input is also an analysis baseline requiring
no generation: current GPT-6 inputs already receive Human Written labels from
Pangram. Returning the input could improve detector results without learning
style. Compare overlap and expression against that baseline rather than treating
detector improvement alone as evidence of writing transformation.

## Fixed small budget

Proposed settings, not an executed configuration:

- Existing pinned `google/gemma-3-4b-it`, NF4 double quantization, FP32 compute and
  rank-8 q/v adapter. The original root 1B training entrypoint remains unchanged.
- Two passes through the seventeen pairs. Each original-paragraph group receives
  one optimizer update, averaging its one or two included rows: **eleven updates
  per pass, twenty-two per arm, forty-four total additional updates**.
- Seed 1729; identical group permutations and proposed learning rate `1e-6` in
  both arms; ordinary sigmoid DPO with proposed beta `0.1`. No search or retry
  with a more favorable setting inside this pilot.
- SFT averages response-token NLL, then sibling-row losses within each group.
  DPO uses summed response log probabilities relative to a fixed reference,
  then averages sibling losses. Their numerical loss/gradient scales differ;
  the same learning rate does not make the objectives equivalent.
- Disable dropout in both continuation arms and reference scoring. Preserve
  1,024-token full sequences, exact response masks and template end tokens.
  No truncation, packing across examples or automatic row removal.
- One Kaggle T4, GPU 0. Keep chosen/rejected forwards sequential where needed to
  control memory. Stop on nonfinite loss/gradients, a broken reference identity,
  an out-of-memory error or a failed adapter reload rather than altering the
  experiment silently.

The planned forty-four updates are much smaller than a full training run. A
**20–40 minute** total runtime is an unmeasured planning estimate, informed by the
earlier roughly thirteen-minute SFT pilot; DPO adds response/reference work.
Propose a **60-minute total GPU cap** including reference-score preparation and
evaluation. A real memory measurement and initial gradient checks are required;
fit on one T4 is not yet established.

Matching updates, group weighting and chosen exposure does not match compute:
DPO also processes rejected responses. Report measured time, tokens, memory and
chosen/rejected forward counts. The same starting adapter and one seed permit a
bounded diagnostic, not an objective-family superiority claim.

## Reference policy and implementation gate

The reference is the **exact initial SFT adapter**, not raw Gemma. Cache its 34
completion log probabilities before either arm changes weights, bind them to
input IDs/masks/model revision/adapter hashes and keep them immutable. At initial
policy/reference identity, the DPO log-ratio difference should be approximately
zero and mean loss approximately log(2). Check this with real forwards before
training; synthetic algebra tests cannot certify reference identity.

Current TRL documentation supports continuing a preloaded PEFT model and
precomputing reference log probabilities. Its current source copies an existing
adapter into a reference adapter in the applicable path. Old examples with
different reference-adapter arguments must not be copied blindly. The exact
TRL release/commit and compatibility with the existing Transformers/PEFT versions
still require pinning and tests; no training implementation depends on an
assumed API here. [TRL documentation](https://huggingface.co/docs/trl/main/en/dpo_trainer),
[reference implementation](https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py).

The tokenizer notebook checks raw conversation packing against archived exact
IDs. Any eventual trainer/collator must reproduce those IDs and labels before
using its loss. Actual reference scores, policy/reference equality, finite
nonzero gradients, target-module identity and adapter save/reload remain GPU
gates. A preflight failure leaves this proposal unapproved and unexecuted.

## Relation to STAMP

STAMP combines pseudo-parallel SFT with repeated preference generation and
optimization, including hope/fear sampling and multiple quality objectives.
Its reported implementation uses **CPO**, not a plain single-pass DPO pilot,
and assumes at least 2,000 texts per style. Our seventeen correlated pairs and
single bounded continuation are a different, much smaller fidelity diagnostic.
The paper motivates testing preferences after SFT; it does not establish that
our labels or setting will work. Multi-iteration sampling, new reward models and
an attempted STAMP reproduction are outside this proposal.
[Liu and May, NAACL 2025, sections 2–3](https://aclanthology.org/2025.naacl-long.135.pdf).

## Evaluation fixed before execution

Use the original **eight-case milestone development subset**, retaining its
two paper-balanced rounds, exact texts and reference/context flags. This is not
the different recent teacher-pair eight-case list. The evaluation manifest hash
is frozen in [pilot_proposal.json](pilot_proposal.json). No development or reserved
paragraph was read to construct the preferences, and no reserved split is used
for this pilot.

Recompute the starting-adapter control under the same pinned environment and
check its generated texts against the archived eight outputs before training.
Use the same greedy decoding and 512-token generation limit for all arms. Keep
reference wording out of generation prompts. Epoch 2 is the fixed primary
checkpoint; epoch 1 is diagnostic, not an invitation to select the best result.

The existing provisional control has four material-issue outputs and three
within the local meaning budget. A useful final advance requires:

1. At most **three of eight** material-issue outputs and at least **four of eight**
   within the unchanged one-minor/no-material local eligibility budget.
2. Paired generic-expression wins exceeding losses against the starting adapter.
   Judge meaning and expression separately and preserve uncertain judgments.
3. No exact source copies, empty or token-limited outputs. Report partial-copy
   overlap, long shared spans and the unchanged-input baseline; increased overlap
   may reflect restored claims but cannot itself count as learned style.

A preference-specific benefit requires a direct advantage over the chosen-only
SFT arm. Better training margins or teacher-forced target loss alone do not
establish better generation. Both reference and actual-rejection mean/total
likelihoods are diagnostic, with lengths visible.

Only after a fixed checkpoint passes the quality gate should it receive sparse
Pangram evaluation using complete frozen paper-balanced rounds. Keep native
categories, score semantics, versions, dates and account provenance. No per-epoch
detector tuning, repeated rescan-until-pass behavior, paid tools or additional
account collection is part of this design. A successful fidelity pilot would
still not establish distinctive Fama style or the full human-style milestone.

The main failure risks are the tiny repeated-target sample, stronger chosen/input
overlap in 16/17 pairs, longer chosen responses in 12/17, provisional same-reviewer
judgments and objective/compute differences. Preserve negative or mixed outcomes
and the matched controls. Do not authorize full-scale training from this pilot.
