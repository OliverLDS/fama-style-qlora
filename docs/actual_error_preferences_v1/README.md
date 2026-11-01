# Actual-error preference candidates: seventeen pairs, no training

The [real CPU tokenizer/mask gate now passes](../preference_preflight_v1/RUN_V2_RECOVERY.md)
after local recovery using the exported tokenizer. The frozen candidate packet
is unchanged; the versioned recovery report resolves its historical pending
input gate. The [bounded preference-versus-SFT proposal](PILOT_PROPOSAL.md) remains
design-only; no GPU training is implemented or approved.

Prepared a private training-only packet from the existing expanded epoch-2
adapter's first greedy generations. It contains **17 candidate preferences from
eleven paragraphs across ten papers**, plus a matched 17-row chosen-only SFT
control. It is not a training release, a completed preference experiment or
evidence of improved Fama style or reduced detectability.

## Selection and review

All 24 adapter outputs from the twelve semantic anchors were considered: an
original input and a meaning-preserving variant for each anchor. Base-model
outputs and hand-edited synthetic negatives were not used as rejections.

| Decision | Outputs |
| --- | ---: |
| Include: clear material error, locally aligned reference, provisional expression preference | 17 |
| Hold: uncertain boundary judgment | 2 |
| Exclude: no clear material error | 5 |

The six outputs previously counted as nonmaterial comprise five exclusions and
one uncertain held case. The other held case was previously marked material but
with uncertain severity. Neither held label was converted into a confident
training rejection. One time-family paragraph has no included outputs because
both alternatives retain the core claims; family balance was not restored using
invented errors.

| Included contrast family | Pairs |
| --- | ---: |
| Scope | 6 |
| Time | 3 |
| Numbers and comparisons | 5 |
| Reasoning | 3 |

Nine pairs use original inputs and eight use the existing variants. The chosen
response is always the **unchanged original human training reference**. The
rejected response is always the exact saved adapter generation. No target repair,
new paraphrase, generation, detector score or user annotation was needed.

Source-first alignment was checked against both inputs, the original reference
and all 56 claims across the twelve anchors. Every claim's existing source,
variant and target evidence span is present. New review notes separately describe
the reference's organization, stable terminology, contrast or conditional
structure, and its provisional expression preference over the rejected output.
The labels are not an assertion of a distinctive Fama signature.

Eight included pairs retain inherited review flags. Some source paragraphs
refer to unavailable tables, sample periods or earlier discussion. Local
source-to-target alignment is supported without reconstructing those contexts;
the underlying empirical truth, OCR and surrounding-paper interpretation are
not independently verified. One unchanged target retains the noted
`considered-up to` typography limitation. These limits remain in the packet,
not silently cleared by the new review.

The same Codex reviewer constructed variants and knows earlier outputs. This
second pass is automated, not independent human review or semantic gold. The
original severity-boundary uncertainty is preserved and excluded. Paragraph-level
decisions and their reasoning are private and bound to source, reference,
generated-output and claim-ledger hashes.

## Important confounds

The chosen reference has more source lexical overlap in **16/17 pairs**. Median
source/chosen word-sequence overlap is **0.593**, versus **0.364** for the rejected
output. Neither side is an exact source copy, but preferring these references
could still encourage excessive partial copying. This packet contains no clear
actual-copying rejections and does not directly solve copying behavior.

Chosen responses are longer in **12/17 pairs**. The median chosen/rejected word
ratio is **1.045**, ranging from **0.662 to 1.364**. Some references restore omitted
supporting claims; other references express the argument more concisely. Length
is not a sufficient semantic or style-quality rule. Meaning, reference wording,
generic exposition, source overlap and length differ together, so these are
**fidelity-oriented joint preferences**, not isolated style-training evidence.

All human targets and original inputs were already seen by the SFT adapter.
Variants are different wording of the same underlying content. A gain after
another optimization phase could result from extra target exposure. The matched
chosen-only SFT file makes that explanation testable; it does not itself declare
that SFT and a preference objective have equivalent compute costs.

## Grouping and future control

Every row retains original paragraph, paper and author-group identities. Original
and variant siblings remain in the same group. The default weighting proposal
assigns one total unit per original paragraph, divided across its included rows;
alternative paper-balanced weights are stored separately. They are not multiplied
together automatically. Do not describe seventeen rows as seventeen independent
examples or randomly split siblings into training and validation.

The matched SFT control has exactly the same prompts, chosen targets, row order
and proposed group weights. Before a future optimizer comparison, specify the
same starting adapter and freeze chosen-target exposure, seeds, update budget,
length handling and compute accounting. Preference objective and hyperparameters
are intentionally not selected by this data-preparation task.

All candidates remain training-side. Parent membership is checked against the
46-pair training payload; development and reserved bodies are not parsed. Existing
datasets, splits, targets, adapter hashes and historical findings remain intact.

## Readiness and next gate

CPU checks cover provenance, full source/reference bindings, claim spans,
conservative exclusions, grouped identities, exact actual rejections and matching
SFT targets. Archived token evidence covers up to **483 tokens** for chosen full
sequences and **471 tokens** for prompt plus generated response among included
rows. These counts are useful budget evidence under the recorded tokenizer and
generation setup; they are **not a fresh preference-training mask/token audit**.

The earlier likelihood check scored synthetic one-edit negatives, not these
actual free-running responses. Actual-rejection completion log probabilities,
prefix boundaries, end tokens and masks need checking under the eventual
training implementation. No model weights or tokenizer were downloaded here.
`training_ready=false` and `training_authorized=false` remain explicit.

The next research decision is whether these seventeen fidelity-oriented pairs
justify a bounded preference-versus-additional-SFT pilot despite their overlap
and length confounds. They do not establish that preference optimization will
repair claim drift or learn human/Fama style. If such a pilot is proposed, keep
meaning, expression, copying and sparse detector endpoints separate, use unchanged
development cases and keep the reserved split sealed. No GPU job is authorized
or started by preparing this packet.

## Artifacts and reproduction

Private packet: `data/actual_error_preferences_v1/`:

- `preferences.private.jsonl`: seventeen actual-error candidate pairs.
- `sft_control.private.jsonl`: matched chosen-only controls.
- `decisions.private.jsonl`: all 24 inclusion, exclusion and hold decisions.
- `reference_alignment.private.jsonl`: twelve source/reference alignment audits.
- `diagnostics.private.jsonl`: overlap, length, numeric and archived-token evidence.
- `groups.private.json`, `provenance.json`, `receipt.json`, `manifest.json`.

Private review: `research_runs/actual_error_preferences_v1/review_v1/`.
The initial packet before adding archived-budget diagnostics is preserved under
`research_runs/actual_error_preferences_v1/initial_packet_before_budget_audit/`.
The final packet reproduced byte-for-byte. Public aggregates are in
[summary.json](summary.json) and the prospective [protocol.json](protocol.json).
No copyrighted paragraphs, generations or private decisions are published.

```sh
PYTHONPATH=src python3 -m fama_style.research.actual_error_preferences \
  --training data/teacher_pair_pilot_v2 \
  --anchors data/teacher_pair_semantic_anchors_v1 \
  --packet data/semantic_anchor_inference_v1 \
  --download research_runs/teacher_pair_reverse_pilot_v2/semantic_anchor_inference_v1/kaggle_v1 \
  --prior-review research_runs/teacher_pair_reverse_pilot_v2/semantic_anchor_inference_v1/review_v1/results \
  --bound-review research_runs/actual_error_preferences_v1/review_v1/bound_review.private.json \
  --output data/actual_error_preferences_v1
```

Use a fresh output path when reproducing; the builder refuses overwrites.
There is no training configuration or new Kaggle notebook in this preparation.

Validation: 636 CPU tests passed, including actual-output binding, conservative
exclusions, unchanged target/claim checks, archive tampering, grouped controls,
reproducibility and leaving development bodies unread. The original training CLI
dry run also passed without downloading or training a model.
