# Input replacement: no measured improvement over the matched control

Completed as `oliverlds/input-replacement-4b`, requested version 1. The run passed
the common-base gate, initialized a fresh adapter, completed **46 finite updates**,
saved both checkpoints and verified the final adapter reload. No further model
calls, training or GPU submissions were made during this review.

This particular twelve-input substitution did **not** meet the predeclared
success criteria. Final claim issues remain **six of eight**, loss is almost
unchanged, and direct generic-expression preferences favor the control in two
cases versus one treatment win. This is a negative/mixed result for this
specific small intervention, not evidence that all paraphrase augmentation fails.

## Matched-budget results

| Condition | Updates | Development loss | Material-issue development outputs |
| --- | ---: | ---: | ---: |
| Original-input control, epoch 1 | 23 | 2.19141 | 6/8 |
| Replacement, epoch 1 | 23 | 2.19555 | 7/8 |
| Original-input control, epoch 2 | 46 | 1.89590 | 6/8 |
| Replacement, epoch 2 | 46 | 1.90146 | 6/8 |

Loss is target-token-weighted, recomputed/validated under the same evaluation
setup. The final replacement loss is **0.293% higher** than control; this small
single-seed difference is not a statistical significance claim. Both models
have zero exact source copies, empty outputs or length-limited outputs.

| Replacement vs matched control | Expression wins | Ties | Losses | Identical development outputs |
| --- | ---: | ---: | ---: | ---: |
| Epoch 1 | 1 | 6 | 1 | 4/8 |
| Epoch 2 | 1 | 5 | 2 | 3/8 |

The replacement final epoch has seven generic expression wins and one tie
against base, compared with control's six wins, one tie and one loss. That
indirect comparison would hide the direct result: the treatment loses more
expression comparisons than it wins against the matched control. Neither set
of preferences measures a distinctive Fama/coauthor signature.

## Per-case changes

At epoch 1, replacement worsens the planning case: it adds coordination and
accountability difficulties as the reason organized-industry planning is
disfavored. The control's epoch-1 account did not add that explanation. Both
retain the insurance case provisionally, and the other six cases remain flagged.

At epoch 2, the treatment removes control's unsupported planning-efficacy and
causal-uncertainty coda. That is one claim-preservation improvement and the
clearest direct expression win. However, its bank-register output reintroduces
the unsupported **solely internal monitoring** restriction. The correct
individual-loan threshold and request-based access remain, but that extra
purpose restriction regresses from the control's final account. The net
material-issue count is therefore unchanged.

Other failures persist:

- IPO output still changes subsequent investment into planned investment.
- Investor-ownership output adds profit/risk-mitigation or other unlicensed
  explanations; one final output is exactly identical to control.
- Reflection-effect outputs are identical to control at each matched epoch,
  retaining omitted numerical indifference examples and scope changes.
- Cooperative ownership still weakens most into frequent/significant shares
  and alters the analytical purpose or its force.
- The price-mechanism paragraph still omits the deliberate rhetorical purpose,
  strongest counterfactual ranking and skeptic position. Its new final opener
  adds an unsupported apparent-success explanation for complacency.

The two unchanged training probes reproduce the control outputs exactly at
both epochs. Their final core information relationships are retained, with the
same minor wording/caveat flags. They are training diagnostics, not extra
development examples or new evidence of generalization.

## Predeclared gate

| Criterion | Result |
| --- | --- |
| Fewer than six final material-issue outputs | **Fail: six** |
| At least six final generic-expression wins over base | Pass: seven |
| At least as many direct expression wins as losses vs control | **Fail: one win, two losses** |
| No increase in copying, empty or length-limited outputs | Pass |

The aggregate gate fails. Do not select a winner solely by loss or by an
improvement in one example, extend this adapter automatically, or promote this
dataset as a proven improvement. Preserve the original-input control and both
replacement checkpoints with the mixed findings.

## Integrity, review and limits

The common base reproduced **all ten generation records exactly**, and the
maximum per-example development-loss difference was **0.0** before gradients.
All 54 token/label checks pass, with a maximum full sequence of **550/1,024
tokens**. Zero-initialized LoRA B matrices verified fresh initialization; no
old adapter was used to start training. All 46 gradient checks are finite and
positive, weights changed and all final adapter tensors match after disk reload.
The reload generation also matches its saved final probe.

Each checkpoint has 136 F32 language-only rank-8 tensors / 2,228,224 parameters.
New checkpoint hashes differ from each other and the original-input control.
The downloaded runtime/data/config hashes and approved 46-update scope match
the frozen packet. All targets, row ordering, target frequencies, metadata,
development rows, probes, training arguments, LoRA settings, seed and decoding
are preserved. Only the twelve training inputs and corresponding prompts/hashes
change, including their wording, organization and token lengths. The comparison
does not isolate lexical overlap alone.

Execution took **769.0 seconds (12.82 minutes)** on one T4, with peak GPU
allocation **6,567,964,160 bytes**. The log has no Python traceback; it includes
a pip resolver warning, but the complete model run succeeded. Downloaded text
artifacts/logs were scanned without printing credentials; no credential patterns
were found. No dependency repair or rerun was needed for this review.

Five conditions—common base, both control epochs and both treatment epochs—were
shuffled independently per case before new output/score inspection. All 50
text-bound judgments and ten five-way style rankings were saved before
unmasking or reading loss trends. **41 judgments reuse exact historical
evidence; nine newly distinct outputs were assessed**. Identical outputs must
tie on style, and reused judgments must retain their exact prior text hash,
material flag and notes.

The reviewer previously saw the historical outputs, created the source variants
and could infer conditions. This is partly masked same-reviewer automated
assessment, not independent human review or calibrated semantic gold. Minor
wording and coverage flags remain on provisionally acceptable cases. The same
eight project-exposed development cases are held out from gradients; no reserved
tests are used. A single-seed, small-case result does not establish general
failure of input variation, a Fama signature, human authorship or detector gains.

## Next preparation

The evidence does not support another unstructured SFT wording tweak or more
epochs by default. A more specific data-only next step is to construct
**preference candidates from actual unfaithful training-side generations** in
the already completed twelve-anchor inference check, rather than generate more
near-identical synthetic errors that the model usually ranks below the target.

Use the unchanged human paragraph as a candidate chosen target and a saved
material-error generation from the same input as a candidate rejection. Bind
the rejected claims to the original ledger, retain full provenance, exclude
borderline/minor-only cases and truncated/empty artifacts, and keep faithful
model outputs as controls. Do not assume every generated paragraph is bad,
force a fixed number of negatives, or use the eight development generations as
training data. Audit length, copying and generic-register shortcuts before
claiming the pairs teach semantics rather than superficial preference.

This would prepare a different supervision signal, not demonstrate that
preference optimization will help. It needs its own assessment and a bounded
feasibility/training proposal before any GPU execution. **No new preference
data, optimizer or training run is implemented or authorized by this review.**

Private raw outputs and log:
`research_runs/teacher_pair_input_replacement_v1/kaggle_v1/`.
Masked evidence, assessments, provenance, per-output metrics and receipts:
`research_runs/teacher_pair_input_replacement_v1/review_v1/`.
Public aggregates: [results.json](results.json).
CPU review implementation: `fama_style.research.input_replacement_review`.
