# Teacher-pair QLoRA pilot: version 1 results

The authorized two-epoch pilot completed all **16 optimizer updates** on one
Tesla T4 in 607.4 seconds. Sixteen positive finite gradient checks passed, both
adapters were saved, weights changed, and disk reload reproduced every final
adapter tensor. The post-reload development probe exactly matches its epoch-2
output. The baseline recomputed in this process reproduces all ten previous base
outputs exactly. No extra inference or training was performed during this review.

The actual Kaggle slug is `oliverlds/teacher-pair-pilot-4b`, with version 1
reported by the user. Runtime/data/model hashes match the frozen packet. Both
checkpoints contain 136 FP32 LoRA tensors, 2,228,224 parameters, targeting only
language q/v projections at rank 8. The files differ between epochs. Use the
recorded pinned base revision when loading them; `adapter_config.json` does not
independently pin its base revision. Adapter artifacts and generated text remain
private.

## Fixed development comparison

| Measure | Base | Epoch 1 / step 8 | Epoch 2 / step 16 |
| --- | ---: | ---: | ---: |
| Target-token-weighted loss | 2.96942 | 2.71450 | 2.61502 |
| Material claim issues identified | 7/8 | 7/8 | 6/8 |
| Exact source copies | 0/8 | 0/8 | 0/8 |
| Median source word-sequence overlap | 0.288 | 0.290 | 0.304 |
| Median human-reference overlap | 0.226 | 0.236 | 0.245 |
| Expression preference versus base: better / tied / worse | — | 3 / 4 / 1 | 5 / 2 / 1 |

Epoch 2 reduces development loss by **11.9%**. This is evidence that adapter
updates change the model's likelihood of the references; it is not a measure of
semantic accuracy or author style. All 30 snapshot outputs and the reload probe
are nonempty and below the output limit. No copy collapse occurred in this
bounded run, though source overlap rose slightly. Two of eight development
outputs are identical between epochs 1 and 2.

The expression preferences concern **generic organization and concision**, judged
separately from claim correctness. They do not establish Fama/coauthor features,
authorship or detector performance. Epoch-2 preferences span three development
papers, but many paragraphs retain the base model's long openings and abstract
academic scaffold. Closer wording to a reference is not, by itself, style learning.

## Meaning and style move differently

At the material-issue level, epoch 2 improves two cases, worsens one and leaves
five in the same category as base:

- **Planning:** both epochs remove the invented coordination/accountability
  explanation. Coverage and scope caveats remain; the new ending is tautological,
  so this is a semantic improvement without a clear style win.
- **Loan register:** epoch 2 restores individual loans rather than portfolios as
  the observation unit and is more direct in expression. Smaller access-condition
  and purpose qualifiers remain unresolved; this is a local improvement, not
  complete equivalence certification.
- **Insurance:** both epochs improve the separation of the final framing
  conclusion but add equal potential consequences to a preference hypothesis
  originally conditioned only on equal unprotected-loss probabilities. Thus an
  expression win coincides with a new material condition error.
- **IPO, ownership incentives, reflection effect, owners as patrons and the
  price-mechanism passage:** substantial errors remain. The outputs still change
  timing or causal scope, invent mechanisms, omit numeric evidence, weaken
  quantifiers or turn an author's question into a claimed finding. One owners-as-
  patrons output also adds a new ongoing-investigation caveat.

Both training probes still have material issues at both epochs. In the monthly-
price probe, epoch 1 restores statistical-nonsignificance wording that epoch 2
weakens again, despite lower aggregate development loss. The fee probe remains
nearly unchanged and does not recover the agent/principal information assumption.
These probes diagnose learning behavior and are not held-out evaluation cases.

## Assessment procedure and limits

Each case's three labels were masked before writing 30 text-hash-bound claim
assessments and ten style rankings. Ties were permitted and retained. The key was
then used to aggregate by epoch. However, the same automated reviewer had already
seen the baseline outputs and execution/loss metadata; wording can reveal a
condition. This is **partly masked, not fully blind or independent review**.
Severity and style preferences are qualitative and not calibrated population
statistics. All per-case flags and rationales are preserved privately, including
residual caveats on cases counted as no identified material error.

The eight development cases cover four papers/authors withheld from updates but
were already exposed to project work. The 45-example Fama reserve and separate
20-example multi-author reserve are not evaluated. Epoch 2 remains the predeclared
final endpoint; the report retains both earlier snapshots rather than choosing a
better-looking epoch or case after seeing outputs.

The relative planning criteria show an early signal: no increase in exact copies,
one fewer materially problematic development output, and modest expression wins
on multiple papers. **The absolute quality gate still fails: material drift
persists in six of eight cases.** Do not scale to full training or treat this as
a successful Fama-style model. Sixteen examples and sixteen updates cannot isolate
whether data volume, learning budget or q/v-only adapter coverage is the limiting
factor. They also do not establish that QLoRA itself causes the remaining errors.

## Next recommendation

Preserve the epoch-2 adapter as a research checkpoint and stop at the authorized
budget. Before another bounded training experiment, curate more of the already
collected teacher pairs, including the 42 training candidates currently held for
review. Keep the eight development examples and reserved splits fixed. Prefer
source-checked pairs with useful expression contrast, without forcing edit
distance or selecting on detector scores. Any expanded data or revised budget
must form a separate versioned experiment; do not silently extend this run.
No further epochs, dataset expansion or new GPU job are authorized by this review.

The private download is
`research_runs/teacher_pair_reverse_pilot_v1/kaggle_pilot_v1/`. The masked bundle,
key, original assessments/style rankings, provenance note, per-output metrics
and receipts are under
`research_runs/teacher_pair_reverse_pilot_v1/pilot_review_v1/`.
The review importer verifies the complete update/snapshot coverage, old baseline,
frozen runtime, target audits, adapter headers, raw-to-masked text bindings and
ranking coverage, and refuses overwrite. Public aggregates are in
[pilot_results.json](pilot_results.json). Review code is
`fama_style.research.teacher_pair_pilot_review`. Original data, prior results and
all first outputs remain unchanged.
