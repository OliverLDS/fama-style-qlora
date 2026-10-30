# Expression replication v1: four additional papers

Prepared, not executed: twelve first-attempt outputs from four additional training
papers. Two cases test each operation that was locally realized in the initial
[claim-bound expression check](CLAIM_BOUND_EXPRESSION_V1_RESULTS.md). Neither
initial paper, FJ1983 or F1980, is included.

| Paper | Operation | Grouping held constant in the primary comparison |
|---|---|---|
| F1965 | Foreground the source distribution in its stated independence relation | Two source sentences; identity KEEP/KEEP |
| ME1997 | Foreground the modeled economic variable while retaining BSV attribution | Existing author/model premise slot; seven output sentences |
| FF2000 | Finite consequence of similar merger rates | Rate comparison and unchanged payer-share consequence together; four sentences |
| VL2018 | Finite negative-premium probability consequence | Volatility, consequence and longer-horizon qualification together; two sentences |

Each case has three fresh conditions: ordinary rewriting, grouping-only control,
and selective expression guidance. The primary comparison is expression versus
grouping-only control, with identical source, evidence, grouping and semantic
reminders. Ordinary outputs give secondary context. Prior generated answers are
historical evidence, not substitute controls.

## Eligibility and selection

Selection was fixed before new outputs, using the frozen 30-case training
annotation package and source/paired evidence inspection. Four cases from four
papers were chosen for source-supported operations, two per operation. This is a
small purposive mechanism check, not a random sample or unseen-paper benchmark.
Cases have already been inspected and may have prior exposure in training or
other project work. No validation or reserved test examples are read.

Original source/reference texts remain unchanged. The new cards re-resolve exact
source spans and relation endpoints; old annotations and offsets are not edited.
Endpoint ownership is checked, but semantic eligibility is an in-session source
review, not something proved by quote matching. Interpretive relations remain
interpretive. The finite operations require an explicit causal edge with the
correct endpoint claims, direction, actor and same-slot membership.

F1965's definition-led opening can foreground the distribution participating in
the independence relation without claiming that actual price processes must be
independent. Its source/reference grouping is already identical. This ceiling
case is retained even if the baseline already suffices; it is not treated as a
packing improvement.

ME1997's opening is author-led. The source-supported economic variable can become
the theme without dropping authorship, the two judgment biases or the distinction
between actual random-walk earnings and mistaken investor regime perceptions.
The shared reminder preserves that context and the hypothetical/conditional
regime explanations in every arm.

FF2000 explicitly connects similar merger rates to an unchanged payer share.
Its count/share distinction, annual rate/cohort/year assignments and opposite
delisting effects remain protected. VL2018 explicitly connects volatility to
substantial negative-premium probabilities while retaining chance, investment
horizons and investor-evaluation context. Its longer-horizon risk must survive;
no numerical probability is supplied.

## Scope extensions, not silent changes to the initial protocol

The versioned operation scope is `v1.1_subset_of_slot_and_identity_grouping`.
No earlier compiler, package, prompt, result or default prototype was changed.

- A theme operation may target a singleton claim slot, with no fabricated companion claim.
- A finite operation may target two claims within a larger slot. VL2018 targets c2/c3 while c4 stays intact in that slot. Payloads record both full slot membership and untargeted claims.
- Identity grouping is accepted explicitly when source and reference partitions already match. It does not require inventing a merge or split.

These extend applicability without relaxing causal evidence or whole-claim
coverage. Theme instructions retain their conceptual purpose but no longer
presuppose another claim in the slot; finite instruction wording is retained.
The package records these generalizations, so it is not an exact prompt replication
of the two initial cases. Plans remain reference-informed except for the explicit
source-identity case; there is no learned source-only mapper.

## Generation files

Read `START_HERE.md` in:
`research_runs/expression_replication_v1/package_v1/`.

For each batch 01–03, attach only `batches/jobs_XX.jsonl` and
`response.schema.json`, then paste `TASK_PROMPT.md`. Use GPT-6 with High thinking
effort consistently if available, matching the prior reported setting. Save each
unchanged first-attempt `responses_XX.jsonl` under:
`research_runs/expression_replication_v1/responses/`.

Each batch contains four distinct papers; no paired versions share a batch.
Conditions rotate across batches, with every paper appearing once per condition.
Four jobs cannot divide evenly among three conditions within a batch; conditions
are balanced overall. Batch assignment/order is deterministic and fixed before
responses. Prefer separate new conversations, but cross-chat memory remains
possible. No memory clearing, chat deletion or independence claim is required.
Record actual model/effort/arrangement if different; hidden decoding is uncontrolled.

Do not upload private keys, eligibility/specification files, evaluation plans,
reports, reference paragraphs, PDFs, old answers or entire directories. Keep all
actual prose and detailed evidence outside the public repository. No credentials
are requested or accessed.

## Assessment fixed before responses

The private `EVALUATION_PLAN.md` specifies case-level checks. Assess operation
realization, source fidelity, grouping, style/clarity and copying separately.
Realizing a theme or finite clause does not by itself mean better writing. A
baseline that already realizes the form remains a valid ceiling result; do not
replace it after seeing outputs. Modest drift can be weighed against a worthwhile
benefit, with both evidenced explicitly.

Shared semantic reminders are identical across conditions. Fidelity gains from
those reminders cannot be uniquely credited to expression guidance. Compare
source and output evidence first; reference similarity is evaluation-only and
must not reward a semantic change. Report per-case and per-operation findings,
including negative/mixed outcomes, without population estimates or author-style
claims. Two additional cases per operation do not validate a classifier or mapper.

No model calls, GPU jobs, adapter training, prototype policy changes or automatic
follow-up generation are part of preparation. Actual outcomes remain unreported
until responses exist.

## CPU reproduction

Private parent artifacts and the selection specification are required. Choose a
fresh output directory:

```sh
PYTHONPATH=src python3 -m fama_style.research.expression_replication \
  --specification research_runs/expression_replication_v1/specification_v1/selection.private.json \
  --output-dir research_runs/expression_replication_v1/package_v2
PYTHONPATH=src python3 -m fama_style.research.expression_replication \
  --check-package research_runs/expression_replication_v1/package_v2
python3 -m unittest discover -s tests -v
python3 train.py --check
```

The package preserves a private eligibility ledger, original paired provenance,
source-only generation payloads, source-span anchors, scope notes and parent/job
hashes. Tests cover actor/endpoint ownership, relation direction/certainty,
subset-of-slot coverage, singleton themes, identity grouping, prompt isolation,
train-only access, deterministic twelve-job coverage and tamper detection. Public
code/tests run without GPU or model weights; private integration checks skip if
the local paper artifacts are absent. Mechanical checks do not prove semantic
alignment or expression quality.
