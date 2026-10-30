# Claim-bound expression v1: prepared diagnostic

Six outputs are requested from the two bounded repaired training cases. No model
call, training, new output result or author-specific benefit is claimed.

| Paper | Ordinary | Grouping-only control | Same grouping + expression guidance |
|---|---:|---:|---|
| FJ1983 | 1 | 1 | Foreground the source-supported decision-control structure |
| F1980 | 1 | 1 | Give the supported consequence an explicit actor and finite predicate |

The primary contrast is expression guidance versus grouping-only control within
each paper. The ordinary outputs provide a secondary organization baseline. All
six are new first attempts; previous answers are historical context only.

Both matched arms share the exact source, evidence, style instruction, semantic
reminders and oracle grouping plan. The expression arm adds one bounded operation
block. FJ1983 retains its three output slots; F1980 retains four, including the
final two-claim slot. In the ordinary arm, grouping is unconstrained. Shared
reminders appear in all arms, so these prompts are not byte-identical to the old
ordinary baseline. Any reminder-induced semantic gain must not be attributed to
the expression operation.

## Operations and scope

FJ1983's source-bound focus belongs to c1 and is applied within its c1/c2 slot.
It changes grammatical presentation of an existing core relationship. It cannot
supply a new centrality judgment, universal scope, decision stage or definition.
Ratification meaning/sequence, organizational scopes, mutual monitoring and its
origin in collaborative work remain part of the shared semantic reminder.

F1980's consequence operation requires the existing explicit causal edge from c5
to c6, with evidence owned by those endpoints. The consequence actor is anchored
inside c6. It must remain in the same output sentence, distinguish qualified
current-pay effects from asserted future-wage effects, and retain causal direction.
No causal edge is invented from mere co-location. Giving the consequence a finite
predicate is a local expression preference, not a universal style rule.

The operations come from the [residual style audit](RESIDUAL_STYLE_AUDIT_V1.md),
which did not establish author specificity. Only one operation is tested per
paper; neither is replicated across papers in this check. Source-label reuse is
deferred because its selected empirical support came from other cases. No new
label glossary or target wording is supplied to the generator.

Only the two repaired, bounded cases are selected. NL2003 and ME1997 remain
qualified and are excluded. Plans/evidence remain reference-informed oracle
information, not deployment-time predictions. The existing prototype, annotations,
corpus, earlier experiments and reserved validation/test examples are unchanged.

## Manual generation

Read the private package's `START_HERE.md` at:
`research_runs/claim_bound_expression_v1/package_v1/`.

For each batch 01–03, attach only `batches/jobs_XX.jsonl` and
`response.schema.json`, then paste `TASK_PROMPT.md`. Use GPT-6 with High thinking
effort consistently if available, matching the reported prior setting. Prefer
separate new conversations and preserve first attempts. No batch contains both
versions of a paper. Cross-chat memory, batch effects and hidden decoding remain
uncontrolled; new chats do not establish independence.

Save unchanged `responses_01.jsonl`, `responses_02.jsonl`, `responses_03.jsonl`
under `research_runs/claim_bound_expression_v1/responses/`. Optional `notes.txt`
can record actual model, visible effort, dates and chat arrangement. Do not upload
private keys, reference text, reports, PDFs or old responses. Keep all actual
paragraphs and generated responses outside the public repository.

## Assessment prepared before generation

Assess separately:

- Actual expression-operation realization within the intended slot.
- Claim fidelity, including local scope, modality, attribution and causal origin.
- Nonpacking style/organization change beyond matching the sentence count.
- Copying and evaluation-only reference similarity.

Exact evidence should support judgments. Modest drift may be weighed against a
worthwhile benefit, with both disclosed. Mechanical anchors, count matching and
connective presence do not establish semantic fidelity or author style. Any
source-meaning improvement from the shared reminders applies to the whole new
comparison, rather than proving the expression operation worked.

If guidance offers no useful gain, revise realization/controls before fitting a
mapper or adapter. A positive result would justify replication on more eligible
papers, not immediate training or an author-style claim. A discriminative style
task still needs comparable non-Fama economics prose and appropriate controls.

## CPU reproduction

With private parents/specification available, choose a fresh output directory:

```sh
PYTHONPATH=src python3 -m fama_style.research.claim_bound_expression \
  --specification research_runs/claim_bound_expression_v1/specification_v1/operations.private.json \
  --output-dir research_runs/claim_bound_expression_v1/package_v2
PYTHONPATH=src python3 -m fama_style.research.claim_bound_expression \
  --check-package research_runs/claim_bound_expression_v1/package_v2
python3 -m unittest discover -s tests -v
python3 train.py --check
```

Parent package/review hashes, source-owned focus/actor evidence, fixed grouping
membership, explicit relation endpoints and prepared job hashes are verified.
Tests reject unsupported edges, misbound endpoints/actors, altered grouping and
package tampering. They check deterministic six-job coverage and that primary
prompts differ only by the expression block. Private integration tests skip when
local paper artifacts are unavailable. No GPU or model weights are required.
