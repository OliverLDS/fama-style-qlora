# Nonidentity grouping diagnostic — preparation protocol

This preserves the preparation protocol. The 15 responses have now been reviewed;
see [completed findings](GROUPING_RESULTS.md).

The first human-prose oracle check found no consistent incremental guidance
benefit, but six of eight source/reference pairs already shared the same claim
grouping. This follow-up selects **all five remaining development pairs** with
different groupings, from three papers. It excludes all previous realization
examples and all reserved examples. Selection uses annotations and previous
selection IDs, not generated response quality. These are unused realization
examples, not pristine unseen paragraphs: the texts were already annotated.

## Controlled comparison

Each example gets three conditions, requesting 15 outputs total:

| Condition | Presentation request |
|---|---|
| Ordinary | Choose sentence organization freely |
| Source-group plan | One sentence for each source claim group |
| Oracle target-group plan | One sentence for each reference claim group |

All three receive identical source text and exact source-quoted claim evidence.
The explicit plans differ only in group assignments. No target lexical controls,
reference prose, original-derived propositions, entity descriptions, paper IDs,
author names or free-text target explanations are exported to the generator.
References and paired annotations remain private. Quotations and citations in
the source can still reveal provenance. The oracle condition still uses target
information unavailable at deployment.

The common task requests natural realization of one sentence per group, preserving
all source-supported logical relations. Meaning/readability deviations are
permitted and must be reported in separate `plan_status`/`plan_note` fields. Minor
wording changes are allowed with the same semantic instructions across conditions;
a request to merge claims never licenses a new causal relationship. Self-reported
adherence or deviation is not used as a verified evaluation score.

Every target has fewer sentences than its source: 3→2, 5→4, 6→4 twice and 6→5.
This tests **sentence merging/grouping**, not splitting, learned mapping, the full
three-level taxonomy, or distinctive Fama/coauthor style. Two source annotations
spread a single claim over two sentences. Repeated claim IDs are intentional
there; prompts ask for relevant components without wholesale claim duplication.

Three counterbalanced batches contain five jobs each, with exactly one variant
of every paragraph per batch. Each condition appears five times overall and one
or two times per batch. Job IDs mask condition/source keys and jobs are sorted
by their deterministic hashes. Condition presence is apparent from the payload,
so this does not blind the generating model. GPT-6 with High thinking effort is
requested to match the earlier check. Separate chats can reduce direct carryover
but do not establish independence from retrieval or pretraining. Actual settings
and context must be reported without inventing hidden decoding parameters.

## Risks and prospective assessment

Source text, annotations and earlier experiments are unchanged. All five contain
annotation/context uncertainties. In one source, “those people” may refer either
to the small informed group or the much larger responding population; the
original attaches the identity qualification to the larger population. This
preexisting ambiguity is recorded privately, not silently fixed or charged to
the realizer. Source-quoted evidence was selected by the original paired annotator
and cannot independently settle intended reference. No exhaustive semantic
verification or training approval is claimed.

After responses return, inspect all 15 outputs without requiring human annotation:

- Check exact IDs, JSONL coverage, nonempty paragraphs and the returned deviation
  fields; preserve raw bytes and model/context provenance.
- Compare target-group versus source-group performance to isolate nonidentity
  grouping, and target-group versus ordinary to measure added value beyond free
  rewriting. Source-group versus ordinary measures general explicit-plan effects.
- Assess observed claim-to-sentence assignments, coverage and unsupported logical
  links. Report sentence-count adherence separately; equal counts do not prove
  the requested claims were grouped together. Compare observed deviations with
  the model's self-report.
- Keep copying, readability, expression benefit and drift separate. Judge new
  drift against the supplied source, with original-reference alignment and input
  defects reported separately. Do not reward lower overlap alone.

A useful mechanistic signal would be that both explicit plans are followed in
at least four of five examples, with target grouping attained more often under
the target plan than under either control. This is a small-sample screening rule,
not a statistical superiority claim or automatic training authorization. Even
such a signal warrants structural learning only if the changed organization
also produces useful expression and an acceptable documented meaning tradeoff.
Minor shifts do not automatically invalidate a useful output; report their
concrete cost alongside the benefit. A positive result establishes at most
controllability on these cases, not distinctive human or author style.

If explicit groups still are not realized, investigate prompt actionability,
claim granularity and realization before learning a mapper. If groups are followed
without useful expression gains, retain ordinary rewriting and focus on pair
quality rather than training a larger taxonomy mapper. No GPU training or test
split access is required for this diagnostic.

## Private manual packet and reproducibility

Active package: `research_runs/human_prose_grouping_v1/package_v2/START_HERE.md`.
The earlier `package_v1` is a retained draft; use **package_v2**. For each batch,
paste its `TASK.md` and attach only `inputs.jsonl`. Save first complete unedited
responses as `responses_01.jsonl` through `responses_03.jsonl` in the active
package's `responses/`. Record retries or changed settings. The four-field output
schema is included in the package. `SESSION_NOTES.md` is a mutable blank user
record and intentionally excluded from the generated-file hash manifest.

```sh
PYTHONPATH=src python3 -m fama_style.research.human_prose_grouping --audit research_runs/human_prose_annotation_v1/audit_v2 --previous-package research_runs/human_prose_realization_v1/package_v1 --output /tmp/human_prose_grouping
python3 -m unittest discover -s tests -v
```

Use a fresh output directory. The builder verifies upstream manifests and freezes
selection, source/plan payloads, output schema, prompts and upstream/code hashes.
CPU tests cover development isolation, exclusion of used and identity cases,
unique selection, exact source evidence, unknown/incomplete groups, deliberately
repeated cross-sentence claim IDs, reference-wording omission, matched semantic
inputs, deterministic jobs and counterbalanced batches. Public preparation counts
are in [grouping_preparation.json](grouping_preparation.json). Private prose stays
ignored by Git. No model generation, training, GPU job or detector call has run.
