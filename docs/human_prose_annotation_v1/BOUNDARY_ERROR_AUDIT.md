# Boundary mismatch audit: reference choice, missed relations and scope risks

All **42** classifier/reference mismatches were audited in source, reference and
predicted-group context: 20 extra merges and 22 missed merges across 34 training/
discovery paragraphs from 12 papers. The original labels, predictions, classifier
metrics and hashes remain unchanged. This is a post-result discovery analysis,
**not a new held-out score** or a corrected semantic-accuracy estimate.

## Findings

| Mismatch type | Evidence-based assistant interpretation | Cases |
|---|---|---:|
| Extra merge | Plausible alternative to the reference layout | 14 |
| Extra merge | KEEP preferred because of scope, pivot or density risk | 6 |
| Missed merge | Explicit/source-supported relation offers a merge opportunity | 15 |
| Missed merge | Reference merge is optional; source KEEP also reasonable | 6 |
| Missed merge | Composite claim granularity affects the gold decision | 1 |

These are unblinded coding-assistant judgments, not certified acceptability
labels. No generated rewrite was inspected, no independent verifier was called,
and no actual generation error or improvement is established by a plan alone.
Many adjacent sentences admit both faithful merging and faithful separation.
Do not subtract the plausible alternatives from the 20 false positives to report
an improved precision: the classifier's original reference-agreement scores stand.

## Extra merges: often plausible, sometimes risky

Fourteen source pairs plausibly support an alternative layout. Examples of the
underlying relations include a result with its partial explanation, a belief
with the content of that belief, an example with its scope qualifier, and a
question with its applicability to another market. These can be realized without
inventing a causal edge, provided all original qualifiers and attributions stay
intact. This is a hypothesis about realizability, not proof that a model would
produce a good paragraph.

The six KEEP-preferred boundaries belong to **five distinct predicted groups**:

- A methodological requirement is absorbed into a group of dated estimation and
  forecasting steps, crossing an obligation-to-procedure transition.
- A strategy definition and historical performance are combined with a broader
  skeptical conclusion, obscuring an argument pivot.
- Two extra boundaries belong to the same three-sentence, 73-source-word group
  combining a named risk component, a nested process description and its
  consequence. They are one group-level density concern, not two independent
  observed failures. Length alone is not an automatic rejection rule.
- A single author's discussion is joined to a finding attributed to several
  studies, creating a risk of collapsing collective attribution onto one actor.
- A reduced-propensity finding is joined across a however/although pivot to a
  qualified generalization about still-strong dividend payment.

KEEP is a conservative preference in these contexts, not a declaration that all
joined versions would be wrong. Faithful realization might preserve the scopes;
that has not been tested. Other dense groups were judged plausible, so density
was not used as a universal post hoc threshold.

## Missed merges: relation features are incomplete

Fifteen missed boundaries connect source-supported contrasts, consequences,
restatements, definitions or comparisons. Several have explicit phrases such as
“for this reason,” “put another way,” “in fact,” or “with that assumption.” The
current feature extractor does not explicitly recognize those phrases. Others
require recognizing shared actors or the same economic comparison rather than
simply counting repeated words. These are hypotheses for richer source-only
representation, not newly validated predictive features.

Six other reference merges are optional organizational choices. Technical
qualifications, section roadmaps, contributions and tentative further directions,
or weekly/monthly comparisons can remain separate without necessarily worsening
writing. A model's failure to copy the reference boundary is not by itself a
meaning or style failure.

One missed boundary divides the setup of an overlapping-return example from its
numerical consequence. Both source sentences carry the same composite claim ID.
The current gold partition is uniquely representable, but a single claim label
spans two source sentences. This makes the merge target partly dependent on
annotation granularity. It is **not a confirmed label error**: the original
reference legitimately expresses both components together, while a split source
can also be clear. No claims or sentence groups were silently repaired.

The audit identified **zero confirmed label errors**. That does not certify the
remaining transcriptions, paraphrases or taxonomies; this audit focused on the
42 selected boundaries and did not independently recheck PDFs or every claim.

## What this means for collecting data

More arbitrary human paragraphs would add volume without resolving the objective
mismatch. The immediate priority is a small **relation-aware annotation pilot**
that separates the author's layout from permissible source-only operations.
Use existing audited cases as design examples, never as fresh validation.

Recommended initial collection: **40 complete narrative paragraphs from four new
pre-2020 economics papers, ten per paper**, with different author groups and no
reuse of existing training/development/reserved papers. This is a practical pilot
size, not a power calculation or a claim of sufficient fine-tuning data. It can
remain part of this multi-author research track while Fama-specific experiments
stay separate.

Within each paper, seek five candidate paragraphs with explicit continuations
and five with potential scope/argument transitions. These are sampling strata,
not required MERGE/KEEP outcomes: do not force GPT-6 to create split sentences,
wording distance or artificial merge opportunities. Preserve already-clear cases
and naturally occurring contrasts. Include topic/length-matched candidates so
longer technical text does not become synonymous with KEEP.

| Collection stratum | Useful contrast |
|---|---|
| Premise and consequence | Existing inferential link versus merely adjacent facts |
| Restatement or definition | Safe elaboration versus a new argument stage |
| Contrast and qualification | Two comparable claims versus a broad skeptical conclusion |
| Attribution and scope | Same actor versus an individual/collective attribution switch |
| Technical or procedural chains | Compact relation versus several nested scopes/steps |

For each source boundary, future annotation should retain separate fields for:

1. The **original author's exact layout**, as provenance/reference information.
2. A source-quoted logical/rhetorical relation and claim-component bindings.
3. **Permissible operations**: KEEP, MERGE, both, or unresolved, with an explanation
   of any scope, actor, modality or attribution restriction. These are prospective
   assessment labels, not newly approved canonical taxonomy IDs.
4. Whether one semantic claim spans multiple sentences; stable component IDs can
   avoid counting a back-reference as a complete repeated proposition.
5. Candidate realization evidence and observed meaning/style tradeoffs, assessed
   separately from matching the author's layout.

Source-only judgments should be recorded before supplying reference plans where
feasible, with payload/provenance separation. UI retrieval concerns and use of the
same GPT-6 family still prevent claims of independent annotation. Automated
judgment remains uncertain; no human panel is required by this proposal.

Freeze the specification and collection strata before processing fresh papers.
Do not retrain on post hoc relabels and present the same folds as improved held-out
performance. The [40-paragraph pilot](../relation_pilot_v1/README.md) is now selected, with
Stage 1 prompt/batches and a separate Stage 2 specification prepared. Stage 2
input batches require the returned neutral text; no annotation responses exist
yet. No new generation batch,
classifier fitting, GPU training or data expansion was started by the audit.

## Artifacts, reproducibility and privacy

Private audit: `research_runs/boundary_error_audit_v1/results_v1/`.
It contains all 42 case cards grouped by paper metadata, exact source/reference
evidence, full predicted-group context, individual rationales and the assessment
snapshot. Public [aggregate results](boundary_error_audit_results.json) contain
counts and methods without paragraphs or per-example evidence.

```sh
PYTHONPATH=src python3 -m fama_style.research.boundary_error_audit --classifier research_runs/boundary_classifier_v1/results_v1 --audit research_runs/human_prose_annotation_v1/audit_v2 --assessments research_runs/boundary_error_audit_v1/assessments_v1.private.json --output /tmp/boundary_error_audit
python3 -m unittest discover -s tests -v
```

Use a fresh output directory. Manifests pin classifier predictions, source audit,
code and all assessment/artifact bytes. New CPU tests verify exact mismatch
coverage, unchanged inputs, training-only access, both source-side anchors,
reference quote occurrences, category/direction compatibility and the need for
shared-claim evidence before a granularity finding. Evidence validation establishes
mechanical consistency, not the truth of the linguistic judgment. The 20 human-
prose development rows and both reserved splits remain outside this analysis.
No credentials, external model calls, neural weights, GPU jobs or detector calls
were accessed; original experiments and metrics are preserved.
