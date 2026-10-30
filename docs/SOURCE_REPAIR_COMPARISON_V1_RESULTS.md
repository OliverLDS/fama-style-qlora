# Source-repair comparison v1: mixed findings

**Repairing meaning improves the economics available in the output, but does not
solve conservative rewriting or establish distinctive Fama/coauthor style.** All
16 first-attempt outputs were accepted. Plans changed the intended grouping in
both source versions; the repaired source did not clearly reduce generation drift.

The user reports GPT-6 with High thinking effort for the four returned files.
Hidden decoding and model identity are not independently verified. Conversation
arrangement was not reported and cross-chat memory cannot be excluded. No new
model call, GPU job or training run was started during this review.

## What changed

| Condition | Outputs | Mean input/output overlap | Mean output/reference overlap | Requested grouping matched |
|---|---:|---:|---:|---:|
| Original + ordinary | 4 | 0.872 | 0.423 | 0/4 |
| Original + planned | 4 | 0.863 | 0.428 | 4/4 |
| Repaired + ordinary | 4 | 0.854 | 0.449 | 0/4 |
| Repaired + planned | 4 | 0.831 | 0.459 | 4/4 |

Overlap is normalized-token SequenceMatcher similarity, not entailment, author
style or a calibrated copying score. No output is an exact normalized copy;
two of sixteen exceed 0.90 overlap with their own supplied input. Changes remain
conservative even when they do not cross that arbitrary threshold. Token edit
distances and overlap against **both** input versions are retained privately.

Mean input/reference overlap itself rises from 0.459 to 0.512 after repair.
Output/reference overlap rises by only 0.026 under ordinary rewriting and 0.031
under plans. These lexical changes cannot establish improved generation beyond
better input content. They also cannot negate semantic recovery when phrasing
differs: reference wording is not a gold requirement for source-faithful output.

The aggregate input/output-overlap decrease conceals differences by paper. In
both bounded cases, ordinary repaired outputs are actually closer to their
inputs than ordinary original outputs (approximately +0.061 for F1980 and +0.067
for FJ1983). Planned counterparts decrease by approximately 0.022 and 0.027.
There is no general demonstration that source repair discourages copying.

## Bounded cases

**F1980:** Both repaired outputs retain the human-capital rental asset, manager/firm
roles, rental rate, labor-market signal and qualified performance dependence.
Original outputs faithfully retain their narrower valuation/compensation version
instead. They should not be penalized as generators for failing to invent the
missing rental mechanism. Planning combines the opening claims and the coach/stake
consequence. Ordinary rewriting already combines the latter, while planned
rewriting still uses a participial consequence. A distinctive expression gain is
not established.

**FJ1983:** Repaired outputs retain the common-core/all-types formulation,
ratification definition, large-organization scope, mutual monitoring and human
capital. Both planned outputs combine hierarchy and operating mechanism. However,
the monitoring-origin relationship becomes an accompanying-context clause rather
than explicitly stating that monitoring arises from collaborative work. This is
an exploratory minor relation-weakening judgment in both input arms. The repair
therefore supplies missing meaning without guaranteeing its full realization.

## Qualified cases

**NL2003:** Both repaired outputs propagate a decline rather than merely a lower
profitability level. The decline comparator is still unspecified, and wording
about change over the period may invite a within-period interpretation. That
scope concern is unresolved, rather than silently scored as a verified error or
reference recovery. No missing comparator was supplied.

**ME1997:** Repaired outputs make the author's qualified presumptions explicit.
The planned repaired opening also shifts a classification into language that can
suggest causal attribution; this is a possible small force shift, not a proven
material error. Both variants preserve the supplied gradual-correction condition.
The unchanged reference's gradual-recognition condition remains different and is
not used to penalize source-faithful wording or credited as a style feature.

## Semantic and style assessment

The review binds all 84 claim instances to exact source/output evidence. Of these,
81 were judged retained, two had minor relation weakening, and one had a possible
force shift. These are in-session assistant judgments, not independent entailment
labels or an accuracy estimate. Broad sentence anchors establish traceability,
not atomic semantic completeness. Local ambiguities remain in the notes.

All eight planned outputs match the reviewed claim grouping, versus none of the
eight ordinary outputs. This verifies grouping, not flawless semantics or style:
the monitoring-origin weakness persists despite matched grouping. No material
new drift was established in the ordinary outputs, so there is no demonstrated
reduction of generation drift from fixing the inputs. Source/reference mismatch
and generator drift must remain distinct.

Beyond grouping, changes mainly involve clause attachment, explicit presumptions,
connectives, citation formatting and synonyms. Theme, finite-consequence form and
stable-label use do not show a convincing additional author-specific gain. These
expression operations were not introduced as new instructions in this experiment.
The findings support better source specification as a prerequisite, not a solution
to author-style transfer or justification for new QLoRA training.

## Next experiment

Test selected claim-bound expression operations on the two bounded repaired cases,
against the same ordinary/shared-evidence baseline. Preserve the current grouping
contrast separately, and disclose local drift alongside any substantive style gain.
Use only operations supported by existing source relations and definitions.

A discriminative author-style task remains a separate useful proposal, requiring
comparable non-Fama economics prose and paper/topic/coauthor controls. The current
experiment neither supplies that comparison corpus nor establishes a source-only
mapper. Retrieval or classifier fitting should wait for evidence that the candidate
operations improve realization. The 21 validation and 45 reserved test examples
remain unused.

## Reproducibility and privacy

The prepared package and its implementation are unchanged. A separate importer
verified prompt IDs/hashes, batch assignments, unique coverage and raw-file hashes;
all original responses were preserved byte-for-byte. Import-only artifacts remain
separate from evidence-assessed results:

- `research_runs/source_repair_comparison_v1/import_v1/`
- `research_runs/source_repair_comparison_v1/review_spec_v1/`
- `research_runs/source_repair_comparison_v1/review_v1/`

Private files contain source/reference paragraphs, outputs, claim evidence and
per-case judgments. Public files contain code, methods and aggregate findings
only. Original corpus, repair overlay, annotations and prior outputs are unchanged.

```sh
PYTHONPATH=src python3 -m fama_style.research.source_repair_review \
  --package research_runs/source_repair_comparison_v1/package_v1 \
  --responses research_runs/source_repair_comparison_v1/responses \
  --assessments research_runs/source_repair_comparison_v1/review_spec_v1/assessments.private.json \
  --output-dir research_runs/source_repair_comparison_v1/review_v2
```

Choose a fresh output directory. Omitting `--assessments` imports surface diagnostics
only and does not report semantic/style assessment. CPU tests check assessment
identity, complete claim coverage, exact evidence, grouping associations, edit
distance, deterministic reproduction and raw preservation. They do not certify the
substantive judgments. Four inspected training cases, oracle plans/evidence,
uncontrolled hidden settings, possible memory/source familiarity and nonindependent
assessment limit any generalization or causal interpretation.
