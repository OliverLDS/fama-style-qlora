# Purpose-aware brief pilot before full recollection

The returned five v3 briefs exposed redundant prose and source-order carryover.
The active retry is now [content abstraction v4](CONTENT_ABSTRACTION_PILOT_V4.md).
The v3 packet and responses remain historical baselines; full 500-target
recollection remains the agreed scope after the pilot protocol is settled.

The user has authorized a new round of **both brief extraction and prose
reconstruction for all 500 targets**, including the 495 previously collected.
The prior recommendation to reuse those 495 as the final inputs is superseded.
Historical data, responses and experiments remain intact as baselines.

First run a five-record pilot on the repaired targets. The active extraction
prompt is [version 3](BRIEF_EXTRACTION_V3.md), with its [JSON schema](brief_v3.schema.json).
The particular-word example has been removed. The purpose vocabulary is unchanged;
the relation vocabulary now has 36 defined labels with directions and scope rules.
The [machine-readable relation catalog](brief_relations_v3.json) accompanies it.

The finer distinctions include:

- Direct and contributing causes, enabling and preventing effects, mechanism
  steps and feedback, separated from association and statistical prediction.
- Necessary versus sufficient conditions, a particular conditional consequence,
  logical implication and counterfactual dependence.
- Similarity, contrast, quantitative comparison, tradeoffs, alternative
  explanations, substitution and complementarity.
- Supporting versus challenging evidence, tests, robustness checks and examples.
- Concept definition versus operational measurement, components and elaboration.
- Scope restrictions, exceptions and concessions; temporal sequence and change;
  an argument's reason and a proposed means to a goal.

These are interpretation aids, not a requirement to populate a large graph.
Use only labels supported by the paragraph and link only real points. Required
conditions are not automatically sufficient; chronology is not causality;
evidence does not become proof. The `qualification` field retains uncertainty,
comparison basis, applicability and other important distinctions. Moderate shifts
of emphasis remain acceptable; no numerical edit-distance target is introduced.

The current category set extends the compact paragraph-level reuse from earlier
taxonomy discovery. The refinements are proposed distinctions for this pilot,
not newly validated discoveries or claims about author intentions. Sentence-level
templates and author-specific lexical controls remain outside this brief.

## Pilot and comparison

Use the new private packet at
`research_runs/data_scaling_v1/purpose_brief_pilot_v3/`.
Upload only `stage_1_brief_batches/batch_01.md`. It covers P250, P264, P291,
P458 and P484. Save `briefs_01.jsonl` in its `responses/stage_1_briefs/` folder.
Do not upload the older version-2 five-record packet.

After the five briefs are reviewed, prepare ten paragraph reconstructions:

| Condition | Input fields | Outputs |
| --- | --- | ---: |
| Content-only | Same task, key points and anchors | 5 |
| Guided | The identical content, plus purpose, takeaway and relationships | 5 |

Both receive the same [base reconstruction prompt](BRIEF_RECONSTRUCTION_V3.md)
and requested GPT-6/High settings. A task may already imply a function, so this
tests the added value of explicit guidance, not the presence of any purpose at all.
Verify that metadata contains no substantive fact missing from the common points.
Use actual reviewed briefs, not fabricated placeholders. Keep reference prose out
of both requests; cross-chat memory isolation and independent outputs are not
assumed or certified.

Compare whether each output performs the intended communicative job, coherently
connects points, changes expression substantially and avoids material content
errors. Retain ordinary shifts of emphasis. Inspect results for added assumptions
or overconfident relations rather than enforcing exact graph realization. Five
cases provide a mechanism diagnostic, not a statistical quality estimate. The
pilot has three author groups and no Fama example; do not claim author-wide coverage.

Using prior outputs as the control would confound the comparison: P264 had missing
content, P291 was a different paragraph, and the other targets were corrected.
The two new conditions use identical repaired target content and the same brief.

After inspecting the pilot, freeze the brief/reconstruction protocol and collect
the remaining 495 under it. Reuse the five pilot annotations if that protocol is
unchanged. If it changes, record the difference before deciding their inclusion.
Full recollection is authorized in scope, but the bulk upload packet is deferred
until this small pilot resolves prompt revisions. No provider calls, GPU training
or detector scans are initiated by this preparation.
