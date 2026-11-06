# Retry five pairs with content abstraction v4

The five returned v3 briefs are valid records, despite their schema-like filename.
They also retain presentation traces: P291's five sentences become five notes in
the same progression, while P484's four notes follow its four original sentences.
The task, takeaway, points, anchors and relation explanations repeat information.
Across those fields, P291 has approximately 284 words against 79 in the original;
P484 has approximately 318 against 92. These are observations about annotation,
not measured copying or author-style behavior in generated outputs.

The user requested refining abstraction and retrying these same five pairs before
the full round. The active [v4 extraction prompt](BRIEF_ABSTRACTION_V4.md) and
[schema](brief_v4.schema.json) supersede v3 for this retry. Earlier artifacts and
raw responses remain intact, including an unchanged-byte baseline copy.

## Revised brief

- A short neutral `topic` replaces the free-text task.
- `purpose` retains primary/secondary function labels and removes the polished
  reader-takeaway sentence.
- `key_points` are compact subject-headed fragments, regrouped by economic
  entity/concept and alphabetically indexed. They are not an original-sentence
  outline. Content can be combined across sentences or split across subjects.
- `anchors` are a dictionary of compact facts with units, scope and citation roles.
  Notes refer to @A1, @A2, etc., avoiding repeated narrative statements.
- The same 36 nuanced relationship labels remain available. Their qualifications
  are short scope/modality fragments, and links capture economic dependencies
  rather than the paragraph's sentence transitions or rhetorical staging.

Alphabetical indexing alone cannot remove all structural information. Grouping
and note formulation still require assessment. High-level purpose, causal links
and evidential scope also necessarily preserve information about the original
account. The intent is to let realization choose a new presentation while keeping
that account useful, not to eliminate every trace of its logical structure.

Compactness is a soft goal. There is no word-change percentage, rigid note count,
strict length admission rule or forced removal of technical terms. Necessary
conditions, quantities and other central content are retained. Some compression
and shifts of emphasis remain acceptable.

## Manual retry

Upload only:

`research_runs/data_scaling_v1/content_abstraction_pilot_v4/stage_1_brief_batches/batch_01.md`

Use GPT-6 with requested High effort. The file includes instructions, relation
definitions and the same corrected targets/hashes as v3: P250, P264, P291, P458,
P484. No earlier responses or additional schema attachment are needed. Prefer a
fresh request; cross-chat memory isolation is not assumed.

Save `briefs_01.jsonl` in:

`research_runs/data_scaling_v1/content_abstraction_pilot_v4/responses/stage_1_briefs/`

The original five v3 responses remain at the user's supplied path and are copied
verbatim to `baselines/v3_briefs.private.jsonl` in the new pilot folder. No v4
briefs or model outputs have been fabricated.

After return, assess content regrouping, redundancy, source-template carryover,
reference resolution and material meaning problems. The CPU helper can sort
notes and remap directed point links while preserving their bindings. This
normalization does not certify semantic fidelity or remove latent source style.

Then prepare the previously planned matched reconstruction conditions from the
reviewed v4 briefs: identical topic, notes and anchors, with versus without
explicit purpose/relations. Use the same [base prompt](BRIEF_RECONSTRUCTION_V4.md).
The common notes/anchors must contain all substantive content. Keep originals out
of both generator requests. This small diagnostic precedes a frozen protocol for
fresh brief AND prose recollection for all 500 targets, including the existing
495. Reuse these five if the final protocol is unchanged.

Verify the immutable packet and baseline bindings with:
`python3 scripts/data/prepare_content_abstraction_pilot_v4.py check`.
No API calls, training, detector scans, development or reserved-test examples are
used by this preparation. CPU validation checks field/reference integrity, not
whether a model has actually abstracted style successfully.

## Stage 2 prepared from unchanged returned briefs

The five v4 briefs returned with valid identities and references. Content-bearing
annotation text decreased from 1,412 words in v3 to 579 in v4 across the five cases.
This is a compactness observation, not a realization result. The user explicitly
requested proceeding without the suggested minor output adjustments. No brief
wording, purpose label, relationship, anchor, note order or edge order was changed.

Two five-record packets are ready in
`research_runs/data_scaling_v1/content_abstraction_pilot_v4/stage_2_reconstruction_v1/`:

- `batch_A_content_only.md` requests `rewrites_A_01.jsonl`.
- `batch_B_guided.md` requests `rewrites_B_01.jsonl`.

Use GPT-6 with requested High effort for both. The two conditions share the exact
topic, notes and anchors and receive the identical base prompt; B includes the
original purpose and relationships as well. Save both files in the pilot's
`responses/stage_2_rewrites/` folder. Do not attach human targets, Stage 1 files,
prior outputs or the other condition. Cross-chat isolation remains unverified.

The compiler deliberately does not invoke alphabetical normalization or repair
helpers; it projects the supplied records unchanged. Hash bindings preserve the
raw response file. Verify with
`python3 scripts/data/prepare_abstraction_reconstruction_v4.py check`.
These are prepared jobs for ten outputs, not executed inference or reported results.

Both files subsequently returned; see the [observed comparison](CONTENT_ABSTRACTION_RECONSTRUCTION_V4_RESULTS.md).
All ten outputs remain unchanged. Guidance has selective organization/precision
benefits, but B is longer in every case and transfers a new annotation caveat in
P484. Content-only is the provisional neutral-reconstruction recommendation,
while enhanced briefs remain available for the separate brief-to-human route.
