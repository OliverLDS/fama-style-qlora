# Stage 1 review and Stage 2 preparation

All 12 responses arrived in the two expected batches. The user reports GPT-6
with High thinking effort. Original response bytes, source hashes and the frozen
selection remain unchanged. Conversation arrangement was not restated; cross-chat
retrieval and independence are not established.

The final review is `research_runs/neutral_reconstruction_v1/stage1_review_v2/`.
The current generation packet is **`research_runs/neutral_reconstruction_v1/stage2_package_v2/`**.
Use its `START_HERE.md`, not an earlier Stage 1 or diagnostic packet.

## Findings

All 12 records pass ID, source-hash, schema and relationship-reference checks.
As received, nine sources were labeled usable and three uncertain. Eleven briefs
had review flags; only one passed the initial mechanical content gate unchanged.
All 12 self-checks reported success, which did not establish semantic fidelity.

Automated source/brief assessment supports **eight provisional paired generation
cases**, across all six papers: three Holmström, three Baker–Wurgler and two
Shiller. Four cases stay held in the original 12-case denominator. There are no
replacement examples. These are provisional alignment judgments, not independent
or blinded semantic gold labels, training approval or generated-writing results.

Four held cases:

- An ambiguous reference to which historical episodes appear in an index.
- A comparative importance statement whose outcome is absent from the paragraph.
- An example that refers to an effect defined in omitted preceding material.
- A source item that the PDF shows to be two separately indented paragraphs
  merged during extraction.

The [post-annotation source-hold overlay](../author_curation_v1/post_annotation_holds_v1.json)
identifies the four source IDs for future selections. It preserves the original
curation snapshot rather than rewriting its labels or deleting text. Context or
boundary repairs would be a separate versioned data task. An earlier unexecuted
diagnostic briefly accepted the episode reference based on semicolon grouping;
that interpretation was withdrawn before generation. The earlier review and
packet remain archived and are not the current run instructions.

## Audited derived changes

Some missing-context flags ask for explanations, measurements or operational
definitions that the source itself does not provide and a faithful rewrite need
not add. Those flags were resolved with explicit reasons; dates, counts,
definitions and evidence were not invented. Genuine missing referents remain held.

The derived briefs correct qualifiers and unsupported inferences, including:

- a higher frequency **level** versus a greater numerical increase;
- a near-definitional statement versus strict entailment;
- a natural feature versus an inherent characteristic;
- missing scope of a common financing principle and incorrect actor binding;
- inferred comparators and relationships that are not asserted by the source.

There are 55 changed brief fields, including flag/context-list updates, semantic
corrections and removal of annotation-only missing-data commentary. Every change
records the exact before/after value and rationale in a private audit. The raw
ChatGPT briefs are never edited in place. The content gate is not weakened to
accept unresolved briefs; derived copies explicitly record the resolution or hold.

Rendered PDF paragraphs confirm four superscript footnote markers in three
approved sources. Only those markers are removed in derived references; economic
numbers and citations remain. Both methods and later fidelity comparisons use
the same derived reference for each pair. Raw Stage 1 references and the full
curation release are unchanged. The audit pins original/derived hashes, PDF
hashes and private evidence-image hashes. This is a narrow source check, not
independent verification of every PDF transcription or claim.

## Ready for generation

The [Stage 2 instructions](STAGE2_START_HERE.md) specify four ChatGPT jobs:
two direct-rewrite batches and two reconstruction batches, with five and three
records respectively. This requests **16 first outputs**. Retain any additional
generation-time blocks. Both conditions receive the same cases; the reconstruction
payload includes only reviewed semantic briefs, opaque/reordered claim IDs and
validated semantic edges. It excludes original paragraphs, paper/author metadata,
source-review commentary and target rhetorical plans. Essential terms and
citations remain, and uncontrolled UI retrieval is still a limitation.

No generations, detector scans or GPU jobs have been executed by the agent.
Keep actual generation model/effort provenance separate from Stage 1 provenance.
Evaluate both outputs against the reviewed reference, not merely against the
brief; report paired outcomes and the full 12-case coverage with group imbalance.
No style improvement or successful source-generation method is claimed yet.

The [aggregate review](stage1_review_results.json) contains no paragraph text.
Code is in `src/fama_style/research/neutral_reconstruction_review.py`; private
response receipts, before/after brief patches, source-marker repairs, holds and
derived references remain under `research_runs/neutral_reconstruction_v1/`.
The review results and Stage 2 packet are byte-reproducible from preserved raw
responses and the explicit decision file. This tests transport/reproducibility,
not the correctness or independence of qualitative alignment judgments.
