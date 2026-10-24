# ChatGPT discovery package: how to use

This package selects **28 training pairs, four from each of seven papers**.
No development-validation or reserved-test paragraphs are included. Original
dataset rows remain unchanged. Generated annotations and repairs will be proposals.

## Recommended upload size

Use the four numbered chunks, **7 items per conversation**, to keep detailed
annotations manageable. The unnumbered files contain all 28 items as an alternative;
do not process both the full file and its chunks as separate observations.

Start a fresh conversation for each chunk. Keep the original-annotation and
neutral-design tasks in separate conversations. Upload the listed files and paste
the relevant prompt. Do not upload this guide, `PRIVATE_MAPPING_DO_NOT_UPLOAD.json`,
the manifest, or the selection report to an annotation conversation. Those reveal
batch roles or selection information. Textual author clues inside paragraphs are
not masked; this is ID/group masking, not guaranteed author blinding.

## Task 1: actual original paragraphs

For each `paragraphs_A_01.jsonl` through `paragraphs_A_04.jsonl`:

- Paste **ANNOTATION_PROMPT.md**.
- Upload that ONE paragraph chunk, **annotation_output.schema.json**, and
  **synthetic_example.json**.
- Save `annotations_A_01.jsonl` etc., plus any coverage/limitation report.

This task describes the actual original text. The annotator is not told to make
it conform to an expected author-style template.

## Task 2: improve the neutral reference and propose its taxonomy

For each `neutral_reference_pairs_01.jsonl` through
`neutral_reference_pairs_04.jsonl`:

- Paste **NEUTRAL_REVIEW_PROMPT.md**.
- Upload that ONE paired chunk, **ANNOTATION_PROMPT.md** as the specification,
  **annotation_output.schema.json**, **neutral_review_output.schema.json**, and
  **synthetic_example.json**.
- Save `neutral_proposals_01.jsonl` etc., plus its coverage/limitation report.

The existing plain English is a reference, not a fixed neutral ideal. ChatGPT may
propose revised text and the corresponding three-level taxonomy when needed.
The original supplies the semantic information; an uncertain original is flagged,
not silently repaired. A redesigned taxonomy must describe the proposed neutral
text, not be mislabeled as an observation of the unchanged reference.

These neutral annotations use paired context. That is recorded explicitly. They
are useful discovery proposals, but do not qualify as blind source annotations.
Before freezing the taxonomy or training a mapper, reannotate the accepted neutral
texts alone under the same specification, in separate conversations.

## Optional diagnostic: the unchanged plain-English reference

`paragraphs_B_01.jsonl` through `paragraphs_B_04.jsonl` contain the existing neutral
references with unrelated anonymous IDs and independently shuffled order. To
describe those texts as they currently exist, use ANNOTATION_PROMPT.md with the
same schema/example. It must not silently rewrite them. This optional task can
show how the proposed neutral design differs from the existing rewrite; it is
not required before the first discovery round.

## Consolidation comes after the responses

Bring back the eight main JSONL outputs (four original annotations and four
neutral-proposal files). They can be checked for completeness, schema validity,
evidence anchors and claim references before consolidation. Partial outputs should
be resumed by missing IDs, not filled with invented records.

Then use **CONSOLIDATION_PROMPT.md** in a new conversation with those checked
outputs. It asks for a candidate shared vocabulary with definitions, merge/split
decisions, full occurrence mappings and semantic protections. It does not freeze
the vocabulary or replace the dataset. Fresh training examples should be used
to check and refine it before annotating held-out development data.

The example labels illustrate the output format only. The discovery prompt leaves
the label inventory open; it does not reuse the previous rule-based taxonomy.
