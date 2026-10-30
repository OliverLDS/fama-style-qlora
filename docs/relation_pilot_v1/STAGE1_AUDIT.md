# Stage 1 audit and Stage 2 handoff

All eight response files cover their assigned forty paragraphs exactly once.
User-reported settings are **GPT-6, High thinking effort**. Raw responses contain
null model/effort provenance; those values remain unchanged, with the user's
confirmation stored separately. Chat arrangement and retries were not reported
for these eight batches. Context independence is not established.

| Check or annotation | Result |
|---|---:|
| Closed-schema and mechanical checks passed | 40/40 |
| Exact original text matches | 40/40 |
| Exact quote/occurrence anchors checked across both texts | 725 |
| Self-labeled simplified / already clear / pending | 38 / 1 / 1 |
| Claims self-labeled preserved | 354 |
| Records flagged for review / external context | 24 / 34 |
| Original / neutral sentences | 177 / 193 |
| Neutral sentence count higher / same / lower | 14 / 26 / 0 |
| Unchanged neutral paragraphs, including pending | 2 |

Mechanical checks cover batch/ID coverage, original integrity, sentence
reconstruction and sequential IDs, claim bindings, back-reference separation,
required evidence, and exact quote occurrence. They **do not verify claim
completeness or semantic equivalence**. All 354 preserved labels come from the
paired annotator, not an independent verifier. Sentence bindings identify content
components; a claim spanning sentences must not be interpreted as multiple full
repetitions. Coding-assistant reading also does not provide independent verification.
One additional antecedent/scope interpretation is recorded privately as a
potential risk, without changing the annotator's labels or declaring a confirmed
error. It stays eligible for source-only judgments of the text as supplied;
reference-alignment analyses must retain this caution separately.

Token sequence overlap has median 0.714 and range 0.486–1.000. This measures
wording reuse, not author style or semantic fidelity. The limited sentence-count
variation is important: 26 paragraphs have the same count in both versions, so
this collection should not be described as forty nonidentity structural targets.
Equal counts do not by themselves establish identical claim organization.

The pending rewrite preserves the original provisionally because an antecedent
cannot safely be resolved from the excerpt. It is held back without altering
text or flags. The other **39 paragraphs** are eligible for source-only annotation,
not approved training data. Remaining review/context flags are retained offline;
a source-only judge cannot recover missing surrounding economic information.
Future reference comparisons must distinguish these limitations from grouping
errors introduced by a model. No annotation or prose was repaired here.

## Stage 2 files

Private entry point:
`research_runs/relation_pilot_v1/stage2_package_v1/START_HERE.md`.

Eight mixed-paper batches contain seven groups of five paragraphs and one group
of four, with **152 adjacent boundaries** under the Stage 1 segmentation. The
judge resegments the actual text and must cover its actual adjacent boundaries;
152 is an audit expectation, not a forced output target. Each payload exports
only `paragraph_id` and actual returned neutral `source_paragraph`. It exports no
original field, paired C-claims, reference sentence layout, source metadata,
sampling strata or uncertainty labels. The already-clear unchanged paragraph
remains a legitimate source input, so field isolation does not guarantee all
source wording differs from reference wording.

For each batch, paste its `TASK.md` and attach only its `paragraphs.jsonl` and
`common/stage2_output.schema.json`. Use the same model and effort. Save the first
complete unedited `responses_01.jsonl` through `responses_08.jsonl` and coverage
reports in the new packet's `responses/`. No Kaggle job is needed.

Stage 2 extracts fresh source-only D-claims and separates permissible
KEEP/MERGE/BOTH/UNRESOLVED judgments from presentation preference. Concrete merge
candidates remain unverified until audited. Pairwise judgments do not authorize
unlimited merge chains. Reference layouts remain offline and are not unique
correctness labels. Shared model, pretraining and possible automatic chat
retrieval still limit independence.

## Reproduction and privacy

`fama_style.research.relation_pilot_review` audits the frozen Stage 1 packet and
builds Stage 2 in fresh output directories. Its CLI accepts `--packet`,
`--audit-output`, `--session` and `--stage2-output`. Existing packets/results are
not overwritten. Raw responses and coverage reports are snapshotted byte for
byte; manifests pin inputs, code and outputs. CPU tests exercise corrupted
originals/evidence/bindings, missing/duplicate IDs, pending/changed-claim exclusion,
raw preservation, output immutability, hash tampering and reference-field exclusion.

The [aggregate summary](stage1_audit_summary.json) contains no source paragraphs.
PDFs, originals, neutral texts, evidence, responses and private mappings remain
Git-ignored. No reserved examples, GPU jobs, fine-tuning, classifier refitting or
new model calls were used by the audit or packet preparation.
