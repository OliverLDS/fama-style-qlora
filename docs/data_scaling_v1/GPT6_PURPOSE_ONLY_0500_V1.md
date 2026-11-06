# All 500 pairs: enhanced abstraction and purpose-only reconstruction

All seventeen reconstruction files have now returned. The [500-output audit](GPT6_PURPOSE_ONLY_OUTPUTS_V1.md)
preserves every observed pair, measures surface contrast, adjudicates model warnings
and prepares one two-record upstream recollection for P139/P400. The candidate
snapshot remains disabled for training; earlier artifacts are unchanged.

All seventeen brief files have now returned. The [audit and Stage 2 handoff](GPT6_PURPOSE_ONLY_BRIEFS_V1.md)
includes every one of the 500 records unchanged, documents sparse relation graphs
and content follow-ups, and prepares seventeen purpose-only reconstruction jobs.

The selected policy applies to **all 500 matched human targets**. The user retains
enhanced purpose-and-relation briefs, and reconstructs with A's content fields
**plus purpose**. This supersedes the provisional content-only recommendation.
Purpose-only differs from both executed pilot arms; its performance has not yet
been measured. The pilot findings are preserved without retrospective relabeling.

| Brief field | Retained in stored brief | Sent for reconstruction |
| --- | --- | --- |
| topic | Yes | Yes |
| purpose | Yes | Yes |
| key_points | Yes | Yes |
| anchors | Yes | Yes |
| relationships and their qualifications | Yes | No |

IDs, target hashes, status and ordinary uncertainty notes use the same wrapper as
the pilot. No relation annotations, original human paragraph or earlier prose are
forwarded. The entire enhanced brief remains available for brief-to-human training
research. No output-label changes, wording edits, note sorting or edge reordering
are performed by this preparation.

## Complete collection scope

The current repaired human registry has 500 training slots, 125 per author group.
The five valid v4 pilot briefs are reused unchanged. Seventeen extraction jobs
collect the other 495 under the same [abstraction prompt](BRIEF_ABSTRACTION_V4.md):
sixteen batches of 30 and one of 15. They are not a separate 495-pair release.
All 500 will receive new purpose-only reconstructions, including the five pilot
targets. None of the five A or five B pilot outputs is substituted for that round.

Private collection folder:

`research_runs/data_scaling_v1/gpt6_purpose_only_0500_v1/`

Upload `stage_1_brief_batches/batch_01.md` through `batch_17.md`, one job per
request, using GPT-6 with requested High effort. Each file includes the prompt,
relation catalog and source records; no other attachments are required. Actual UI
settings and cross-chat memory isolation remain unverified. Save completed data as
`briefs_01.jsonl` through `briefs_17.jsonl` in `responses/stage_1_briefs/`.

Every target retains its ID, paper, author group and repaired target hash. Source
paragraphs come from the frozen training registry only; no development/test text
is read. No author expansion or selection based on output quality is made here.
The original dataset versions and archived PDFs remain unchanged. Gemini's older
source qualification still requires separate repaired-target reassessment.

## Reconstruction after brief assessment

After the actual responses return, assess coverage, abstraction and material
content issues across the records. Preserve completed work and record an explicit
decision for each item rather than silently discarding a large fraction. Do not
invent missing briefs or auto-certify semantic equivalence from schema validity.

The builder then combines the five reused records with the 495 responses in the
original 500-slot order, retains all enhanced fields, and emits only the four
selected fields for reconstruction. The same [base realization prompt](BRIEF_RECONSTRUCTION_V4.md)
is used. There are 17 reconstruction jobs: sixteen of 30, then 20. Source concerns
are retained in status/notes and all resulting paragraphs require later assessment.

This packaging step does not itself conduct semantic assessment or make a
training-ready release. Formal validation and projection commands are:

```sh
python3 scripts/data/prepare_gpt6_purpose_only_500.py check
python3 scripts/data/prepare_gpt6_purpose_only_500.py check-briefs
python3 scripts/data/prepare_gpt6_purpose_only_500.py prepare-reconstruction
python3 scripts/data/prepare_gpt6_purpose_only_500.py check-reconstruction
```

The actual briefs are now available, and Stage 2 packets have been prepared;
use `check-briefs` and `check-reconstruction` to verify them. Do not rerun the
refuse-overwrite preparation action on the existing packet. No model results
are fabricated. All candidates remain disabled for training. No provider
calls, paid services, GPU jobs, detector scans or training are initiated here.
