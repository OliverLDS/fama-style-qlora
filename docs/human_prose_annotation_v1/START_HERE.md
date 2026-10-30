# Start the 100-paragraph manual annotation pilot

The package selects **100 original paragraphs: five from each of 20 non-Fama
papers published in 1937–2019**. The conservative cutoff is publication before
1 January 2020. This concerns the original paper, not the date of the downloaded
PDF, scan, repository upload, or search-engine index. Bibliographic evidence,
source PDF hashes and extraction locations are recorded privately. The cutoff
does not prove absence from a language model's pretraining data.

This is a pilot of meaning-preserving neutral-to-human academic prose rewriting.
The selection covers economics/finance, not all human English. No detector scores
were used in sampling or requested in annotations. These are not yet accepted
training examples. The package records limited PDF-grounded extraction repairs while preserving all
upstream data. Full transcription is not independently verified. Claim alignment
and taxonomy annotations need the next
CPU audit after your responses arrive.

## What to send to GPT-6

There are 20 batches of five paragraphs. For each batch:

1. Open `batches/batch_01/TASK.md` (then batch_02, etc.) and paste its full text.
2. Attach these THREE files from this same package:
   - `batches/batch_01/paragraphs.jsonl` (change only this path for the next batch);
   - `common/annotation_output.schema.json`;
   - `common/taxonomy_catalog.compact.json`.
3. Use your usual GPT-6 / High thinking effort setting. Confirm that all three
   attachments are readable. Do not attach earlier responses, the source PDFs,
   the private selection key, source registry, or this guide.
4. Download the requested `responses_01.jsonl` and `coverage_01.md`. The batch's
   TASK.md gives exact filenames; use the matching number through 20. Preserve
   first responses. If GPT-6 cannot finish, preserve completed records and a
   missing-ID report, rather than silently regenerating the whole batch.

A fresh conversation per batch is preferable to reduce direct carryover. We do
not assume that separate chats prove independent contexts or eliminate memory.
Record whether chats were reused; do not spend time trying to guarantee isolation.
No human rating or manual semantic review is required at this stage.

## Where to save responses

Paths are relative to `research_runs/human_prose_annotation_v1/`:

| Batches | Frozen role | Save JSONL and coverage reports in |
|---|---|---|
| 01–12 | Training/discovery: 60 paragraphs from 12 papers | `responses/training/` |
| 13–16 | Development: 20 paragraphs from four papers | `responses/development/` |
| 17–20 | Reserved evaluation: 20 paragraphs from four newly added papers | `responses/reserved_test/` |

`BATCH_INDEX.json` records every input and response path. All 20 batches can be
annotated with this frozen specification. Keep the reserved responses sealed from
prompt/schema changes, candidate selection, and development analysis. If we revise
the annotation specification after inspecting early responses, stop before doing
remaining reserved batches so the protocol version can be recorded. Unchanged
first responses always stay preserved.

The split keeps every selected author's papers together. Fourteen source papers
were already exposed in earlier comparison-author research. The reserved papers
are newly added for this package; this is not a guarantee of absence from GPT-6
or Gemma pretraining. The existing Fama 21-example development set and 45-example
reserved set are not read, repurposed or merged into this pilot.

Also save `responses/SESSION_NOTES.md` with:

- model label and thinking-effort setting shown in the UI;
- dates and batch IDs;
- one chat per batch or which batches shared a chat;
- any regeneration, missing attachments, truncation, or continuation.

Do not include credentials or private account information. A hidden snapshot,
sampling temperature or system prompt need not be guessed.

## What the output contains

Each pair retains the original exactly and adds a neutral rewrite, semantic
ledger, source/target schemas under the SAME existing provisional taxonomy,
claim-bound structural differences, provenance and uncertainty flags. Original
transcription errors must be flagged, never silently corrected. Already-clear
originals may remain unchanged. The compact catalogue is a lossless shared-rule
view of all 132 original categories, not a new taxonomy or evidence that the
vocabulary generalizes to every author.

The semantic ledger and both schemas are jointly produced in the same context;
they are not independently verified. Later source-only generation must exclude
original reference wording, target quotes and target-only claim descriptions.
A target structural plan can be used only in a declared oracle condition.

After the responses arrive, the next work is import/coverage/evidence checking,
automated pair audit and a small development-only inference comparison. This
package starts no model calls, training or GPU runs. Private original text and
responses must stay outside Git publication.
