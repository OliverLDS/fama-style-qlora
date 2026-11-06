# Local pair review

Run from the repository root:

```sh
node scripts/review_pairs.js
```

Open `http://127.0.0.1:8765`. Node 18 or later is sufficient; there are no package
installs, model downloads, provider calls, or credentials. The server listens only
on this Mac. Stop it with Ctrl+C in the terminal that runs it. Use `--port 8766`
if needed. This viewer does not submit notebooks or change datasets.

Filter by training/development/test, author group, individual coauthor, paper,
teacher, text, or example ID. A human target appears once, with matching teacher
inputs beside it. The neutral teacher text is the fine-tuning input; the original
human paragraph is the target. Expand the response/repair details to compare a
raw teacher response against its minimally repaired version. Search for P250,
P264, P291, P458, or P484 to inspect the five inherited target issues.

The interface is read-only and has no feedback export or form. Give feedback in
the conversation using the displayed P/G ID or `pair-…` identifier. The latter
depends on split, paper identity, teacher and exact input/target hashes. Identical
copies from multiple releases merge their provenance; edited text remains a
separate variant. Comments do not automatically edit or approve any record.

Current coverage:

The GPT-6 brief-reconstruction candidate is now included alongside the earlier
version. Fresh inputs have the badge `Candidate · fresh brief reconstruction`;
retained identical inputs merge provenance. The full reconstruction expansion adds
320 more fresh alternatives. The subsequent corrected-target version recovers
two earlier cases and implements six target-text changes, including one paragraph
replacement. The purpose-only round then adds 500 newly observed paragraphs.
The catalog now has 2,000 input variants and 506 target-text versions
across the same 500 cohort slots: old target versions remain visible. These counts
must not be interpreted as 2,000 independent examples or a 506-pair release.
The latest entry is `GPT-6 purpose-only 500 · observed candidates`.
It includes all 500 returned paragraphs unchanged, with P139/P400 labeled for
targeted upstream regeneration. Other scope advisories are in the output audit.
Gemini's original staging entry remains historical until source alignment is
reassessed against the shared repaired-target registry.

| Split | Human targets | Teacher inputs | Availability |
|---|---:|---:|---|
| Training | 506 text versions / 500 cohort slots | 2,000 | Historical and current GPT-6/Gemini variants; latest GPT-6 version has all 500 observed purpose-only outputs, with two targeted corrections pending. Gemini staging still includes 46 pending reviews and requires repaired-target alignment. |
| Development / validation | 80 | 88 | Existing GPT-6 inputs; eight matching Gemini counterparts. Includes 21 historical Fama validation examples. |
| Test | 45, from historical split metadata | GPT-6 legacy only | Reserved Fama test; no Gemini counterpart collected. |

Test bodies are not read on startup, in catalog metadata, by the training view, or
by `--validate`. Selecting Test explicitly loads them for your requested review.
Feedback arising from that exposure should not be used for development/model
selection. Historical validation/test provenance and prior exposure limits remain
as recorded; this tool does not make them a new blind benchmark.

## Adding future data

The local editable catalog is
`research_runs/pair_review_v1/catalog.private.json`. It is ignored by Git, as are
all data and response files. The public JavaScript contains paths and adapters,
not private paragraphs. The catalog is read again when you reload local data.

For an ordinary JSONL release, append a descriptor such as:

```json
{
  "id": "gpt6-training-1000-v1",
  "label": "GPT-6 1000 training v1",
  "adapter": "pairs",
  "split": "training",
  "teacher": "GPT-6",
  "model": "GPT-6 (reported)",
  "path": "data/future_release/training.private.jsonl",
  "status": "Use the actual release/qualification status"
}
```

Supported fields are `id`/`example_id`/`paragraph_id`, `source`/`plain_english`,
`target`/`original_paragraph`/`reference`/`completion`, `authors`, `paper_id`,
`style_group`, and optional exact `source_sha256`/`target_sha256`. Alternate text
columns can be declared with `sourceFields` and `targetFields` arrays. Use `where`
for exact filters on combined JSONL files. Declare each split separately; do not
label development records as test. Add paper metadata files to `paperRegistries`
using a `{ "papers": [...] }` object when expanding author/paper coverage.

New teacher/model names require only catalog changes. Specialized adapters also
support the current GPT-6 quality candidates and Gemini draft/qualification
checkpoint. When a new checkpoint or final release is created, update its catalog
path/status, or add a separately named version. The viewer never assumes a newer
file is more qualified. Additional schemas may require a small adapter; there is
no guarantee that every arbitrary future file format works without adaptation.

To initialize a fresh catalog if it is absent:

```sh
node scripts/review_pairs.js --init-catalog
```

This refuses to overwrite an existing catalog. Use `--catalog path/to/catalog.json`
for another project-local catalog. All declared files must stay inside the
project; external symlinks and arbitrary file-serving URLs are rejected.

## Verification

```sh
node scripts/review_pairs.js --validate
node --test tests/review_pairs.test.js
```

Validation loads only training/development. Tests use synthetic test content,
verify future generic releases, text-hash bindings, copy deduplication, lazy test
loading, read-only/local-origin HTTP access, and current staged training coverage.
No automatic test reads the real reserved test paragraphs.
