# Brief review complete: Stage 2 ready

All six Stage 1 files were imported: **175 distinct expected IDs**, correct
target hashes, valid brief fields and complete batch coverage. No brief triggered
the long ordinary-prose-copy screen. P002's notes remove the original hopes framing;
P024 is represented by a compact procedure rather than the old paragraph template.
This remains automated assessment, not independent semantic certification.

Seven local edits across six records clarify content/status, without changing raw
responses or human targets. They add the reason for reconstructing dividend income
(P024), avoid an unsupported causal connector (P118), distinguish usable sentiment
measures from investable assets (P136), avoid assuming a price interpretation of
an ambiguous discount (P139), retain the investor-response mechanism behind signal
validation (P176), and clear an unnecessary absent-equation warning (P282).
These edits do not restore the original's rhetorical phrasing.

The effective briefs have **173 ready** and **two source concerns carried**:

- P134: the supplied historical account's chronology needs source checking.
- P139: the model's liquidity-premium/discount comparison is unclear in the supplied text.

Neither item is dropped, guessed, or allowed to block the other reconstructions.
P282's variable definitions are usable without an equation; the writer is told
not to invent a formula. Minor compression and emphasis shifts elsewhere are
accepted without exhaustive restoration of citations, qualifiers or prose.

## User action

The six self-contained attachments are ready under:

`research_runs/data_scaling_v1/gpt6_track_0500_v1/contrast_rewrite_v1/stage_2_reconstruction_batches/`

Upload only `batch_01.md` through `batch_06.md`, one per GPT-6 request, and request
High thinking effort. Prefer a new chat without attaching original prose or old
rewrites; no isolation from cross-chat memory is claimed. Each attachment contains
only instructions, effective briefs, IDs, hash strings and source-concern notes.
The batch sizes are 30/30/30/30/30/25. All 175 selected items are included.

Save `rewrites_01.jsonl` through `rewrites_06.jsonl` in:

`research_runs/data_scaling_v1/gpt6_track_0500_v1/contrast_rewrite_v1/responses/stage_2_rewrites/`

No additional attachments, taxonomy, CSV entry or recollection of Stage 1 is needed.
The goal is fresh neutral composition, with contrast and material errors assessed
separately afterward. No model training or detector scans are started by this packet.

## Audit

`brief_quality_v1/` preserves the raw-file hashes, effective briefs, exact local
edits, reasons and advisory flags. Its manifest binds the complete Stage 2 packet.
The original preparation and all previous files remain unchanged. Reproduce checks
with `python3 scripts/data/review_gpt6_contrast_briefs.py check`.
The exporter refuses overwrite, replays text-bound edits and forwards an explicit
five-field payload; CPU tests verify coverage and absence of the complete original
paragraph and old rewrite from each corresponding attachment. Actual per-job UI
model/effort settings remain unverified by the package itself.
