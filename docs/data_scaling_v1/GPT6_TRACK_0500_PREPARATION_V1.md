# GPT-6 matched 500: reuse and version-lineage audit

Update: all eight response files have returned. The [response quality audit](GPT6_TRACK_0500_QUALITY_V1.md)
retains all 227 new inputs, records ten local repairs and five inherited target
follow-ups, and stages all 500 candidates. The preparation instructions below
remain historical; do not repeat completed batches.

The manual track retains the exact 500 human targets and their order from the
frozen mixed release. **273 accepted GPT-6 inputs can be reused; 227 need GPT-6
counterparts.** Preparation makes no provider calls and starts no training.
There is no new training-ready GPT-6 release until response checks are complete.

| Author group | Reused | Missing | Total |
|---|---:|---:|---:|
| Fama/coauthors | 122 | 3 | 125 |
| Shiller | 58 | 67 | 125 |
| Holmström | 42 | 83 | 125 |
| Baker–Wurgler/coauthors | 51 | 74 | 125 |
| Total | 273 | 227 | 500 |

Reuse comprises 158 inputs with recorded GPT-6 provenance, 111 earlier Fama inputs
whose historical model the user now confirms was GPT-6, and four reviewed
Baker–Wurgler inputs linked to the earlier 20-batch package. Of the 227 missing
inputs, 214 currently come from Gemini and 13 were wholly supplied by Codex after
the original annotation returned no neutral text. The latter are not relabeled
GPT-6. Small recorded repairs to existing GPT-6 prose retain repair authorship.

The user recalls the earliest workflow as raw PDF upload followed by GPT-6
extraction/OCR and rewriting. This confirmation is stored separately, without
changing original records or inventing historical thinking settings. It establishes
reported model identity, not independent OCR or semantic verification. Current
batches request GPT-6 with High thinking effort; actual settings remain unconfirmed
until the user returns the responses.

## Overlap findings

The audit binds original file hashes and compares normalized human targets,
earlier IDs, and raw-to-reviewed parent relationships across training pools:

- All 46 reviewed teacher-pilot targets are included in the earlier 58 training
  candidates. They are versions of existing pairs.
- The later 171 salvage rows descend from 170 of 185 raw training parents from the
  54-batch collection. One parent produced two children; only one enters the
  matched 500. There are 117 unchanged normalized targets; remaining target edits
  retain their parent links and do not create independent examples.
- Between the early 247 Fama training pairs and the later 185 raw training records,
  there are zero exact normalized target matches and zero flags from a conservative
  fuzzy candidate screen. The screen requires token-length ratio ≥0.55, shared
  vocabulary fraction ≥0.65 of the smaller set, and token sequence similarity ≥0.76.
  This is not proof that every semantic near-duplicate is absent.
- Each of the 500 selected targets is represented once. No reserved test text is
  accessed. Development metadata is used only for the PDF leakage check.
- A further check groups all 500 targets by canonical PDF, including aliased paper
  IDs, and screens all 3,305 within-paper pairs with the same fuzzy rule. None is
  flagged. This check is reproduced by the private-packet CPU tests.

**Paper-count correction:** the frozen mixed release has 53 stored paper IDs but
**48 distinct PDF sources**, because five Fama papers have both short and long IDs.
Canonical counts are Fama 11, Shiller 12, Holmström 12, and Baker–Wurgler 13. No
training PDF overlaps the development PDFs checked. Existing hash-bound releases
and checkpoints remain unchanged; use canonical PDF identities for coverage claims.

## Manual packet

Private folder: `research_runs/data_scaling_v1/gpt6_track_0500_v1/`.
Its `START_HERE.md` explains eight single-attachment jobs: seven batches of 30 and
one of 17. Each `manual_batches_v1/batch_NN.md` contains the complete prompt and
inputs. Save each `responses_NN.jsonl` in the packet's shared `responses/` folder.
No PDFs, additional schema attachments, taxonomies, or manual CSV entry are needed.
The first job includes all 13 gaps needed for the nested GPT-6 200-pair cohort.

The five-field response format is `id`, `target_sha256`, `plain_english`,
`rewrite_status`, `review_notes`. All 227 selected targets are intended for
completion; uncertainty is retained for automated repair or regeneration, rather
than selecting only a small subset after manual annotation. The generation task
is human paragraph → neutral input. Training reverses that direction.

Private artifacts include per-target reuse decisions, raw/reviewed lineage,
canonical PDF aliases, user model confirmation, source hashes, and a packet
manifest. Historical and new prompts/workflows differ, so matching human targets
does not make this a fully prompt-controlled teacher comparison. Prior acceptance
remains automated, not independent human gold labeling.

Reproduce with `python3 scripts/data/prepare_gpt6_track_500.py prepare`; the builder
refuses to overwrite an existing packet. Check with the same command using `check`.
CPU tests cover teacher identity, reviewed-version lineage, source bindings, target
coverage, deterministic rebuilding, and tamper/overwrite rejection. Source prose
and response packets remain private and are not distributed with the public code.
