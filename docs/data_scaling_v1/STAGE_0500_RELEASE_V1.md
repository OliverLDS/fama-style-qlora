# Frozen 500-pair cohort

The 500-pair expansion is complete. It uses existing downloaded material and
retains the original 200 JSONL records as an exact byte prefix. No training was
submitted. Input is neutral generated prose; target is the human author paragraph.

| Author group | Pairs | Stored paper IDs | Distinct PDFs |
| --- | ---: | ---: | ---: |
| Fama/coauthors | 125 | 16 | 11 |
| Shiller/coauthors | 125 | 12 | 12 |
| Holmström/coauthors | 125 | 12 | 12 |
| Baker–Wurgler/coauthors | 125 | 13 | 13 |
| Total | 500 | 53 | 48 |

The [subsequent lineage audit](GPT6_TRACK_0500_PREPARATION_V1.md) found five Fama
paper-ID aliases. It corrects coverage reporting to 48 distinct PDFs without
altering the frozen cohort, historical manifests, or experiments.

The 300 additions consist of ten unused approved pairs, 75 requalified older
Fama pairs and 215 newly reviewed drafts. The last group retains 214 Gemini
inputs and one recovered historical GPT-6 input. The older Fama rows retain
their original model/provenance fields. The subsequent GPT-6 track stores the
user's historical model confirmation separately, leaving this release unchanged.
Existing responses were reused before requesting missing annotations.

## Curation and repair

Gemini 3 Flash preview generated missing neutral inputs through the installed
inferencer package and the user-confirmed free-tier project. Bounded reasoning
and JSON output avoided the initial response-budget failure. Ten complete
records from that interrupted response were recovered instead of regenerated.
Provider failures and all raw results remain archived. No paid fallback ran.

A first semantic reviewer frequently proposed the original human text as its
repair. Those complete-paragraph proposals were **not admitted**: they would
erase the neutral-to-author rewriting signal. A separate meaning-only reviewer
returned minimal, exact substring corrections. Codex adjudication checked the
flagged notation against PDF page renderings and screened numeral differences.
In particular, PDF checks overruled an erroneous strict-inequality suggestion.

The final 215 decisions include 15 inputs with patches and 27 PDF-checked child
target repairs. Target repairs separate section headings, restore fused-word
spacing and correct visibly damaged mathematical/font notation. Three
non-narrative records—an acknowledgment, a figure caption and a table-source
note—were replaced with prose from an existing training paper. Original
extractions, rejected drafts, reviewer proposals, response-ID repairs, before/
after hashes, applied patches and PDF evidence remain in the private run folder.

Numeral counters are diagnostics, not semantic judges: spelled-out numbers,
abbreviated years, equivalent quintile expressions, repeated index names and
unattached footnote anchors have explicit adjudication notes. No unexplained
numeric difference is accepted by the finalizer.

## Executed checks and limits

- Exactly 500 unique targets and parent IDs; 125 per group.
- Original 200 records preserved byte for byte, in their original order.
- No target pair flagged by the recorded token-sequence near-duplicate screen
  at 0.85 similarity, with its documented length/token-overlap prefilters.
- New reviewed targets have no development-paper or development-PDF-hash
  overlap. Source/PDF hashes checked; sealed test bodies were not read.
- No normalized exact input/target identities. Median word-sequence similarity
  is 0.6555; this is a lexical diagnostic, not evidence of learned author style.
- A separate CPU rebuild reproduced every release file byte for byte. Ten
  focused CPU tests passed, including tamper rejection and patch replay.

These are **automatically accepted pilot pairs**, not 500 independently verified
gold examples. The new model reviews use the same model family as generation;
Codex PDF checks are targeted rather than full visual verification of every
paragraph. The 75 older Fama rows reuse prior explicit semantic acceptance with
new split, hash, source-flag and numerical screens, not a fresh independent
semantic assessment. Structural annotations are not qualified for training by
this text-pair release.

Teacher composition and paper exposure change along with size. A later
200-versus-500 comparison will therefore be a practical expansion experiment,
not a clean causal estimate of sample count alone. Detector outcomes remain
secondary under the [teacher/evaluation policy](TEACHER_AND_EVALUATION_POLICY_V2.md).
This release contains no new fine-tuning result or detector measurement.

## Reproduce locally

Private dataset: `data/data_scaling_v1/stage_0500_expansion_v1/training.private.jsonl`.
Private audit: `research_runs/data_scaling_v1/stage_0500_expansion_v1/`.
The repository publishes code and aggregate results, not source paragraphs or
private datasets.

```sh
python3 scripts/data/expand_stage_500.py check
python3 scripts/data/audit_stage_500.py
python3 -m unittest discover -s tests -p test_expand_stage_500.py -v
python3 -m unittest discover -s tests -p test_stage_500_finalization.py -v
```

`finalize_stage_500.py` replays hash-bound decisions without provider calls; it
refuses to overwrite the frozen release. The archived decision manifest binds
all source jobs, raw responses, amendments and PDF evidence used in assembly.
`stage_0500_release_v1.json` publishes aggregate checks and the dataset hash.

Next: prepare a matched 500-pair pilot using the existing development examples
and agreed simple inference prompt. Keep the 200-pair adapter/results intact.
Training still requires explicit authorization; no new annotation uploads or
paper downloads are needed for that comparison.
