# Original versus repaired source comparison v1

Prepared, not executed: sixteen first-attempt outputs across four previously
inspected training papers. This CPU-only package tests whether restored economics
helps rewriting, before introducing richer style controls or another training run.

| Input | Ordinary, shared evidence | Planned, same shared evidence |
|---|---:|---:|
| Original | 4 | 4 |
| Repaired | 4 | 4 |

The original-input arm is freshly rebound under the same annotation policy as
[the repaired-input checkpoint](REPAIRED_SOURCE_BINDINGS_V1.md). Legacy relation
errors are corrected in both arms. Claim IDs, epistemic categories, protection
categories and oracle grouping plans are matched; evidence always resolves to
the supplied input's actual wording. Original inputs do not receive facts or
technical definitions available only in their repairs. Matching claim IDs denote
reviewed correspondence, not identical propositions across source revisions.

This design tests repaired text together with its grounded evidence. It does
not isolate bare-source wording from the evidence inventory. No new expressive
style operation is introduced. The ordinary/planned difference is the existing
sentence-packing instruction only. Plan grouping is fixed between source arms;
source evidence is rebuilt instead of reusing stale offsets or erroneous edges.

Four balanced batches each contain one example per paper and one of each
condition. No batch contains two versions of a paragraph. Deterministic hash
ordering and a Latin rotation distribute conditions across batch context, but do
not make model calls independent or conceal the visible plan format.

## Manual generation

The private package is:
`research_runs/source_repair_comparison_v1/package_v1/`.
Read `START_HERE.md`. For each batch 01–04, attach only its
`batches/jobs_XX.jsonl` and `response.schema.json`, then paste `TASK_PROMPT.md`.
Use GPT-6 with High thinking effort consistently if available, preserving all
first attempts. Prefer separate conversations and record the actual arrangement.
Cross-chat memory remains possible; no memory clearing or independence claim is
required. Temperature, seed and hidden ChatGPT settings are uncontrolled.

Save `responses_01.jsonl` through `responses_04.jsonl` unchanged under
`research_runs/source_repair_comparison_v1/responses/`. Model/visible settings and
chat arrangement may be recorded in `responses/notes.txt`; credentials must never
be included. No GPU job, training run or external model call is part of package
preparation. Do not upload private keys, reference paragraphs, reports or PDFs.

## Interpretation specified before responses

Assess input fidelity, reference-claim recovery, copying and organization/style
separately. Exact evidence should support any added/lost claim finding. Quantify
copying relative to both input versions and distinguish changes already present
in the repaired input from those made by the generator. Sentence count is not
proof of operation adherence, and semantic lexical overlap is only a proxy.

Compare repaired versus original within each rewriting mode, then planned versus
ordinary within each input version. Report case-level differences; historical
outputs are not concurrent controls. Modest drift can be weighed against a real
style benefit, with the changed claim and benefit disclosed separately.

FJ1983 and F1980 are bounded source-fidelity diagnostics. NL2003 retains its
unspecified comparison/context flag; ME1997 retains recognition-versus-correction
antecedent wording. Report the latter two separately. Reference resemblance must
not override fidelity to the supplied input or reward an unresolved meaning
substitution. These are four inspected training cases with reference-informed
evidence and oracle plans, not paper-held-out generalization, a source-only
mapper, or evidence of distinctive Fama/coauthor style.

The private `EVALUATION_PLAN.md` records the assessment axes and limitations.
No output assessments or empirical success are claimed before responses exist.
Original inputs, references, repairs, old outputs and frozen training files remain
unchanged; public files contain code and aggregate preparation metadata only.

## CPU reproduction

With private parent artifacts present, choose a fresh output directory:

```sh
PYTHONPATH=src python3 -m fama_style.research.source_repair_comparison \
  --output-dir research_runs/source_repair_comparison_v1/package_v2
PYTHONPATH=src python3 -m fama_style.research.source_repair_comparison \
  --check-package research_runs/source_repair_comparison_v1/package_v2
python3 -m unittest discover -s tests -v
python3 train.py --check
```

Tests cover balanced coverage, deterministic prompts, isolation of the plan
instruction, original-wording retention, reference exclusion, endpoint ownership,
parent hashes and package tampering. Private integration tests skip when the
local paper artifacts are unavailable.
