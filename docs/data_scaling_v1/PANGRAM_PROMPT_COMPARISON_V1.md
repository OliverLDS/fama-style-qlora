# Pangram results: the prompt changes the detector benefit

The user supplied all 32 PDFs: eight matched development cases under two prompts
and two model conditions. Both the older strict-prompt packets and replacement
simpler-prompt packets were scanned. All reports show **Pangram 4.1**, dated
**4 November 2026**. No detector calls or GPU jobs were made during import.

Latest follow-up: the [completed input controls](INPUT_BASELINE_RESULTS_V1.md)
show all eight inputs already receive human-written labels. The earlier
output-only comparison below remains historical evidence, not an input-relative
conversion result.

Earlier follow-up: the [matched input/output protocol](MATCHED_INPUT_OUTPUT_DETECTOR_PROTOCOL_V2.md)
now prepares the exact eight missing input controls. Their baseline scores are
needed to distinguish detector improvement from retaining an already
human-labeled source. The 32 output scans will be reused.

## Training and inference direction

Pair preparation starts from a historical human paragraph and asks GPT-6 to
produce neutral prose. Training then runs in the opposite direction:

`GPT-6-generated source → original human-written target`

At inference, the model receives that generated source plus the rewrite
instruction, and produces a proposed human-style rewrite. The original human
paragraph is retained for offline comparison; it is not included in generation
messages. Versioned source repairs remain part of the approved dataset provenance.
This 200-pair pilot pools four academic author groups; its generic instruction
does not select a particular author's style at inference.

## Same eight detector cases

| Condition | Entirely human-written labels | Mean displayed human-written share | Median source overlap | Outputs with substantive meaning flags |
| --- | ---: | ---: | ---: | ---: |
| Strict prompt, base | 0/8 | 0.08% | 0.243 | 8/8 |
| Strict prompt, adapter | **6/8** | **75.89%** | **0.808** | 1/8 |
| Simpler prompt, base | 0/8 | 2.20% | 0.330 | 5/8 |
| Simpler prompt, adapter | **0/8** | **8.70%** | **0.412** | 3/8 |

The shares above are averages of Pangram's displayed authorship breakdowns,
**not probabilities that a document is AI-generated**. Mixed summaries are
retained in the machine-readable aggregate and private observations. Tiny
displayed shares sometimes produce totals over 100%; no rescaling is applied.
Meaning flags come from the previous provisional automated reviews and are not
used to filter cases. One strict-adapter output marked entirely human also has
a substantive meaning flag.

The paired input IDs are identical across all four conditions. These copying
and meaning figures describe the **eight scanned cases**, whereas the earlier
[inference review](SIMPLE_PROMPT_RESULTS_V1.md) describes all sixteen development
cases. Their denominators must not be mixed.

The trained checkpoint has a substantial detector benefit over base with the
strict prompt on this panel. That benefit largely disappears when the simpler
prompt induces more rewriting. A shift from AI-generated toward AI-assisted
shares remains under the simpler prompt, but none is labeled entirely human.

This pattern is consistent with the strict checkpoint benefiting from input
retention. It does not isolate that mechanism: these eight source texts were not
separately scanned as controls in this comparison. Earlier GPT-6 input controls
were different examples and cannot supply missing scores here. The result does
not establish distinctive author-style learning, nor does it establish failure
of the entire approach. It demonstrates prompt-dependent behavior of one
checkpoint on eight reused development cases.

## Import and provenance

Every PDF was matched using its full scanned text, not its filename alone:
15 matched after whitespace normalization and 17 after PDF visual reading-order
and punctuation-spacing normalization. Filename IDs also agree. All 32 bind
uniquely; no report or unfavorable outcome is omitted. Scores come from page 1;
all pages are retained for full-text binding. Raw reports, extraction JSON and
text hashes remain private under:

`research_runs/data_scaling_v1/prompt_comparison_pangram_v1/`

The user reports using multiple accounts. Account count, per-report account
assignment and independently verified settings equivalence are unavailable;
no account identities or credentials are requested or stored. Identical visible
provider versions do not by themselves verify identical account configuration.

The CPU importer is `scripts/data/import_corpus_200_pangram.py`. It verifies
frozen packet hashes, receipt hashes, complete coverage and output bindings,
and retains native summaries and shares. Three parser tests cover mixed labels,
display floors, unknown layouts and percentage ranges. An independent CPU
rebuild reproduces the import byte for byte. The raw source packets, model
results and previous reports remain unchanged. No new annotation or training
is initiated by these findings.
