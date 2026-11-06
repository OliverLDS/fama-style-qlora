# Reference controls complete; reuse prior human calibration

The eight new Pangram PDFs all match the complete frozen reference paragraphs
after whitespace normalization. All eight are labeled **Human Written**, with
**100% human-written document share**. Reports are Pangram 4.1, dated 5 November
2026. Raw PDFs, extractions, hashes and the combined 32-report panel are archived
privately; the earlier 24 input/output reports remain unchanged.

| Text condition | Entirely human-labeled |
| --- | ---: |
| Published human references | 8/8 |
| Gemini inputs | 1/8 |
| Base Gemma outputs | 0/8 |
| Mixed-500 outputs | 2/8 |

The complete input/output conclusion remains one improvement, four deteriorations
and three ties against actual Gemini inputs. One of seven AI-related inputs
becomes entirely human-labeled; the other human-labeled output preserves a label
already present in its input. The reference controls do not change that finding
or establish distinctive Fama-style learning.

## Credit-use correction

The repo already contained [four published-human controls](../milestone_checkpoint_4b_v1/DETECTOR_RESULTS_V2.md)
correctly labeled human by both Pangram and GPTZero. Requesting eight more was a
low-value repeat of that corpus-level calibration. The prior lookup considered
exact text matches but did not sufficiently weigh this existing genre-level
evidence. The user is right to object to the unnecessary manual effort and
credit cost. The current results are retained, with no further routine
published-reference scans requested.

Before any future manual detector packet, review prior controls and reuse them
where they answer the question. New scans should have a distinct decision value
and a stated credit budget; lack of an exact prior paragraph match alone is not
sufficient justification. Favor matched generated input/output comparisons and
reuse the existing human calibration. Propose new human controls only for a
specific domain/version/anomaly question that could alter the next action.

## Detector-mechanism hypothesis

Training exposure, memorization or exact/near-exact matching could help explain
reliable recognition of published texts. This is a hypothesis, not established
by an 8/8 result. Learned genre features and conservative calibration can also
produce that observation. Our scans cannot identify the internal mechanism or
establish that human-text false positives never occur.

[Pangram's public description](https://www.pangram.com/research/how-it-works)
describes a neural classifier trained on human/AI examples, augmented with
misclassified human examples and mirrored AI documents.
[GPTZero's public description](https://gptzero.me/technology) also describes
trained neural classification and calibration favoring fewer human-text false
positives. These descriptions do not establish an automatic published-text
allowlist, and do not rule out training exposure or undisclosed mechanisms.
No additional detector experiment is proposed merely to distinguish them.

Human authorship established by provenance and a detector's predicted label
remain separate. A human-written label on known AI output measures detector
response, not actual authorship or independently verified writing quality.

Private receipts: `research_runs/data_scaling_v1/corpus_500_pilot_4b_v1/pangram_reference_reports_v1/`.
Import and four-condition summary: `pangram_reference_import_v1/` under the same
run. Public aggregate: `stage_0500_reference_controls_v1.json`. Importer:
`scripts/data/import_corpus_500_reference_pangram.py`. CPU tests ensure unexpected
AI-related reference labels would be retained and mismatched references rejected.
No new generation or training was launched.
