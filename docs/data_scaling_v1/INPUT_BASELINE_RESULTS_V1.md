# Exact input controls: all eight already receive human-written labels

All eight new Pangram PDFs bind uniquely to the frozen inference inputs: six
after whitespace normalization and two after PDF reading-order and punctuation
spacing normalization. Together with the existing outputs, the panel now has
40 reports for eight cases. All show Pangram 4.1 and 4 November 2026. Multiple
accounts were used; configuration equivalence is not independently established.

| Text condition | Entirely human-written labels | Mean displayed human-written share |
| --- | ---: | ---: |
| Actual GPT-6-derived input | **8/8** | **100%** |
| Strict prompt, base output | 0/8 | 0.08% |
| Strict prompt, adapter output | 6/8 | 75.89% |
| Simpler prompt, base output | 0/8 | 2.20% |
| Simpler prompt, adapter output | 0/8 | 8.70% |

The strict adapter preserves six human-written classifications and loses two;
it does not improve on the eight inputs. Both base conditions and the simple
adapter lose all eight. No detector-positive input exists in this panel, so
AI-related-to-human label conversion cannot be evaluated here. Shares are
provider-displayed authorship breakdowns, not document AI probabilities.

This supports the proposed explanation that input retention contributes to the
earlier strict-adapter detector advantage. It does not prove the mechanism or
establish distinctive author-style learning. It also does not mean GPT-6 pairs
are unusable, or Pangram is unusable for every experiment: these particular
inputs provide no detector headroom. Keep existing data, semantic assessments,
copying measurements and results intact. Future detector evaluation must include
the exact input baseline and distinguish input generators.

Private evidence and paired five-condition records:
`research_runs/data_scaling_v1/prompt_comparison_pangram_v1/input_output_import_v1/`.
The CPU importer is `scripts/data/import_corpus_200_input_reports.py`; it checks
packet, PDF, extraction and output-import hashes. Two focused tests ensure
human-label retention cannot be counted as conversion and incomplete pairing
fails. No new model, detector or GPU calls were made during import.

## Automating pair preparation on Kaggle

A larger local teacher can replace manual ChatGPT UI batching while preserving
the training direction: generated neutral prose as input, the unchanged human
paragraph as target. This is teacher inference, not teacher fine-tuning.

Google publishes [Gemma 4 12B instruction-tuned weights](https://huggingface.co/google/gemma-4-12B-it)
and documents [local use on 16 GB GPU devices](https://developers.googleblog.com/gemma-4-12b-the-developer-guide/).
Kaggle's [T4 x2 option](https://www.kaggle.com/discussions/product-announcements/735239)
has two separate 16 GB GPUs. My proposed starting configuration is the complete
12B instruction model with 4-bit weight quantization, short text contexts and
batch size one; actual memory use, numerical behavior and throughput need a
hardware preflight. Quantization reduces precision while retaining the model's
parameters. It does not mean using only an adapter. Unquantized 16-bit 12B weights
alone require about 24 GB before caches, so they cannot fit on one 16 GB T4.
Using two GPUs requires explicit model sharding; memory is not automatically pooled.

The Hugging Face token authenticates weight downloads; it is not a paid inference
endpoint subscription. The notebook would load weights on Kaggle and obtain any
needed token from Secrets without printing it. Access to the particular repository
and runtime support remain preflight checks, not verified results of this review.

Proposed execution order, before changing the student or expanding training:

1. Prepare a resumable, inference-only notebook with a frozen model revision,
   explicit decoding and final-answer parsing. Use the existing human targets;
   no new manual annotation upload cycle is needed.
2. Generate a small fixed panel across the four author groups using ordinary
   neutral rewrites and at most a few training-only demonstrations. Never use
   held-out targets as demonstrations. Keep development and training roles
   unchanged. Do not request deliberately detectable or detector-targeted prose.
3. Check alignment, copying, completeness and throughput. Preserve every output
   and explicit repair/rejection reasons rather than silently discarding rows.
   Scan a preselected subset of generated inputs; stronger Gemma may also be
   labeled human, so detector suitability must be measured rather than assumed.
4. If usable, generate the training corpus in resumable batches with automatic
   JSONL export, bounded retries, provenance and claim checks. Keep the original
   human paragraph as the supervised target, unchanged by the generator.
5. Compare the current student on separately frozen Gemma-input development
   cases with matched input/output detector scans, retaining GPT-6 cases as a
   separate condition. Only then choose whether a new student-training stage
   is warranted.

This is a proposed generation workflow. No Gemma 4 weights were downloaded, no
new Kaggle job was submitted, and no generation or training outcome is claimed.
