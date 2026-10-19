# Data availability

The public checkout contains the frozen rewrite instruction and `historical/train.example.jsonl`: three **synthetic** educational examples. Paper PDFs, extracted paragraphs, full text pairs, taxonomy annotations, detailed repairs and generated outputs remain local and are not distributed here.

## Recorded corpus

Thirteen papers were processed; the selected corpus contains 313 pairs from nine papers. Raw extraction/annotation used an LLM and is not independently certified transcription or verified author intent.

| Split | Rows | Paper boundary |
|---|---:|---|
| Train | 247 | Seven papers, including Fama–Jensen 1983 |
| Validation | 21 | Fama–French 1997, Taxes, Financing Decisions, and Firm Value |
| Test | 45 | Fama–French 2014 five-factor draft |

The local raw collection has 1,856 records. Selection uses narrative type, review flags, length, meaning alignment and provenance. There are 117 recorded input-meaning repairs and 39 target whitespace normalizations. Targets were not replaced with newly generated prose. Uncertain scanned/mathematical text was held; Fama 1965 extraction is partial.

Rows contain `id`, `paper_key`, `split`, `plain_english`, `original_paragraph`, `prompt`, `completion`, `prompt_format` and bibliographic/page provenance. `completion` equals the original paragraph; `prompt` uses `fama_corpus_v1/rewrite_instruction.txt`. Taxonomies are metadata, not targets.

Synthetic direction illustration, not paper text:

```json
{"plain_english":"If demand falls, the firm may reduce output.","original_paragraph":"A decline in demand may induce the firm to reduce output."}
```

Whole-paper splits do not establish absence from model pretraining. Validation was used during development. The 45 test rows were length-audited by the baseline but not generated or scored in current corpus experiments. Assistant semantic review is not independent human verification.

See [reproduction instructions](../docs/REPRODUCIBILITY.md) for exact local artifact paths. Hash guards reject substitutions. The generic original CLI accepts your own prompt/completion JSONL via `train.py --check --data PATH`; the frozen 4B notebooks require a new reviewed experiment definition for new data.
