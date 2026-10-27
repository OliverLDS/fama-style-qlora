# Elementary claim-bound packing edits, version 1

The existing annotations support a broader sentence-packing diagnostic than the three original rhetorical templates. A separate CPU compiler now derives oracle edit programs from the 24 rows labeled `already_structured`, without changing their annotations or counting them as successes for the frozen template protocol.

## Result

| Disposition | Paragraphs |
| --- | ---: |
| Compiled oracle candidates | 10 |
| Of these, new nonprobe candidates | 9 |
| Of these, reserved diagnostic probe | 1 |
| Blocked: repeated claim IDs require finer subclaim binding | 6 |
| No sentence-packing change | 8 |

The ten candidates span six papers. Nine paragraphs involve merges; one involves a split. Across complete plans there are **11 MERGE, one SPLIT and 31 KEEP operations**. Composite claims may be merged intact but cannot be split internally. A repeated composite ID across target sentences does not establish the missing subclaim boundaries.

No prerequisite-isolation program is established by the available maps. Conditional language inside a composite claim is insufficient to invent a new, separately executable prerequisite. This remains a representation gap rather than a successful operation.

## Contract and information boundary

`elementary_packing_edits_v1` accepts only existing `already_structured` annotations that pass the unchanged evidence/coverage validator. Claim IDs must form complete partitions of both sentence inventories, with each claim assigned exactly once on each side. Connected membership components produce:

- `KEEP`: one source sentence retained with its complete claim inventory.
- `MERGE`: contiguous complete source sentences expressed in one output sentence.
- `SPLIT`: separately express independently anchored claims already present in one source sentence, preserving claim order. Overlapping claim evidence is rejected.

Noncontiguous merges, sentence reordering and many-to-many regrouping are unsupported. There are no unrestricted rhetorical descriptions or lexical substitutions in this protocol. Each plan explicitly accounts for untouched sentences.

The generation payload contains only source text, source sentence IDs/text, source claim evidence, protected source evidence, source-anchored relations with their original certainty, and finite operations with output claim IDs. Reference wording, target evidence, annotator notes and references are excluded. Source claim IDs and output grouping are still derived from paired annotations: this is an **oracle diagnostic**, not a deployment evaluation. Sentence co-location does not license a new causal or inferential relationship. Interpretive relation annotations remain interpretations.

The compiler does not independently verify annotation accuracy, completeness, semantic equivalence or preservation by a generator. Composite claims remain indivisible bundles. Source metadata errors can propagate. The plans encode organization, not a validated signature of Fama/coauthor authorship. No model generations or style gains are reported from this compilation.

## Reproduction

From the repository root, with the existing private annotation package available:

```bash
PYTHONPATH=src python3 -m fama_style.rhetoric.elementary_edits \
  --package research_runs/operation_expansion_v1/package_v1 \
  --audit-dir research_runs/operation_expansion_v1/response_audit_v1 \
  --output-dir research_runs/elementary_packing_edits_v1/compiled_v1
```

The output directory must be new; use another version suffix when repeating. The utility verifies the frozen training split, package and original audit hashes, then writes private oracle candidates, dispositions, aggregate counts and a hash manifest. Original annotations, source/reference paragraphs and earlier scoped recoveries remain unchanged. The compiler reads neither development nor reserved test rows.

Code: `src/fama_style/rhetoric/elementary_edits.py`. Tests: `tests/test_elementary_edits.py`, including negative cases for ambiguous splits, evidence overlap, reordering, noncontiguous edits, coverage, reference leakage and unchanged private parents. All 297 CPU tests passed; `python3 train.py --check` also passed without model downloads or training. No training logic changed.

## Next decision

Use a small paper-diverse **inference-only** comparison of ordinary rewriting versus these oracle edits, including the one split and retaining the reserved probe only as a diagnostic. Inspect separately whether the requested organization is realized, whether uncertainty/attribution/claims shift, and whether the edit produces worthwhile style change. A sentence-count match alone is insufficient. This check has not been submitted or executed.

The nine nonprobe candidates are not nine training-ready labels. The original family gates remain unchanged, and no training dataset, training notebook, external-model call or GPU job was created. Only if realization provides useful evidence should finer subclaim annotations or a larger elementary-edit collection be pursued.

[Aggregate counts](elementary_packing_edits_v1_results.json). [Previous response audit and frozen gate](OPERATION_EXPANSION_RESULTS.md).
