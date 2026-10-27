# Operation expansion: annotation audit

All six files and all **30 expected annotations** were received. The proposed 20–30-pair training set is **not ready**: the three exact operation families remain too sparsely supported across papers. No GPU training was started.

The useful finding is not that these data lack all structural variation. Sixteen pairs labeled `already_structured` still change claim packing or order. The labels mainly indicate insufficient support for the current narrow families, rather than identical rhetoric. The next step should use those observed edits to improve the operation catalogue before further training.

## Original response audit

| Original annotations | Count |
| --- | ---: |
| Records present, with correct pair identities | 30 |
| Whole-paragraph schema/evidence/program contract passed | 28 |
| Rejected because programs omit paragraph claims | 2 |
| Reported `supported` | 6 |
| Reported `already_structured` | 24 |
| Missing records or duplicate IDs | 0 |

The six supported labels comprise four exact-seed declarations and two related variants. Only three exact-seed **whole-paragraph** programs and one related whole-paragraph program passed the original contract. The other two labels describe local operations whose programs do not cover all claims in the supplied paragraph.

All raw files and the original audit are preserved. No JSON was silently repaired, no preferred retry was selected, no source/reference text was replaced, and the whole-paragraph validator was not weakened. Evidence/schema checks are executable consistency checks, not independent proof of semantic equivalence or author-style labels. The annotation model/version was not supplied for these six files.

## Two local programs recovered separately

A useful local operation need not cover every topic in a long paragraph. Instead of relabeling the original records as valid, a separate recovery creates **verbatim contiguous excerpt tasks** whose complete claim inventories fit the declared program.

| Parent record | Explicit source scope | Explicit reference scope | Result |
| --- | --- | --- | --- |
| Volatility Lessons 2018, `0017` | First two supplied units | First three supplied units | Provisional exact premise/inference/qualification excerpt |
| Market Efficiency 1997, `0018` | Units 2–3 | Units 2–5 | Related explanatory premise/inference excerpt |

The source/reference strings are exact slices of the archived parent paragraphs, with character boundaries, original unit/claim IDs, hashes and excluded claims recorded. No sentence or factual wording was rewritten. Evidence occurrences are rebased to the excerpt; any evidence outside the declared scope causes failure. Outside claims remain in the full original record and are not silently removed from a whole-paragraph task.

For the 2018 excerpt, relations belonging to excluded later horizon claims are removed from the **derived** program. One erroneous `negation` protection tag is also removed from the derived copy: its evidence describes a positive-mean inference and uncertainty above/below an estimate, without negating the proposition. Quantity, comparison and uncertainty protections remain. The original annotation is unchanged and the exact change is logged.

The new excerpt retains 0.51% mean, 0.17% standard error, the 2.92-standard-error inference and the qualification about the true mean. The target's confidence wording is stronger than the source's evidence wording, and its “As noted above” marker remains context-dependent. These caveats are recorded, not silently normalized. No model realization has been run on this excerpt.

The bad-model excerpt retains the joint-testing premise, horizon comparison and its growth-rate explanation. The roadmap and AAR/CAR recommendation lie outside both excerpt tasks and remain in the parent paragraph. Its “Foremost” framing still depends on discourse context; this is a related variant, not a newly validated seed.

## Inventory and training gate

| Provisional program type | Full paragraphs | Contiguous excerpts | Total |
| --- | ---: | ---: | ---: |
| Exact seed declarations passing their respective unit contracts | 3 | 1 | 4 |
| Related candidates needing realization assessment | 1 | 1 | 2 |

The three full-paragraph exact seeds are the already-known 1965 anchors. The single additional exact-seed candidate comes from the 2018 excerpt. The related roadmap candidate from 1980 is also weak contrast evidence: its source already says first/then, while the reference separates the question introduction from the question. It remains outside primary training.

The original paragraph-only gate stays unchanged: **three exact paragraph pairs from one paper**. Counting the alternate excerpt units would yield four provisional exact programs from two papers, still far below the size, balance and per-family diversity requirements. The excerpt inventory explicitly does not override the original protocol. None is a newly certified gold label or training-ready model result.

## What `already_structured` conceals

Mechanical comparison of the accepted claim-to-unit annotations finds:

- **16/24** `already_structured` rows have different claim packing or first-evidence order.
- **10/24** introduce at least one new pair of co-located claims.
- **6/24** spread a claim over more target units, often involving composite-claim ambiguity.
- **8/24** have no measured packing/order change under this representation.

These sets overlap where applicable; they are not an additive taxonomy. In ten of the changed rows, no claim is spread over additional target units, making them cleaner starting points for inspecting combinations of already-distinct claims. This does not establish semantic correctness or automatic training eligibility.

Several notes explicitly recognize a split, merge, isolated prerequisite or separately expressed finding, but reject it as an instance of the particular seed plot. Therefore it would be wrong to conclude that 24 pairs have no stylistic or structural contrast, or that their source and reference trees are identical. Lexical/clausal style differences may also exist beyond these measurements. The finding is a **catalogue coverage and granularity problem**, with some genuine source-structure retention as well.

## Recommended next step

Keep the training gate closed and reuse the existing evidence maps to build a smaller set of **elementary claim-bound edits**:

- Separate already-distinct propositions or isolate an explicit prerequisite.
- Integrate an existing finding/comparison with its supported inference.
- Preserve untouched claims explicitly while applying a local operation.

Treat these as a new versioned representation, not retroactive success for the three narrow templates. Start with the ten cleaner changed rows and inspect source/target evidence. Cases that split one composite claim need finer source-subclaim binding before a generator can know which content belongs in each output move; repeating a composite ID is not enough.

Then test a small paper-diverse realization set before collecting more labels or training. Existing returned annotations can support this step; another round of the same narrow-family annotation is unlikely to solve the coverage gap. If later evidence shows too little presentation contrast even under the broader edits, test meaning-preserving, structurally varied neutral-source augmentation while retaining original author paragraphs as targets. Do not create those sources merely to satisfy an edit-distance quota or train on GPT-generated author-style targets.

## Reproduction and saved records

The new recovery utility is `src/fama_style/rhetoric/operation_scope_repair.py`, tested by `tests/test_operation_scope_repair.py`. It preserves the original validation failure and compiles alternate scoped tasks through the unchanged annotation validator. All **292 CPU tests passed**, and the original 1B CLI check passed. No ML dependency, model weights, Secret or GPU was needed.

Private locations:

- `research_runs/operation_expansion_v1/response_audit_v1/`: original coverage, checked responses, raw copies and original gate.
- `research_runs/operation_expansion_v1/scoped_recovery_v2/`: active scoped cards, exact repair log, original paragraph gate and separately labeled expanded-unit inventory.
- `research_runs/operation_expansion_v1/response_review_v1/`: per-record audit decisions, packing inventory, caveats and verification. The earlier scoped draft remains preserved but is superseded.

[Public aggregate results](operation_expansion_v1_results.json). All original data, raw responses, previous model outputs and the reserved test examples remain unchanged. No training notebook, external-model request or GPU run was created during this review.
