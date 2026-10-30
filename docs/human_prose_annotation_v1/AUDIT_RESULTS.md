# Human-prose annotation audit and next diagnostic

This records the annotation-audit checkpoint. The subsequently returned 16
realizations are reviewed in [REALIZATION_RESULTS.md](REALIZATION_RESULTS.md).

All 20 manual batches were received: 100 records from 20 papers. All pass the
frozen output schema, expected-ID coverage and exact-original checks. Raw files
and coverage reports are preserved by SHA-256. User-reported provenance is GPT-6,
High thinking effort, separate chats with possible automatic retrieval of prior
chat context. Hidden snapshots and decoding settings are unknown. Separate chats
do not establish independence; the blank session template was not invented.

Only the 60 training and 20 development annotations were content-audited. The 20
reserved records received transport/schema/original-integrity checks only. No
reserved metrics, category discovery, prompt selection or generation was done.

| Check | Training (60) | Development (20) |
|---|---:|---:|
| Mechanical evidence, IDs, sentence coverage and taxonomy consistency pass | 60 | 20 |
| Median sequence lexical overlap | 0.777 | 0.756 |
| Overlap at least 0.90 | 4 | 1 |
| Normalized token copies | 0 | 1 |
| Different sentence count | 29 | 7 |
| Different claim coexpression grouping | 27 | 7 |
| Annotator requests review | 44 | 17 |

There are 6,093 exact evidence anchors, 539 claims, 172 relations and 4,120 taxonomy
observations in the nonreserved annotations. Exact quotes and resolvable links
establish mechanical consistency, not that labels, claim inventories or semantic
alignments are correct. Numeric literal multisets agree in all 80 pairs; this
does not verify their roles, comparisons or scope.

## Repairs and limitations

Six transformation labels used SPLIT in the opposite direction to the requested
neutral-source → human-target task. Their sentence bindings support MERGE. One
additional transformation omitted a source sentence carrying a declared claim.
Seven restricted program-field patches across five rows were applied in audit_v2;
audit_v1, all raw responses, text, claims, evidence and taxonomy labels remain
unchanged. Original free-text explanations remain raw annotation statements;
consult the patch rationale for normalized operation direction. Neither training
nor prior Fama experiments were modified.

There are 129 warnings where taxonomy observations cite claims outside their
bound sentences. They are not silently expanded or treated as executable plans.
Guidance construction excludes observations without contained claim bindings and
excludes semantic-sensitive or merely descriptive category controls. These
filters still cannot establish that a label is linguistically correct.

The annotator flags two training claims uncertain, including an ambiguous
antecedent and an explicit interpretation of a possible source typo. A limited
assistant reading of the eight selected development pairs finds another issue:
“probably” becomes “possible” in one rewrite, weakening epistemic strength despite
the annotator's preserved label. This is recorded privately without changing
prose. The other seven had no clear substantive change noticed in that limited
reading; this is not exhaustive or independent model-based verification.
Most annotation review flags concern missing discourse context. Do not equate
all such flags with failed pairs, or assume unflagged pairs are faithful.

The pairs offer some organizational contrast, but high overlap limits the
available transformation signal. These measurements do not establish distinctive
author style, predictable source-to-target mappings, or training readiness. No
pairs are certified semantically equivalent or approved for training by this audit.

## Prepared inference comparison — not executed

Private package: `research_runs/human_prose_realization_v1/package_v1/START_HERE.md`.
Two balanced batches request 16 outputs: ordinary rewriting and optional
presentation guidance on eight development examples, two from each of four
papers. Both conditions receive the same neutral input and source-quoted claim
evidence. Only guidance differs. Author names, paper identifiers, reference prose,
original-derived propositions and free-text annotation explanations are withheld.
Preserved citations may still reveal provenance. Plans include target sentence
claim groups and finite, checked presentation-category definitions.

This is a reference-informed oracle feasibility check, not a source-only mapper,
author imitation result or evidence of human authorship. No generation has run.
The known weakened-input case remains explicitly flagged as a sensitivity case.
Compare generated claims with the supplied input first; report reference-alignment
changes separately so existing input drift is not credited to or blamed on the
realizer. Keep copying, readability, organization, plan adherence and claim drift
separate. Do not interpret increased reference similarity alone as improvement.
The annotations and proposed realization use the same user-visible model family;
cross-chat retrieval and hidden decoding settings remain uncontrolled.

## Reproduction and tests

CPU-only entry points (use fresh output directories):

```sh
PYTHONPATH=src python3 -m fama_style.research.human_prose_review --package research_runs/human_prose_annotation_v1/package_v2 --responses research_runs/human_prose_annotation_v1/responses --session research_runs/human_prose_annotation_v1/session_confirmation.private.json --repairs research_runs/human_prose_annotation_v1/program_repairs_v1.private.json --output /tmp/human_prose_audit
PYTHONPATH=src python3 -m fama_style.research.human_prose_inference --audit /tmp/human_prose_audit --package research_runs/human_prose_annotation_v1/package_v2 --output /tmp/human_prose_realization
python3 -m unittest discover -s tests -v
```

New tests cover closed-schema failures, exact quote occurrence, constrained repairs
and immutable inputs, source-only generator evidence, reference-wording exclusion,
matched condition payloads, and rejection of unsupported or protected controls.
No ML imports, downloaded weights, detector calls, model generation or GPU jobs.
Public artifacts contain aggregate counts and methods; private prose stays ignored.
