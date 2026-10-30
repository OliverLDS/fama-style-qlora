# Claim-bound expression v1: local control succeeds, broader gain limited

**Both requested expression operations were realized at fixed claim grouping.**
The finite consequence offers a modest local clarity gain; the theme change is
controllable but not clearly better prose. No clear substantive drift was found
in this reading. Neither operation establishes distinctive Fama/coauthor style or
solves copying.

All six first-attempt outputs passed coverage, prompt-hash and batch checks. The
user reports GPT-6 with High thinking effort. Hidden decoding, identity and chat
arrangement are not independently verified; arrangement was not reported. No
additional generation, GPU job or training was started during import/review.

## The primary matched comparisons

| Paper | Grouping-only control | Expression guidance | Meaning assessment |
|---|---|---|---|
| FJ1983 | Hierarchies remain the grammatical theme | Decision-control structure becomes the theme; grouping remains three sentences | Core relation, scopes, stages and monitoring origin retained |
| F1980 | Manager-stake consequence is a participial tail | Explicit manager subject and finite predicate inside the same final slot; grouping remains four sentences | Rental information, qualified current pay, asserted future wages and stake relation retained |

FJ1983 shows that a source-anchored focus can control grammatical presentation.
The output preserves the ratification definition, lower/higher-level actor roles,
large-organization scopes, mutual monitoring and human-capital development. The
new construction is not clearly more concise or readable than its control, and
almost the entire remaining paragraph stays unchanged. Realization is stronger
evidence than before, but quality superiority is not established.

F1980 makes the manager-stake consequence explicit with its own actor and finite
predication, instead of leaving it as a participial result. This retains the causal
link to future wages while preserving uncertainty about current pay. The result
is a modest increase in explicitness at the same grouping, not a newly inferred
relationship or a uniquely author-specific construction. The opening also changes
outside the intended slot, so whole-paragraph metric differences cannot be
attributed solely to the final-clause operation.

## Ordinary baselines and shared reminders

The new ordinary FJ1983 output has four sentences and foregrounds hierarchies.
Ordinary F1980 has six sentences and already gives the consequence a finite
predicate, but in a separate sentence. Thus the finite operation is not absent
from ordinary writing: the diagnostic success is retaining it **inside the same
packed claim group**.

The previously weakened monitoring-origin relationship is explicit in all three
new FJ1983 outputs. All conditions received identical source-meaning reminders,
including that relationship. Its retention cannot be attributed to expression
guidance, and the earlier run is not a concurrent reminder-ablation control.

Thirty claim instances were bound to source/output evidence and judged retained
on in-session reading. No clear substantive drift was identified. These are not
independent entailment labels, an accuracy estimate or proof of exhaustive claim
coverage. Broad sentence evidence anchors establish traceability; they do not
mechanically establish grammatical role or causal interpretation.

## Copying and reference resemblance

| Paper / condition | Source overlap | Token edits from source | Reference overlap, evaluation only |
|---|---:|---:|---:|
| FJ1983 ordinary | 0.825 | 19 | 0.538 |
| FJ1983 grouping control | 0.825 | 19 | 0.538 |
| FJ1983 expression | 0.850 | 17 | 0.526 |
| F1980 ordinary | 0.824 | 27 | 0.343 |
| F1980 grouping control | 0.825 | 24 | 0.393 |
| F1980 expression | 0.867 | 22 | 0.370 |

Overlap is normalized-token SequenceMatcher similarity; token edits use token
Levenshtein distance. Neither is a semantic or author-style score. No output is
an exact normalized source copy. Nevertheless, both guided outputs are closer to
the supplied source than their grouping controls. Their lexical similarity to
the reference also decreases slightly. Neither observation disproves correct
local realization, but there is no copying reduction or broad reference-style
gain to report.

## Decision

The mechanism now has a positive feasibility result: small source-grounded
expression instructions can change presentation while retaining the requested
grouping and apparently preserving the inventoried meaning. This supports a
bounded replication across additional eligible training papers, with the same
ordinary/grouping/expression controls and explicit local benefit/drift judgments.
It does not yet support mapper training, another QLoRA run, automatic prototype
policy changes or claims of learning an author style.

Replicate the finite-consequence operation where an independently source-supported
edge is present, and the theme operation where the source actually supports the
relation/function. Do not impose either universally. The label-reuse operation
remains untested. A discriminative author-style experiment still requires a
comparable economics corpus and paper/topic/coauthor controls; these two cases
cannot establish author specificity. No validation or reserved test rows were used.

## Artifacts and reproduction

The prepared package and original responses remain unchanged. Private artifacts:

- `research_runs/claim_bound_expression_v1/import_v1/`: raw preserved files and surface diagnostics before assessment.
- `research_runs/claim_bound_expression_v1/review_spec_v1/`: hash-bound claim and operation judgments.
- `research_runs/claim_bound_expression_v1/review_v1/`: raw files, evidence assessments, metrics, aggregates and manifest.

Public files contain aggregate methodology/results only; source paragraphs,
references, generated prose and detailed evidence stay private. No corpus,
annotation, repair overlay, earlier result or historical hash was overwritten.

```sh
PYTHONPATH=src python3 -m fama_style.research.claim_bound_expression_review \
  --package research_runs/claim_bound_expression_v1/package_v1 \
  --responses research_runs/claim_bound_expression_v1/responses \
  --assessments research_runs/claim_bound_expression_v1/review_spec_v1/assessments.private.json \
  --output-dir research_runs/claim_bound_expression_v1/review_v2
```

Choose a fresh output directory. Omitting assessments imports surface diagnostics
only and leaves quality pending. CPU tests verify text identity, claim coverage,
operation/evidence ownership, grouping consistency, deterministic reproduction
and preservation of raw bytes, not correctness of semantic/style judgments.

Limitations include two previously inspected training examples, one instance per
operation, reference-informed oracle evidence/grouping, hidden settings, possible
memory/source familiarity, unblinded in-session assessment and off-target wording
changes. These are mechanism diagnostics, not held-out generalization evidence.
