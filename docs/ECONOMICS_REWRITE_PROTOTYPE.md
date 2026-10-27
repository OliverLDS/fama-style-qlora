# Economics rewriting prototype: ordinary by default, optional edit control

The [six-paper comparison](PAPER_PLAN_COMPARISON_V1_RESULTS.md) supports ordinary rewriting with shared evidence as the baseline and explicit plans as optional structural control. The local prompt builder implements that policy. **It prepares prompts for manual model use; it does not execute a model or claim distinctive Fama/coauthor-style performance.**

The historical training prompts and every completed experiment remain unchanged. This utility is separate from `train.py`; no model ID, training configuration or saved result is replaced.

## Ordinary mode

Write the input paragraph to a UTF-8 text file, then run from the repository root:

```bash
PYTHONPATH=src python3 -m fama_style.prompting.economics \
  --source-file source.txt \
  --output-dir research_runs/my_rewrite/ordinary
```

Ordinary is the default. The output contains `prompt.txt` to paste into a model, a `request.json` with the same prompt and its hash, and a preparation manifest. Use a new output directory; existing outputs are never overwritten.

With only a source file, the utility uses the established concrete name-free economics instruction and full source. **This plain-source mode has not been evaluated in the six-paper comparison.** That experiment supplied annotated source evidence in both arms; its results must not be presented as validation of the plain-source shortcut.

To reproduce the evaluated ordinary/shared-evidence format, also supply `--evidence-file evidence.json`. Evidence contains exactly `source_units`, `claim_index` and `source_relations`, using the existing source-only representation. No reference or target fields are accepted. The evidence is expanded into readable source excerpts and retains its recorded epistemic status/protected quantities/conditions/relations.

## Explicit planned mode

```bash
PYTHONPATH=src python3 -m fama_style.prompting.economics \
  --source-file source.txt \
  --evidence-file evidence.json \
  --mode planned \
  --plan-file plan.json \
  --output-dir research_runs/my_rewrite/planned
```

Planned mode requires explicit source evidence and an edit program. The plan uses only the existing `KEEP`, `MERGE` and `SPLIT` operations with source-unit IDs and output claim IDs. Ordinary mode rejects a supplied plan rather than silently activating it. This keeps deliberate restructuring separate from the default policy.

The checks require exact source spans, complete source-unit coverage, unique claim IDs, grounded relation evidence, finite operation labels, all whole claims accounted for and source units consumed once in order. KEEP uses one source/one output unit; MERGE uses multiple consecutive sources/one output unit; SPLIT uses one source/multiple output units. Splits require distinct claim evidence and preserve claim order. The builder cannot guess subclaim boundaries inside a composite claim or invent a semantic relationship to satisfy grouping.

These are structural/evidence checks, not semantic verification. They do not establish annotation accuracy, exhaustive atomic claims, causal validity, successful model execution or acceptable drift. There is no automatic claim extractor, rhetorical mapper, model API, response selector or training action. Evidence/plan preparation for a new paragraph is a separate task; none of the source/reference annotation packages is treated as a deployment-ready automatic annotator.

## Synthetic example

An invented two-sentence source, source-only evidence and a MERGE plan are saved at:

`research_runs/economics_rewrite_prototype_v1/example_v1/`

Its `ordinary/prompt.txt` and `planned/prompt.txt` are immediately inspectable. Both demonstration modes receive the same evidence, matching the design of the evaluated baseline. They are prompt artifacts, not generated rewrites or new performance results. No causal relation is supplied in this example; combining two statements must not manufacture one.

Keep real source paragraphs, evidence and generated text outside the public file selection. No credential belongs in an input file. Manual generation can use the already discussed GPT-6 High setup, but the builder does not select or invoke a model and records zero model calls.

## Interpretation and checks

Use explicit plans when a specific local organization matters. Do not equate plan compliance, low copying or the surface-style proxy with author-style success. Assess actual expression improvements and claim drift separately, retaining the existing allowance for minor shifts when benefits justify them. The mixed six-paper result remains the research conclusion; this policy does not constitute a new learned system or evidence for mapper/QLoRA training.

All **331 CPU tests passed**, and the original `python3 train.py --check` passed. New checks cover the default policy, explicit opt-in, reference-field rejection, changed/uncovered evidence, incomplete/unknown/out-of-order plans, ambiguous splits and output preservation. With the six existing cases and evidence, both utility modes reproduce all twelve frozen experimental prompts exactly. Tests also exercise credential-free, weight-free CLI preparation on synthetic text.

Implementation: `src/fama_style/prompting/economics.py`; tests: `tests/test_economics_prompting.py`. No existing dataset, annotation, model output, held-out example, training logic or frozen prompt was changed. No model requests, GPU jobs or training were started.
