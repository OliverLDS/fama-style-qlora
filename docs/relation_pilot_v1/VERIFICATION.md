# Preferred-merge verification packet

Current status: [all thirteen responses audited](VERIFICATION_RESULTS.md).
The preparation description below is retained as the original protocol.

**Prepared, not executed:** 52 fixed candidates from 27 neutral paragraphs across
four papers, in thirteen batches of four. Selection includes every Stage 2
MERGE-preferred candidate, including all nine with surface warnings. No candidate
or annotation is changed and no verification result is claimed.

Private entry point:
`research_runs/relation_pilot_v1/verification_package_v1/START_HERE.md`.
For each `batches/batch_01` through `batch_13`, paste `TASK.md` and attach only
its `items.jsonl` plus `common/verification_output.schema.json`. Use GPT-6 and
request High effort. Report effective effort as uncertain when it is unknown;
the requested setting does not establish the effective setting. Save first
complete unedited `responses_01.jsonl` through `responses_13.jsonl` and coverage
reports in this packet's `responses/`.

## What is checked

Each item exports exactly an opaque `item_id`, full neutral `source_paragraph`,
the adjacent `left_sentence` and `right_sentence`, and their fixed `candidate`.
It exports no author reference, prior D-claims, relation, preference, permission,
rationale, source metadata or risk flags. The verifier extracts fresh V-units
from the indicated pair and supplies exact source/candidate evidence. The full
paragraph supports referent checks; its other substantive claims need not be
repeated and must not be silently imported into the local replacement.

Fidelity is judged as **faithful, local_shift, material_shift or unresolved**.
Readability is assessed separately as **merged_better, separated_better,
comparable or unresolved**. A more readable candidate can change meaning; a
faithful one can read worse. Limited shifts must be described rather than
automatically accepted or rejected for training. Added content and relations,
omitted claims, conditions, actors, quantities, modality, negation, attribution
and speech acts are explicitly checked. Candidates are assessed as supplied;
repairing them during verification would change the experiment.

Exact evidence includes one-based occurrence numbers. Reference resolution can
use surrounding source text with recorded context evidence. Missing context and
pre-existing ambiguity must be distinguished from changes introduced by the
candidate. No evidence-free economic reconstruction is allowed. Literal word
or numeric inventory differences alone are not semantic errors.

The same paragraph never repeats within a batch; each batch contains three or
four papers. Some paragraphs recur across batches for different local candidates.
Private mapping retains warning status for later sensitivity analysis. The
merge operation is visible, while earlier judgments are withheld. Because this
is a selected-positive sample, it cannot establish preference classification
accuracy, rejected-candidate specificity or whole-corpus merge quality.

GPT-6 checking its earlier GPT-6 candidates is not independent verification.
Separate chats do not establish independence from automatic retrieval or
pretraining. No human review is required. Pair checks do not license chaining
overlapping merges. No acceptance thresholds, classifier fit, new generation
submission or GPU training are implemented by this preparation.

## Files and reproduction

Public: [prompt](verification_prompt.md),
[output schema](verification_output.schema.json),
[preparation counts](verification_preparation.json).
Private: source text, candidates, mappings and eventual responses, all under the
Git-ignored research run. Existing Stage 1/2 packets, raw responses, labels,
hashes and results remain unchanged.

Build with `python3 -m fama_style.research.relation_pilot_verification` using
`PYTHONPATH=src`, `--audited` pointing to the frozen Stage 2 audit, `--selection`
pointing to the private Stage 1 selection metadata, and a fresh `--output` path.
Manifests pin code, audit, metadata and packet bytes. The mutable session-notes
form and future response files are excluded from the preparation manifest.
CPU checks cover prior-judgment/reference exclusion, exact candidate retention,
deterministic paragraph separation, inclusion of flagged jobs, schema categories,
uncertain effort, stale hashes, wrong counts and overwrite refusal.
