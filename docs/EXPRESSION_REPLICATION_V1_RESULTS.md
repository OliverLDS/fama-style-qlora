# Expression replication v1: controllability replicates, quality gain does not

**The requested forms were realized in all four guided outputs, but no clear
incremental style benefit was found over the matched grouping controls.** Several
controls already contained the relevant form. These results do not justify
training a rhetorical mapper or another adapter yet.

All twelve responses were accepted with valid job IDs, prompt hashes, batch
assignments and unique coverage. The user reports GPT-6 with High thinking effort.
Model identity and hidden decoding are not independently verified; conversation
arrangement was not reported. Three raw files are preserved byte-for-byte. No
additional generation, GPU execution or training was started in this review.

## What replicated

All four guided outputs follow the fixed grouping and the strict requested form.
All four matched grouping controls also follow that grouping. Operation evidence
is bound to the affected claims; VL2018's untargeted longer-horizon claim remains
in its three-claim slot. F1965 keeps its identity grouping, rather than inventing
a merge/split improvement.

| Case | What the control already does | Increment from guidance | Quality finding |
|---|---|---|---|
| F1965 theme | Distribution already is the grammatical subject, after an initial independence frame | Moves the independence frame to the end | No clear improvement; not a new subject choice |
| ME1997 theme | Starts with model authors | Promotes the modeled economic variable to an initial passive subject; attribution retained | Genuine controllable theme change, but no clear writing-quality advantage |
| FF2000 finite consequence | Already has an explicit finite result clause | Uses the specified similar-merger-rate actor as subject instead of payer percentage | Mostly repeats the actor; not a new finite-clause gain |
| VL2018 finite consequence | Already has the requested actor and finite predicate in the fixed slot | Rephrases the predicate | Ceiling case; no clear incremental benefit |

The strict theme score requires both subject choice and initial position. Thus
F1965's control does not meet the full instruction, although its subject already
matches. Likewise, FF2000's control meets the finite-predicate requirement but not
the specified grammatical-actor preference. Strict instruction matching must not
be reported as a quality gain, an absent source actor or improved semantics.

The four guided forms therefore demonstrate that the model can respond to local
instructions. They do not show that these instructions are generally useful or
that a trained source-only mapper would improve ordinary writing. The modest
finite-consequence clarity benefit in the initial F1980 example did not clearly
replicate in the two additional finite cases.

## Claim fidelity

Fifty-seven claim instances received source/output evidence bindings. In this
assistant review, 56 were judged retained and one received a possible force-shift
flag. These are exploratory judgments, not independent entailment labels or an
accuracy estimate. No clear major semantic drift was found.

The flagged VL2018 grouping control changes a description of risk importance to
one that can suggest risk magnitude. This is a possible small calibration change,
not a verified material error. The guided wording retains source importance more
closely, but that off-target difference cannot be attributed to the c2/c3
expression operation. All conditions received the same semantic reminders.

The regime example retains BSV attribution, two judgment biases, actual
random-walk earnings and investors' mistaken regime perceptions. The dividend
example retains annual rate/cohort/year assignments and count-versus-percentage
directions. The volatility example retains chance, investment-evaluation horizons
and nontrivial ten/20-year probabilities. All three F1965 conditions express the
definition as a property under independence; identical definitional/pragmatic
force is not independently certified.

## Copying and off-target changes

| Paper | Grouping-control/source overlap | Guided/source overlap | Token edits, control → guided |
|---|---:|---:|---:|
| F1965 | 0.765 | 0.716 | 15 → 19 |
| ME1997 | 0.874 | 0.826 | 21 → 30 |
| FF2000 | 0.834 | 0.792 | 23 → 27 |
| VL2018 | 0.809 | 0.797 | 23 → 22 |

No response is an exact normalized copy. Mean source overlap decreases from about
0.821 to 0.783 in the primary comparison, but much source wording remains and
some edits occur outside the intended slot. Lower overlap is not automatically
better style, and word order alone can change it. The source-token edit count is
Levenshtein distance; overlap uses normalized-token SequenceMatcher. Its greedy,
order-sensitive alignment can also make reference-overlap values unstable despite
retained claim content. Reference overlap is retained privately for diagnostics
and is not used as evidence of semantic recovery or distinctive author style.

F1965 ordinary is especially close to its source (overlap 0.952). That valid first
attempt remains in the analysis. No ceiling case, conservative answer or negative
result was replaced. The ordinary regime answer spontaneously matches the planned
grouping, further limiting what grouping compliance alone establishes.

## Decision and next research step

The mechanism is feasible, but these features are neither established as
consistently beneficial nor author-distinctive. Keep ordinary rewriting as the
prototype default, with expression guidance optional and diagnostic. Do not fit a
mapper, start QLoRA training or generalize an unconditional style policy from
these examples. No prototype code or default was changed by this review.

The next useful research step is to investigate author specificity using comparable
non-Fama academic economics prose, rather than another round of clause-level
realization checks. A small topic/period/genre-matched comparison corpus would
permit a paper-grouped discriminative task with rhetorical features, a lexical
baseline and topic/length/coauthor controls. Successful classification alone would
still not prove that a selected feature improves rewriting. No comparison corpus
was acquired and no classifier was trained in this task.

The source-only mapping question remains open. If meaningful, distinctive features
are identified, test their realized value before learning to select them. The
existing direct-QLoRA experiments remain a separate preserved baseline.

## Reproduction and limitations

Private artifacts are separate from every earlier release:

- `research_runs/expression_replication_v1/import_v1/`: preserved raw replies and diagnostics before assessment.
- `research_runs/expression_replication_v1/review_spec_v1/`: hash-bound judgments and component observations.
- `research_runs/expression_replication_v1/review_v1/`: raw copies, evidence, metrics, summary and manifest.

Source/reference paragraphs, generated outputs and detailed evidence stay private.
Public files contain aggregate findings and methods. Original corpus, inputs,
annotations, prompt package, previous results and historical hashes are unchanged.

```sh
PYTHONPATH=src python3 -m fama_style.research.expression_replication_review \
  --package research_runs/expression_replication_v1/package_v1 \
  --responses research_runs/expression_replication_v1/responses \
  --assessments research_runs/expression_replication_v1/review_spec_v1/assessments.private.json \
  --output-dir research_runs/expression_replication_v1/review_v2
```

Choose a fresh output directory. Without assessments, import reports surface
metrics only and leaves quality pending. CPU tests check component/strict-form
consistency, inherited claim/evidence validation, exact raw bytes and reproducible
results. Mechanical checks do not prove grammatical roles, entailment or quality.

Limitations include a purposive four-case training sample, two cases per
operation, prior source inspection/exposure, oracle grouping except for the
identity case, the disclosed v1.1 scope extensions, uncontrolled hidden settings,
possible memory/source familiarity, off-target wording changes and unblinded
assistant assessment. No validation or reserved test rows were used. No held-out,
population-level or distinctive Fama/coauthor-style result is claimed.
