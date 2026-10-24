# Direct versus rhetorical-structure style transfer

The next hypothesis is that explicit presentation controls can improve rewriting
before another QLoRA run. Keep the direct corpus pilot as the established baseline;
consider a hybrid only after an oracle plan helps. Neither approach is yet shown
superior. The oracle experiment is **prepared, not GPU-executed**.

## 1. Representation and semantic boundary

The training inventory contains 156 `level_1/level_2/level_3` annotations, 40
`R1/R2/R3` annotations, and 51 missing taxonomies. Their levels encode discourse
objectives, moves and relations. They are retained unchanged as target-side
provenance, not reused as independently extracted source labels.

The new `rhetoric_v1` specification applies the same deterministic rules separately
to source and reference. Its closed [JSON Schema](rhetorical_schema_v1.json) has:

| Level | Finite controls and observations |
|---|---|
| Paragraph | Organization: single, sequence, conditional, contrastive, reporting, mixed; sentence-count bucket; ordered move roles linked to claim IDs |
| Sentence | Simple/coordinated/subordinated/mixed syntax; condition position; author voice; length bucket; source evidence-unit identities |
| Language | Author voice, semicolon/parenthesis rates, suffix-based nominalization proxy; hedging and citation observations |

Move roles are assertion, definition, reporting, contrast, condition, conclusion,
unknown. Length buckets use ≤16, 17–32, >32 heuristic word/number tokens.
Count buckets are one/two/three-plus. Rate buckets are none/positive ≤2.5%/higher.
These are reproducible surface approximations, **not a validated discourse or
constituency parser**. The default assertion role can hide distinctions the rules
cannot recognize. Nominalization suffix counts are not a morphological analysis.

The separate [semantic ledger schema](rhetorical_semantic_ledger_v1.json) retains
source evidence, actor/predicate hypotheses, quantities, modality, scope,
conditions, negation, and tentative causal directions. Claim IDs currently identify
sentence evidence units, potentially containing several atomic claims. Actor
binding, coreference and atomic claim coverage remain unresolved. No positional
source/target sentence match is declared a validated claim alignment.

Every proposed edit references source evidence IDs. Source and target ledgers
remain separate. Lexical alignment hypotheses are explicitly unverified;
reordering or introducing causal links is prohibited. Replayable controls can
join independent-looking sentences, split clauses or relocate a simple explicit
conditional, using the earlier conservative grammar. Unknown operations, changed
source hashes and invalid claim scopes fail closed. Such replay is not semantic
proof. Template edits specify presentation, not permission to invent arguments.

Hedging and citations are **protected observations**, excluded from learned style
controls and oracle language templates. They must remain semantically faithful;
the system must not learn to add uncertainty or citations just to match frequency.
No reference words, technical-term lists, names, numbers, citations, text spans,
hashes or target claim IDs are serialized into the oracle plan. Source evidence
wording is allowed and is identical in B/C.

## 2. Executed training-only analysis

All 247 pairs have independent source/target annotations, provenance hashes,
claim-link checks, observed template differences, lexical alignment hypotheses,
and bounded executable-control diagnostics. Private files are under
`research_runs/rhetorical_v1/final_cpu/preparation/`; aggregate numbers are in
[rhetorical_v1_metrics.json](rhetorical_v1_metrics.json).

Across nine presentation controls there are **136 source signatures and 163 target
signatures**. Sources are not structurally constant. Syntax buckets change in
43.7% of pairs; sentence-length buckets in 42.9%; organization in 27.5%.
Sentence-count buckets change in only 11.7%, so simply merging every paragraph
is not supported. Bucketing hides some exact sentence-count changes.

Exploratory probes hold out each of the seven **training papers** in turn:

| Probe | Mean accuracy across nine categorical controls |
|---|---:|
| Retain source controls | 76.1% |
| Fixed target prior | 69.5% |
| Nearest source-structure retrieval | 66.8% |
| Smoothed conditional frequencies | 77.8% |

The conditional probe uses five prior-equivalent observations per source bucket;
retrieval uses Hamming distance and deterministic ties. None sees development
validation or reserved-test labels. No deployable mapper is fitted or saved.
The modest 1.7 percentage-point lift over source retention does not establish
useful generation gains. Common unchanged/absent features inflate accuracy;
source signatures need not encode enough information to predict all target choices.

**Recommendation:** test the oracle first. If it helps, retain fixed-prior and
source-identity baselines, then try regularized categorical edit probabilities.
The initial retrieval probe is weaker; a neural mapper is premature. These probes
test predictiveness of noisy annotations, not authentic authorship or causal effects.

## 3. Working oracle experiment

Upload [rhetorical_oracle.ipynb](../kaggle/rhetorical_oracle.ipynb) to a **new private
Kaggle notebook**. Attach the exact frozen corpus `train.jsonl` and
`validation.jsonl` split files, enable `HF_TOKEN`, internet and T4 x2. The notebook
exposes GPU 0. Do not attach the combined corpus or reserved test file. Use a new
session/output directory when rerunning; existing outputs are never overwritten.

| Condition | Generator input | Interpretation |
|---|---|---|
| A | Frozen ordinary rewrite instruction + source | Prompt-only anchor |
| B | Same instruction/source + source evidence units + source template | Structure/evidence control |
| C | Same instruction/source/evidence + reference-derived template | Oracle diagnostic; unavailable at deployment |

**C versus B is primary.** A versus B also changes explicit evidence prompting,
so that contrast cannot isolate structure alone. Target slots are unbound
presentation templates, not a positional assignment of target claims to source
claims. Every condition preserves the original input. C is a *noisy reference-derived
oracle*, not a gold annotation: failure can reflect annotation quality, the
representation, or realization ability, rather than disproving all structural control.

The runner prepares 63 jobs over the same 21 validation examples, cycles condition
order, pins the corpus pilot's Gemma 3 4B revision, and uses NF4 FP32, greedy
decoding, seed 42, 512 new tokens and a 4,096-token context guard. All prompts are
token-audited without truncation. Empty/cutoff outputs stop the run; partial results
remain marked incomplete. The original 1B training workflow is unchanged.
Runtime compatibility and memory fit of this new notebook are not GPU-tested.

### Automated evaluation and decision gate

- **Semantics:** protected-marker mismatches, lexical evidence coverage and bounded
  replay checks, plus independent bidirectional document/sentence-unit NLI. These
  are separate measures; surface equality alone never certifies fidelity.
- **Adherence:** equal-weight mean of nine bucket matches, ordered paragraph-role
  similarity and ordered sentence-pattern similarity. Report requested-plan
  adherence and oracle adherence separately; A has no requested plan.
- **Copying:** exact copying and source token-sequence similarity ≥0.90.
- **Style:** distance change to training-reference features; report independently
  of claim checks and structural adherence. This is not an author-style classifier.
- **Cost:** input/output tokens, per-generation seconds, NLI/wall time and peak GPU
  memory. B/C duplicate source evidence and have longer prompts; costs are measured.

After generation, Gemma is unloaded. The separate
[DeBERTa NLI model](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli)
runs on CPU with its resolved revision and label mapping recorded. Each assessment
checks both whole-paragraph directions and each sentence against the opposite
paragraph. Context overflow abstains; nothing is silently truncated. Three
equivalence controls and seven actor/causality/condition/scope/modality/negation/
quantity mutations must pass automatic sanity checks before reporting a usable
NLI proxy rate. These small controls are not economic-domain calibration. NLI
can still miss scope errors and is not a semantic proof; no human review is required.

The preregistered exploratory gate requires C to improve combined oracle adherence
over B by ≥0.05, improve the style proxy over A and B, not worsen near-copy rates,
and achieve ≥90% NLI proxy consistency without degrading relative to A/B. All
synthetic controls must pass. Unknown semantic assessments or a failed control
gate leave mapping selection unresolved. Even a pass supports only a proxy-based
follow-up; 21 paragraphs from one paper cannot establish broad generalization.
Do not tune these thresholds on the reserved 45 examples.

## 4. Follow-up design, conditional on oracle evidence

Compare fixed prior, retrieval and regularized probabilistic edits under paper-grouped
development controls before considering a neural structure model. Protect semantic
features and claim identities; do not synthesize unconstrained rhetorical trees.
If C fails, inspect extraction, schema information and language realization first.

The prepared eventual factorial design is:

| Weights | No learned mapping | Learned mapping |
|---|---|---|
| Base Gemma | Ordinary rewriting | Structural transfer |
| Frozen QLoRA | Direct adaptation | Hybrid |

Use identical inputs, base revision, quantization and decoding controls. Freeze
the mapper before generation comparison. The existing adapter was not trained
on plan prompts; a future plan-conditioned adapter would be a **separate** experiment.
No new GPU training, mapping deployment or factorial result is reported here.

## 5. Reproduce and inspect

```sh
python3 -m unittest discover -s tests -v
python3 train.py --check
PYTHONPATH=src python3 -m fama_style.rhetoric.experiment \
  --dry-run --output-dir /private/tmp/fama-rhetoric-new-dry
# Private local split files, CPU preparation only:
PYTHONPATH=src python3 -m fama_style.rhetoric.experiment \
  --output-dir research_runs/rhetorical_v1/new_preparation
python3 scripts/build_rhetorical_notebook.py
```

The standalone notebook export passes a synthetic CPU run outside the checkout.
Dry runs label identity outputs as synthetic wiring checks, never model results.
Code and aggregate metrics are public candidates; paired schemas, prompts,
generations and paragraph-level scores remain Git-ignored. Credentials are read
only inside explicitly requested inference, never written into outputs or code.
Existing experiments remain byte-identical. The notebook was not uploaded or run.

### Research precedents

[StoryTrans](https://aclanthology.org/2023.acl-long.827/) investigates discourse
representations for story author-style transfer;
[RSTGen](https://aclanthology.org/2022.naacl-main.133/) controls long-form generation
with rhetorical structure;
[SCPN](https://aclanthology.org/N18-1170/) conditions paraphrases on syntax;
[low-level linguistic controls](https://aclanthology.org/W19-8628/) use explicit
linguistic features. These support investigating the components; none establishes
that this schema or a learned mapper improves faithful economics rewriting.
