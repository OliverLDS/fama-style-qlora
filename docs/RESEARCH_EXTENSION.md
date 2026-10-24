# Claim preservation and style contrast: research extension

Status: CPU analysis and bounded prototype executed; candidate datasets prepared;
**no new model inference, training, GPU submission, or human scoring**.
The three behaviors are not solved. Existing pilot code and results are unchanged.

## A. What the 247 training pairs show

The extension reads only frozen training and validation files. Reference features
are fitted on training data only. The reserved test split is excluded from this
extension. Aggregate results are in [research_extension_metrics.json](research_extension_metrics.json).

| Measurement | Executed result |
|---|---:|
| Input/reference token sequence similarity, median [IQR] | 0.495 [0.399, 0.549] |
| Similarity p10–p90 | 0.339–0.629 |
| Exact normalized copies / pairs with similarity ≥0.85 | 0 / 0 |
| Reference/input token length ratio, median [IQR] | 1.197 [1.114, 1.372] |
| Mean tokens per sentence: input → reference | 20.45 → 27.74 |
| Pairs with longer reference sentences | 90.3% |
| Papers with positive mean sentence-length delta | 7 of 7 |
| Standardized style-change magnitude, median [IQR] | 0.275 [0.154, 0.604] |

Similarity is `SequenceMatcher` over case-folded word/number tokens, with autojunk
disabled. It is not an embedding score. Style-change magnitude averages absolute
paired differences in five features divided by training-reference standard
deviations (floor 0.1): sentence length, semicolons, we/our, initial if clauses,
and parentheses. These are limited surface proxies, not an author classifier.

There is substantial wording contrast to learn; an identity-dominated dataset
does not explain the observed copying. Sentence length is a consistent structural
signal, but semicolon frequencies are almost equal (0.061 vs 0.064 per 100
tokens). Lexical contrast alone does not establish distinctive author style.
Longer targets could also reflect lost information in simplified inputs.

The marker screen flags 218 pairs: modality 106, scope 123, causality 126,
negation 90, conditional cues 71; categories overlap. **These are not 218 semantic
errors.** Equivalent formulations can change markers, while identical markers
can conceal changed actors, scope or direction. Semantic alignment remains
unresolved without a calibrated semantic verifier. No semantic accuracy is reported.

## B. Executed training-free prototype

New modules in `src/fama_style/research/` implement:

1. `claims.py`: sentence evidence spans, actor/proposition hypotheses,
   conditions, quantities, negation, modality, quantifiers and causal cues with
   tentative direction. Unparsed scopes and actors remain explicit; implicit
   claims and coreference are not fully resolved.
2. `signal.py`: training-only empirical feature profile and pair distributions.
3. `prototype.py`: ledger → empirical operation plan → multiple bounded rule
   candidates → protected-marker checks and replay verification → style/copy
   filtering, with abstention.
4. `experiments.py`: separate SFT and preference preparation with parent IDs,
   operation chains, source/target hashes and weak-label quarantine.
5. `run_extension.py`: immutable output directories, split/hash gates and manifests.

The only empirically eligible operation here is joining independent-looking
sentences. The candidate grammar also supports simple conditional movement,
clause splitting and explicit writer-voice controls. These controls are not
automatically eligible style improvements. Protected-marker equality alone
never accepts an arbitrary candidate; unfamiliar paraphrases return `unknown`.
Replay proves membership in the bounded grammar, **not natural-language entailment**.

On 21 validation inputs: 56 candidates including identities; **zero nontrivial
selections, 21 unchanged abstentions**, mean source similarity 1.0 and style gain
0.0. Selection requires an eligible operation, style-distance reduction >0.05
and source similarity <0.90. This is an executed negative baseline: punctuation
rules cannot deliver adequate transformation under the copying constraint.
No LLM candidate generator or independent entailment model was executed.

Next CPU-capable step: richer candidate generation plus bidirectional,
claim-level entailment checks. Verification must test both input→output and
output→input, preserve entity/quantity bindings, and reject unresolved critical
claims. Independently generated adversarial controls must include actor swaps,
reversed causal direction, omitted conditions, negation and quantifier changes.
Require ≥95% mutation rejection and ≥90% acceptance on independently specified
equivalent controls before using scores as training filters. These are proposed
gates, not measured performance. Automated agreement is still not proof.

## C. Competing experiments prepared, not trained

| | A: Semantic-invariant SFT | B: Contrastive DPO |
|---|---|---|
| Hypothesis | Multiple faithful inputs per target weaken a single-input copying shortcut | Explicit bad alternatives teach fidelity and transformation beyond target likelihood |
| Prepared output | 397 bounded variants; original targets unchanged | 236 weak pairs: 129 semantic mutations, 102 copy controls, 5 generic surface controls |
| Main deficiency | Only 126 parents have ≥2 variants; only 8 have ≥2 variants with ≥10% lexical change | Only 14 pairs pass the surface-risk screen; 222 quarantined; no unquarantined generic controls |
| Failure mode | Tiny edits, target leakage, correlated paraphrase errors, overweighting prolific parents | Incorrect chosen labels, trivial negatives, length bias, reference-policy mistakes, reward exploitation |

All prepared rows have `training_ready: false`. Even the 14 unquarantined pairs
are weak labels; absence of marker mismatches does not establish faithfulness.
Generic controls are only voice/clause-splitting proxies, not a representative
generic-prose distribution. No target is silently repaired or relabeled.

Proposed A uses up to four diverse inputs per parent, unchanged targets, equal
parent weights and the existing explicit rewrite instruction/chat format at
training and evaluation. First require ≥80% parent coverage with two verified
inputs differing by ≥10% lexical tokens, and reject target leakage. Count parent
and token budgets rather than treating duplicated targets as independent data.

Proposed B starts from the same SFT snapshot, contrasting faithful target style
against copying, fluent generic prose and controlled semantic errors. Require
at least 100 automatically validated pairs, including ≥25 of each category;
then compare under matched parent/token budgets. Current preparation fails
both experiments' readiness gates. **Prefer A if diverse faithful augmentation
passes; prefer B only if reliable preference contrasts also pass. Otherwise
continue training-free generation/verification rather than another training run.**

### Literature and one-T4 feasibility

[STAMP, NAACL 2025](https://aclanthology.org/2025.naacl-long.135/) combines
pseudo-parallel SFT with iterative preference optimization and dynamically
weighted content/style/fluency rewards. Its high-resource setting assumes at
least 2,000 texts per style and evaluates other transfer domains. Our 247-pair
economics setting is not a replication. Candidate overgeneration is useful;
here semantic checks should gate style selection so a style reward cannot
compensate for a lost claim.

The current [TRL DPO documentation](https://huggingface.co/docs/trl/dpo_trainer)
and [implementation pinned at 88365e5](https://github.com/huggingface/trl/blob/88365e546b5be84821f1a8bd9f167ef3230950e2/trl/trainer/dpo_trainer.py)
support PEFT and cached reference log probabilities. The inspected source
recognizes a frozen adapter named `ref`; disabling adapters without it selects
the base reference. Preserve the intended SFT reference and fingerprint its
cached scores. Validate the fused LM-head API on Gemma, the text-only path,
frozen-reference equality and complete tokenization before adopting this TRL
version. No TRL package upgrade or GPU implementation has been performed.

Proposed settings retain the corpus pilot's Gemma 3 4B NF4 FP32, language q/v
LoRA rank 8, microbatch 1, accumulation 2 and complete sequences ≤1,024 tokens.
Original 1B workflows remain intact. DPO options are recorded separately:
sigmoid loss, beta 0.1, LR 5e-6, cached reference scores, `max_length=None`,
explicit external length rejection, no padding-free claim and no reference sync.
These configuration sketches are not a tested trainer.

The prior SFT pilot measured 6.79 GiB allocated / 11.30 GiB reserved on one T4
and about 12.4 training minutes for 247 examples. With equal parent budgets,
A should be near that order of cost; processing every augmentation instead
could multiply it. Planning allowance: **0.25–1 T4-hour for A, 0.5–1.5 T4-hours
for B at ≤500 preference pairs**, including evaluation/reference caching. These
are unmeasured estimates, not quoted prices or measured new runtimes. DPO has
two completion sequences per pair; one-T4 fit is plausible but unproven. A later
authorized five-update memory probe must precede either pilot. No job is submitted.

For eventual model selection, require no degradation in calibrated claim
preservation, fewer outputs with source similarity ≥0.90 than the frozen pilot,
and improved held-out surface-style distance on more examples than it worsens.
Report all three measures separately, including abstentions and paper-level
variation. Twenty-one validation paragraphs from one paper limit conclusions;
freeze decisions before any eventual reserved-test evaluation.

## Reproduce and inspect

```sh
python3 -m unittest discover -s tests -v
python3 train.py --check
PYTHONPATH=src python3 -m fama_style.research.run_extension \
  --dry-run --output-dir /private/tmp/fama-claim-style-dry-new
# Requires private frozen train/validation files; choose a new directory each run:
PYTHONPATH=src python3 -m fama_style.research.run_extension \
  --output-dir research_runs/claim_style_v1/new_run
```

Executed private outputs: `research_runs/claim_style_v1/final_results/`.
This directory is Git-ignored and contains paragraph-level diagnostics, ledgers,
candidates, prepared rows and manifests. Public files contain only code,
synthetic tests and aggregate metrics. No credentials are needed or accessed.
The preservation check hashes existing artifact bytes without using reserved
test text as development examples.
