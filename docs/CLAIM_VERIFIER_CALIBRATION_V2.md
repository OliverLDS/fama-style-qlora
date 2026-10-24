# Claim-aware NLI verifier: calibration v2

Status: implemented, frozen and CPU-tested. **Pretrained-model calibration has
not run. No improvement in verifier accuracy is claimed yet.** The previous
checker, benchmark fixtures, results and blocked 21-example style comparison
remain unchanged.

## Next Kaggle run

1. Upload `kaggle/verifier_calibration.ipynb` into a **new private notebook**.
2. Enable internet and **GPU T4 x2**. The first cell exposes GPU 0 only.
3. No input dataset, Gemma access or `HF_TOKEN` Secret is needed. The notebook
   contains only synthetic examples and uses `token=False` for the public NLI model.
4. Save & Run All, then share the notebook slug and version.

This evaluates a pretrained verifier. It performs no fine-tuning, optimizer
updates, Gemma text generation or structural-mapper training. No notebook was
uploaded or submitted by the implementation or builder.

Main output: `/kaggle/working/verifier_calibration_v2/evaluation.json`.
Other files include `results.json`, `runtime.json`, `run_status.json`, a hash
manifest and intermediate results. An interrupted run leaves partial records;
only a completed run with a final evaluation is a finished calibration.

## Why this differs from the previous checker

The CPU-only v1 checker detected 5/12 changed outputs and accepted none of twelve
nontrivial faithful paraphrases. It remains a useful warning system rather than
a general rewrite verifier. Adding lexical patterns alone would not address the
full task.

The new implementation in `src/fama_style/research/claim_verifier.py` combines:

- The same independent, pinned DeBERTa NLI model used in prior runs. Its original
  whole-document and sentence-unit assessment is retained as a baseline.
- Additional source-coverage and output-support checks on candidate claim units
  and adjacent two-unit bundles. Source coverage seeks missing information;
  output support seeks unsupported assertions. Evidence IDs and probabilities
  are retained for every candidate, not only the best alignment.
- Finite actor/quantity-role guards for fully parsed explicit payments,
  reimbursements and holdings, including supported passive forms. Swapping the
  same numbers between actors cannot escape merely because all numbers remain.
- The existing scoped bound-direction warning, which cannot be canceled by a
  correct bound repeated elsewhere.

Sentence units are not guaranteed atomic claims. Coordination is split only
when both sides fully parse in the finite fact grammar; otherwise the original
sentence stays intact. Conditions and causal links are not split away. Global
NLI checks retain the original unsplit text, so local units do not replace
paragraph-level context.

The role grammar rejects modality, quantifier and pronoun cases it cannot safely
scope. Other surface warnings remain visible but do not automatically reject a
paraphrase; the model must be tested for its ability to recognize equivalence.
This is not a general semantic parser, entity resolver, arithmetic engine or
verified claim ledger. Guard matches can be wrong; NLI can miss errors.

## Frozen evaluation

All 72 previously exposed v1 controls are now **development-only**, including the
old held-out cases. The new held-out set has 48 fresh synthetic cases across
sixteen scenario groups: sixteen identities, sixteen faithful paraphrases and
sixteen changed outputs. Beyond the twelve original error classes, it includes
strict bound endpoints, negation scope, association versus causality and
hypothesis versus established finding.

Implementation hashes were fixed before authoring the fresh cases. The cases
and labels were finalized before any pretrained-model evaluation. No v2 NLI
results have been observed or used to tune rules. Protocol and fixtures are in
`docs/claim_verifier_calibration_v2/`. These are author-written controls, not
blinded or independently annotated economic-domain data. Similarity to development
constructions and synthetic simplicity limit what a pass could establish.

The predeclared held-out gate requires:

- Error detection at least 80%.
- Faithful false alarms at most 10%.
- Nontrivial faithful-paraphrase acceptance at least 50%.
- No changed outputs accepted on this small set.

Thresholds for each NLI support check remain entailment at least 0.8 and
contradiction at most 0.1. Unknown/context-overflow checks abstain without
truncating text; abstentions remain in the denominators. The report compares
the new verifier with the original global/sentence NLI baseline on the same
cases, so gains from the guards/alignment can be distinguished from simply
using a pretrained semantic model.

`model_consistent` means the fallible proxy checks passed. It does not certify
meaning. `risk_signal` and `abstain` remain distinct. Every verdict records
`semantic_proof: false`. Reject-all, accept-all and abstain-all strategies fail
the gate. Passing this small benchmark would warrant further domain evaluation,
not automatically unlock training or certify the 21 development examples.

## Runtime and local checks

The model revision is `6f5cf0a2b59cabb106aca4c287eed12e357e90eb` for
`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`. Inference uses FP32 on GPU 0,
cached probability requests, a 512-token NLI limit, at most twelve units and
256 prediction pairs per case. Actual runtime, software versions, device,
cache size and peak allocated GPU memory are recorded. No execution-speed
result is claimed before the real run.

CPU tests use explicit fake predictors to test wiring, not to claim semantic
accuracy. They cover role swaps, quantities, passive forms, scope refusal,
bidirectional coverage, caching, probability validation, abstention budgets,
bound warnings, degenerate strategies, frozen-file checks and standalone export.
Stub execution is labeled and can never produce a passing readiness result.

```sh
python3 scripts/build_verifier_calibration_notebook.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m fama_style.research.claim_verifier \
  --output-dir /private/tmp/verifier-calibration-new-preflight
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

The self-contained notebook embeds code, synthetic cases and hashes only.
Preflight verifies the frozen source/fixtures before model loading. ML imports
occur only inside `run_training`, named for the repository convention; that
function calls no trainer. Mac checks require neither GPU nor model weights.
