# Expanded teacher-pair pilot: stronger fit, unresolved claim drift

Completed as `oliverlds/teacher-pair-expanded-pilot-4b`, requested version 1.
The authorized run performed **46 finite optimizer updates**, saved both
language-only LoRA adapters and passed its exact adapter-weight reload check.
This review retrieved the full artifacts and log without further model calls,
GPU jobs, dataset changes or training. The original 16-pair pilot is preserved.

The result is mixed: loss and some expression improve, but material claim issues
remain in **six of eight development outputs**, at both epochs. The predeclared
criterion required fewer than six; it fails. No full-scale training is justified
by this result, and no additional run or budget is authorized.

## Fixed-case comparison

| Measure | Base | Earlier pilot, epoch 2 | Expanded epoch 1 | Expanded epoch 2 |
| --- | ---: | ---: | ---: | ---: |
| Training pairs | — | 16 | 46 | 46 |
| Optimizer updates | 0 | 16 | 23 | 46 |
| Target-token-weighted development loss | 2.96942 | 2.61502 | 2.19141 | 1.89590 |
| Development outputs with material claim issues | 7/8 | 6/8 | 6/8 | 6/8 |
| Generic expression wins / ties / losses vs base | — | 5 / 2 / 1 | 5 / 3 / 0 | 6 / 1 / 1 |
| Exact source copies | 0/8 | 0/8 | 0/8 | 0/8 |
| Median source word-sequence overlap | 0.288 | 0.304 | 0.325 | 0.351 |
| Median reference word-sequence overlap | 0.226 | 0.245 | 0.262 | 0.257 |
| Material issues in two training probes | 2/2 | 2/2 | 1/2 | 0/2 |

The final loss reduction versus base is **36.15%**, compared with 11.9% for the
earlier pilot. Lower loss indicates better target-token fit, not a measured
36% improvement in semantic accuracy or author style. No output is empty or
length-limited; all ten outputs differ between epochs. The new pilot's base
snapshot reproduces all ten reviewed baseline outputs exactly.

Training data volume, optimizer budget and the linear learning-rate trajectory
all change relative to the earlier pilot. This comparison does not isolate
the causal contribution of additional data. Style counts are each against the
common base, not a direct blinded preference between the two adapters.

## What changed in meaning

| Development case | Expanded epoch 1 | Expanded epoch 2 |
| --- | --- | --- |
| IPO and investment | Adds internal-control benefit comparison; subsequent investment becomes planned investment | Removes that invented mechanism, but subsequent still becomes planned investment |
| Central/decentralized planning | Main institutional distinction retained; explicit not-whether-planning statement missing | Adds unsupported uncertainty about institutional efficacy and causal effects |
| Probabilistic/contingent insurance | Main conjecture and probability/outcome framing retained; some example/procedural details omitted | Similar main account; minor coverage/wording flags remain |
| Investor ownership incentives | Invents monitoring-cost and dispute mechanisms | Adds information-advantage/enforcement and ownership-retention uncertainty claims |
| Reflection effect | Omits numerical indifference prospects and attribution details; adds inaccurate framing | Restores 80%/4,000/3,000 and citations, but still omits both numerical indifference examples and weakens majority scope |
| Owners as patrons | Changes most to frequently; turns analytical purpose into recommendation | Retains planned comparison better, but most still becomes frequently and corporation ownership becomes often |
| Price-mechanism marvel | Omits rhetorical purpose, strongest counterfactual and skeptic position; adds perspective-shift claim | Improves first-person counterfactual organization but retains material omissions |
| Bank loan register | Corrects individual-loan threshold but adds solely internal monitoring and loses on-request access | Retains individual-loan threshold, on-request access and reported monitoring/confidentiality claims; minor wording limits remain |

The identical material-issue count conceals a tradeoff: epoch 2 improves the bank
register case and regresses on planning relative to epoch 1. Compared with the
earlier pilot's final epoch, both expanded epochs recover the insurance case
but lose a different previously acceptable case: the bank example in epoch 1,
and planning in epoch 2. Report these transitions rather than claiming unchanged
counts mean unchanged behavior.

The two training probes show a clearer learning signal. By epoch 2, the fee
example states that the agent knows the principal's information, and the price
example retains successive monthly means and statistical nonsignificance.
They are not exact target reproductions: acknowledged-versus-known wording and
an added generic uncertainty caveat remain minor flags. These are training-side
diagnostics, not extra held-out evidence or proof of complete semantic fidelity.

## Style and copying

Epoch 2 wins generic organization/concision comparisons in six development
cases across four papers, ties in one and loses in one. The clearer counterfactual
organization in the mechanism example and shorter non-elaborative phrasing in
the IPO and bank examples are useful expression changes. Planning's unsupported
coda is also diffuse prose, producing its style loss. Claim judgments and style
rankings are recorded separately; an expression win does not cancel a material
claim error.

Source overlap rises from 0.288 to 0.351 without exact copying. That may partly
support retention, but reference overlap is slightly higher at epoch 1 than
epoch 2, and neither overlap metric certifies meaning. These are mixed-author
academic paragraphs; generic expression preferences do not establish a
distinctive Fama/coauthor signature, human authorship or detector improvement.

## Review and execution checks

Per-case A/B/C labels were shuffled before the reviewer saw the new outputs or
loss trends. All 30 assessments and ten style rankings were saved before
unmasking. The reviewer had previously seen base outputs and knew execution had
completed; wording could reveal conditions. This is partly masked automated
assessment, not independent human review or a calibrated semantic benchmark.
Nonmaterial coverage and wording flags remain visible even on acceptable cases.

The eight development cases are the same project-exposed cases used previously,
held out from gradient updates by paper and author. No reserved test records
were read for model selection. The loss comparison is exact; the qualitative
counts are provisional judgments on a small, reused development set.

All 54 target-label audits and context budgets pass. All 46 gradient checks have
finite positive norms, and adapter weights changed. Each saved adapter contains
136 F32 language-only tensors / 2,228,224 parameters with the approved rank-8
configuration. Epoch adapter hashes differ. All final adapter tensors match
after disk reload, and the reload generation matches the saved epoch-2 probe.
Data/config/baseline/runtime hashes match the authorization and immutable packet.

Elapsed time was **770.9 seconds (12.85 minutes)**, with peak GPU allocation of
**6,567,964,160 bytes**. The log has no Python traceback. It contains a pip
dependency-resolver warning, as the completed baseline did; no unrelated package
change or rerun was made during review. The downloaded text artifacts and log
were scanned without exposing credential values; no credential patterns were
found. HF_TOKEN retrieval remains through Kaggle Secrets.

## Next decision

Preserve both adapters and this negative/mixed result. Do not extend training
just because loss is still falling, or pick epoch 2 solely by loss: its claim
count does not improve over epoch 1 and it introduces a new planning assertion.
The predeclared final gate passes the generic-style and copying/format checks
but fails the required claim improvement.

The next useful preparation is a **training-only coverage audit of the remaining
failure types**: temporal distinctions, quantifiers, complete numerical examples
and unsupported explanatory additions. Determine whether the existing 46 pairs
teach those distinctions before preparing a targeted fidelity intervention.
Development outputs may diagnose error categories, but must not become new
training targets; keep reserved tests sealed. Additional training requires a
separate bounded proposal and authorization. No new experiment is started here.

Raw artifacts remain private under
`research_runs/teacher_pair_reverse_pilot_v2/kaggle_pilot_v1/`; masked inputs,
assessments, provenance, hash receipts and per-output metrics are under
`research_runs/teacher_pair_reverse_pilot_v2/pilot_review_v1/`.
[pilot_results.json](pilot_results.json) contains public aggregate results.
CPU review code is `fama_style.research.teacher_pair_expanded_pilot_review`.
