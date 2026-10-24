# Sentence-slot realization: version 1 results

Reviewed `oliverlds/sentence_slot_realization`, version 1. All twelve first-attempt
generations and independent NLI assessments completed on the four frozen training
pairs. There was no training, adapter, repair attempt or held-out data use.
Runner time was 274.1 seconds (4.57 minutes), including 163.9 seconds of generation
and 14.8 seconds of NLI. Queue and notebook installation time are excluded.

## Primary results

| Measurement | A: ordinary | B: count only | C: JSON sentence slots |
|---|---:|---:|---:|
| Strictly valid format | 4/4 | 4/4 | 0/4 |
| Valid format and correct sentence count | 2/4 | 3/4 | 0/4 |
| Mean count error among valid outputs | 0.50 | 0.25 | Undefined |
| NLI proxy passes among valid outputs | 2/4 | 2/4 | Undefined |
| Format + count + NLI proxy pass | 2/4 | 1/4 | 0/4 |
| Near copies among valid outputs | 0/4 | 0/4 | Undefined |
| Mean style-gain proxy among valid outputs | -0.584 | -0.601 | Undefined |

All C outputs have Markdown code fences despite the instruction prohibiting them.
The strict validator therefore rejects all four. The saved aggregate contains zero
counts for some valid-only C metrics, but their valid denominator is zero: these
are undefined rates, not evidence of semantic failure or absence of copying.
Higher style gain is better; the measure is distance change toward the frozen
reference feature profile, not validated author-style quality.

## Explicitly post-hoc format diagnosis

For diagnosis only, removing exactly one outer JSON code fence permits parsing
all four payloads. Raw outputs and primary scores remain unchanged. This is not
a predeclared success criterion or a model repair run.

| Training paper key | Requested | A sentences | B sentences | C array slots inside fences |
|---|---:|---:|---:|---:|
| F1965 | 3 | 4 | 3 | 3 |
| FF2000 | 2 | 2 | 2 | 2 |
| NL2003 | 2 | 2 | 2 | 1 |
| VL2018 | 6 | 5 | 5 | 6 |

Three C payloads pass the existing slot validator after fence removal; NL2003
still has one slot instead of two. The six-sentence example no longer collapses
into a single semicolon-linked sentence. This is useful evidence of partial
count controllability, not evidence that the structural approach has failed
entirely. Boundary checks remain surface heuristics.

No NLI or style scores were recomputed on the normalized payloads. Saved C NLI
scores used the raw malformed output, including JSON and fences, and cannot be
presented as verified semantic scores for normalized paragraphs.

## The substantive bottleneck

Formatting is not the only failure. Direct inspection of the F1965 C response
shows the source's upper-bound condition on dependence rewritten as a lower-bound
condition on intertemporal correlation. This reverses the eligibility condition
relative to the supplied input. It also adds explanatory material not supplied
by the input. The NLI checker flags this example; satisfying the requested three
slots would not fix its meaning.

Count-only guidance improves count error, but its style proxy is essentially
unchanged and its combined count/semantic pass count falls from two to one.
For B, F1965 matches the count but is NLI-flagged; FF2000 matches and passes;
NL2003 matches but becomes flagged; VL2018 passes NLI but misses the count.
Thus the aggregate two NLI passes hide a change in which examples pass.

All three equivalent NLI controls passed and all seven semantic mutations were
rejected. This small sanity check does not establish domain accuracy. Three of
the four unchanged source/reference pairs remain NLI-flagged. Those flags are
not confirmed label errors, and no pair was dropped after observing outputs.

## Verification

The input pack identity matches the approved SHA-256
`83d834c37bfb5233c2a3dbf8b620ff522e8b467c9f0e1440f57c676257b9e0ee`.
All downloaded manifest hashes and embedded runtime sources match. Jobs match
the local frozen pack exactly. Re-evaluating saved outputs using saved NLI
probabilities reproduces both metrics and summary exactly. All twelve outputs
are nonempty, non-synthetic and below the token limit. The log contains neither
a Python traceback nor CUDA out-of-memory error.

Private artifacts and a separate post-hoc fence diagnosis are retained under
`research_runs/sentence_slot_v1/version_1_review/`. Copyrighted paragraph text is
not reproduced in this report. Earlier experiments remain intact.

## Recommendation

Do not train a rhetorical mapper or another QLoRA adapter yet. The count-control
mechanism is partly actionable, but faithful realization remains unresolved.

Prioritize a small claim-constrained inference check over another formatting-only
run: explicitly protect comparison direction, quantifiers, modality and scope,
then test whether the model can reorganize sentences without inventing content.
Use the same semantic constraints for matched conditions. A future JSON protocol
can predeclare deterministic outer-fence normalization, while reporting raw and
normalized format compliance separately. Normalization must never conceal wrong
slot counts, change content or be counted as semantic verification.

These are four previously inspected training examples, not a generalization
estimate. No new GPU job was submitted during this review.
