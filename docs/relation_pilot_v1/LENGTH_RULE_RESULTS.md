# Fixed 30-word rule: modest additional-paper gain

All eight verification files cover **32/32 candidates** and pass schema,
fixed-text, quote-occurrence and internal-consistency checks. The sources,
candidates, annotations and null raw model/effort fields are preserved exactly.
The user confirms GPT-6 with High requested; effective effort remains unconfirmed.
No cutoff was refitted and no classifier was fitted in this replication.

## Verification judgments

| Fidelity | Merge reads better | Separation reads better | Comparable | Total |
|---|---:|---:|---:|---:|
| Faithful | 7 | 17 | 7 | 31 |
| Local shift | 0 | 1 | 0 | 1 |
| Total | 7 | 18 | 7 | 32 |

These are GPT-6 second-assessment judgments, not independent semantic
certificates. The one local shift changes a direct valuation question into an
embedded question; the verifier also prefers separation. All 32 candidates are
judged single sentences. No material-shift or unresolved verdict is returned.
The favorable fidelity count is specific to these candidates and this judge;
differences from earlier experiments cannot establish a causal improvement in
generation because papers, samples and tasks differ.

## Locked policy comparison

The primary score uses the **24 faithful, decisive cases**: seven MERGE and
17 KEEP. Seven faithful ties and the one shift remain separate. All 32 assigned
items produced candidates, so there are no availability exclusions or generator
abstentions. Candidate text is unavailable to the fixed source-length rule;
faithfulness is known only for retrospective scoring and side reporting.

| Policy | Correct / 24 | Accuracy | Mean paper accuracy | Beneficial merges selected | Worse merges selected |
|---|---:|---:|---:|---:|---:|
| KEEP | 17 | 70.8% | 70.7% | 0 | 0 |
| MERGE all available | 7 | 29.2% | 29.3% | 7 | 17 |
| Fixed source pair ≤30 | **19** | **79.2%** | **79.3%** | **6** | **4** |

The rule's primary gain over KEEP is **two decisions**, or 8.3 percentage points.
It selects six of seven judged-beneficial merges, misses one and makes four
primary false-positive selections. Primary precision is 6/10 (60.0%), recall
6/7 (85.7%) and F1 0.706. This precision denominator excludes side cases; it must
not be described as fidelity or usefulness of every selected edit.

Across **all 16 selected candidates**, the judgments are six faithful/readability
improvements, four faithful/readability losses, five faithful ties and one local
shift/readability loss. The shifted case is reported outside primary accuracy,
but its selection remains visible. Ties are neither silently counted as failures
nor discarded from the record. No example is automatically approved for training.

| Held-out-from-cutoff paper | Primary cases | KEEP correct | Fixed-rule correct |
|---|---:|---:|---:|
| Holmstrom 1979 | 5 | 4 | 4 |
| Pastor–Stambaugh 2011 draft | 7 | 3 | 7 |
| Boudoukh–Israel–Richardson 2019 | 7 | 7 | 4 |
| Shiller 1998 draft | 5 | 3 | 4 |

The gain is heterogeneous: +4 decisions on Pastor–Stambaugh, −3 on Boudoukh et
al., +1 on Shiller and no change on Holmstrom. These papers were excluded from
the four-paper cutoff experiment but had appeared in earlier project work.
This is fixed-rule replication on additional papers, not a pristine unseen test.

## Length strata and failure modes

| Source-length stratum | Faithful/merge better | Faithful/separation better | Faithful tie | Local shift |
|---|---:|---:|---:|---:|
| At most 30 | 6 | 4 | 5 | 1 |
| Over 30 | 1 | 13 | 2 | 0 |

The rule screens out many long-pair readability losses but also misses a useful
35-word merge. Shortness alone does not establish useful organization. Among
short selected pairs, the verifier favors separation when a merge buries a
standing assumption in a roadmap, weakens the prominence of a warning, or joins
a specific robustness check to a broader claim. A compact source can also yield
a substantially expanded candidate when pronouns are resolved.

The legacy word counter excludes numerals. One short-count pair contains several
numerical alternatives and carries a speech-act shift/readability loss. This is
a recorded limitation, not a reason to change the counter after seeing outcomes.
Original lexical counting and the cutoff stay frozen. Pairwise success does not
authorize chaining several adjacent merges into a long paragraph sentence.

## Interpretation and next decision

Retain the rule as an **optional candidate filter and research baseline**, with
ordinary rewriting still the default. Its modest gain and visible bad selections
do not support automatic application. Keep fidelity/readability checks separate;
allowing the observed local shift adds no improvement in this sample.

This local-organization study can now be frozen as a mixed finding. Further
fine-tuning decisions need evidence about the actual author-style objective,
beyond sentence grouping. This result does not establish Fama/coauthor-style
improvement, human authorship, detector evasion or that a trained rhetorical
mapper will generalize. No new Gemma training is justified by this result alone.

The sample deliberately contains four short and four long pairs per paper and
therefore does not represent natural boundary prevalence. Paragraphs and pairs
repeat; four papers and same-model judgments do not support an independence-based
significance claim. Original source/context cautions remain unresolved even
when local comparison is possible. Threshold hiding and fresh verification
payloads do not establish independence from pretraining or automatic chat retrieval.
The [locked protocol](length_replication_protocol.json) governs the comparison;
no post-result threshold search or classifier fit was performed.

## Artifacts and checks

Private snapshot and per-item policy decisions:
`research_runs/length_rule_replication_v1/verification_results_v1/`.
Raw responses, coverage reports, settings, checks and original protocol are
hashed. The [public summary](length_replication_results.json) contains aggregates
only. Source text, candidates and unit evidence remain Git-ignored. Earlier
experiments, labels, hashes and reserved splits are unchanged.

CPU entry point: `fama_style.research.length_rule_verification_review`, with
`--initial`, `--generation-audit`, `--packet`, `--session` and a fresh `--output`.
It checks the full packet/hash chain, actual candidate transport, coverage and
evidence, recomputes the frozen word count/rule and scores only after mechanical
checks pass. It never infers claim completeness from exact quotes alone.
Tests cover the exact 30/31 boundary, unavailable-candidate KEEP behavior,
separate ties/shifts, frozen-count tampering, raw preservation, changed-candidate
scoring blocks, ID coverage and the absence of training approval. No reserved
examples, new model calls, classifier fits, GPU jobs or neural/adapter training
were used by this audit and evaluation.
