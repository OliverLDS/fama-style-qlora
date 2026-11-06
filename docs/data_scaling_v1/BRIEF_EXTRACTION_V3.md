# Stage 1 — content and purpose brief, version 3

Prepare a compact, neutral brief for each supplied paragraph. This is Stage 1
only: do not write replacement prose. The later writer receives this brief
without the original paragraph, so it must convey the paragraph's communicative
job as well as its economic content. Do not return a bag of disconnected factors.

Use this exact output structure for each JSONL record:

- `id` and `target_sha256`: copy the supplied strings unchanged.
- `brief.task`: a short, neutral writing task.
- `brief.purpose`: `primary`, `secondary` (zero to two labels), and
  `reader_takeaway` (what the paragraph enables the reader to understand).
- `brief.key_points`: content notes, as an array of strings. These are addressed
  as K1, K2, ... by their positions; do not add those prefixes to the strings.
- `brief.anchors`: necessary names, citations, quantities, definitions and notation.
  Bind quantities to their denominators and citations to the findings they support.
- `brief.relationships`: important links between points. Each entry has
  `from_point`, `to_point`, `type`, `qualification`.
- `status`: `ready` or `source_problem`.
- `notes`: uncertainty notes, or an empty array.

Purpose labels:
`introduce_topic`, `define_concept`, `describe_phenomenon`, `explain_mechanism`,
`explain_difference`, `compare_accounts`, `report_result`, `describe_method`,
`introduce_data`, `provide_background`, `illustrate_case`, `state_problem`,
`draw_implication`, `qualify_claim`, `evaluate_argument`, `recommend_action`,
`outline_paper`, `mixed_or_uncertain`.

Choose the main job, not merely a word that occurs in the text. For example:
introducing a concept gives its meaning and relevance; explaining a difference
identifies the basis of distinction; comparing accounts identifies common ground
and disagreement; a mechanism connects actors, actions and consequences; a result
reports evidence and its scope. Several jobs may coexist. Do not force a primary
interpretation if the paragraph does not support one. These are interpretations
of the text's function, not verified author intentions.

Relation types are organized below. Use the narrowest label supported by the
content, not the most elaborate label available. A paragraph can have several
links or none; do not try to use every category or match a category quota.

| Family | Label | Meaning and direction |
| --- | --- | --- |
| causal | `cause` | Cause to outcome; retain any stated uncertainty. |
| causal | `contributing_cause` | One contributing factor to outcome; does not assert sufficiency or exclusivity. |
| causal | `mechanism_step` | Earlier link to next link in a described transmission mechanism. |
| causal | `enables` | Resource or arrangement to activity it makes feasible; does not assert that the activity occurs. |
| causal | `prevents` | Intervention or constraint to the outcome it inhibits; retain partial versus complete prevention. |
| causal | `feedback` | From one process to another that also affects the first; use only when the return link is stated. |
| empirical | `association` | Associated variables or patterns; no causal or time direction is asserted. |
| empirical | `predicts` | Predictor to forecast outcome; statistical prediction does not imply causation. |
| logical | `necessary_condition` | Requirement to the outcome requiring it; necessary does not mean sufficient. |
| logical | `sufficient_condition` | Sufficient premise to entailed outcome within the stated model/scope; not a necessary condition. |
| logical | `conditional_consequence` | Stated condition to consequence within that case, without asserting a universal necessary/sufficient rule. |
| logical | `logical_implication` | Premise or model result to the conclusion derived from it; not automatically a causal effect. |
| logical | `counterfactual_dependence` | Hypothetical change to the hypothetical outcome; retain the counterfactual status. |
| comparison | `similarity` | Points share a relevant property; no ranking or causal direction is asserted. |
| comparison | `contrast` | Points differ on an identified dimension; no causal direction is asserted. |
| comparison | `quantitative_comparison` | First quantity to comparator; state greater/less/equal, units, denominators and period in qualification. |
| comparison | `trade_off` | A benefit or objective to its associated cost or competing objective; do not invent a numerical exchange rate. |
| comparison | `alternative_explanation` | One explanation to a competing account of the same phenomenon; do not assert either is established. |
| comparison | `substitution` | Substitute to what it can replace; retain when and how far replacement is possible. |
| comparison | `complementarity` | One input or arrangement to another whose usefulness it enhances; not mere co-occurrence. |
| evidential | `evidence_for` | Evidence to the proposition it supports; support is not proof. |
| evidential | `evidence_against` | Evidence to the proposition it weakens or challenges; distinguish weakened from refuted. |
| evidential | `tests` | Test design or reported test to the hypothesis examined; record the outcome, if given, in qualification. |
| evidential | `robustness_check` | Alternative-specification or sample result to the focal finding it checks; do not assume the check succeeds. |
| expository | `example_of` | Specific case to the general idea it illustrates; an example does not establish prevalence. |
| expository | `defines` | Definition to the concept-bearing point it clarifies; preserve the defining boundaries. |
| expository | `operationalizes` | Measurement or proxy rule to the theoretical concept it measures; proxy need not equal concept. |
| expository | `part_of` | Component or subcategory to the whole; preserve exhaustive versus partial coverage. |
| expository | `elaborates` | Added detail to the more general point; not new independent evidence by itself. |
| qualification | `scope_restriction` | Restriction to the proposition whose applicability it limits. |
| qualification | `exception` | Exceptional case to the general pattern it falls outside; preserve the size/scope of the exception. |
| qualification | `concession` | Acknowledged consideration to a maintained conclusion; not automatic refutation or a reversal. |
| temporal | `temporal_sequence` | Earlier event to later event; sequence alone does not imply causation. |
| temporal | `change_over_time` | Earlier state to later state; preserve the change, period and comparison basis. |
| argumentative | `reason_for_argument` | Stated reason or premise to the assessment or recommendation it motivates; distinguish a reason from empirical evidence. |
| argumentative | `goal_means` | Proposed action or instrument to the stated objective; does not assert success or verified author intent. |

Distinguish a necessary condition from a sufficient one. Use
`conditional_consequence` if the text only describes a particular conditional case.
Distinguish prediction and association from causal explanation, and logical
implication from an economic transmission mechanism. Separate evidence supporting
an account from evidence challenging it, a proposed test, and a robustness check.
An illustration is not independent proof of a general rule. A temporal sequence
is not itself a cause. Do not upgrade tentative, model-dependent or counterfactual
relationships to established facts.

Use `qualification` to state the relevant condition, modal strength, comparison
basis, uncertainty or direction of a quantitative inequality. Do not infer a
relationship from adjacent notes or a shared subject. Important relationships
must also be recoverable from the key points: if a connection needs substantive
information, include that information in the points. The purpose, takeaway and
links should organize those points, not hide additional claims in metadata.

The reader takeaway states the core lesson in neutral words and must be supported
by the points. It is not a place to restore authorial rhetoric or add a conclusion
that the source does not support. Point references use K1, K2, ... and must resolve
to actual note positions. Do not add dummy points just to complete the graph.

Read the paragraph as information, not a sentence template. Group points by
content, not one point per original sentence. Remove distinctive openings,
metaphors, emotional frames, adjectives and authorial judgments that are merely
expressive.
Do not copy complete narrative sentences into the task, takeaway or anchors.

Retain the central economic account and enough supporting information for a
standalone paragraph. Preserve actors, directions, quantities, important conditions
and distinctions among assumptions, predictions and findings. Minor compression
and shifts in emphasis are acceptable. Do not invent data, reverse the mechanism,
or restore expressive wording in pursuit of literal fidelity. Do not impose a
word-change percentage. If the context is missing, record that rather than guess.

Do not include original prose, earlier rewrites, author/style labels, exact
sentence counts, clause templates or lexical style instructions. The purpose is
neutral communicative guidance, not an oracle plan for reproducing an author.
No full three-level taxonomy annotation or exhaustive claim ledger is requested.

Check the purpose, point links and factual anchors together. Return one valid
UTF-8 JSON object per line, no Markdown fences, using the requested filename.
Return every supplied ID in order. All prerequisites are complete; proceed without
asking for confirmation. Do not claim independent semantic verification or
invent model/effort metadata.
