# Stage 1 — content and purpose brief, version 2

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

Relation types:
`cause`, `association`, `contrast`, `condition`, `temporal_sequence`,
`evidence_for`, `definition_part`, `qualification`, `alternative`.

Use only relationships supported by the supplied content. Preserve direction:
cause to consequence, condition to conditioned point, evidence to supported point,
qualifier to qualified point, or earlier to later. For contrast or alternatives,
the arrow has no causal meaning. An empty list is appropriate when no link is
needed. A shared subject or adjacent sentences do not establish a mechanism.
Use `qualification` for uncertainty, applicability or scope that matters to the link.

Read the paragraph as information, not a sentence template. Group points by
content, not one point per original sentence. Remove distinctive openings,
metaphors, emotional frames, adjectives and authorial judgments that are merely
expressive. Do not retain a word such as “hope” solely because the author used it.
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
