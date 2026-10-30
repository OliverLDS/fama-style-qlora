Create one conservative single-sentence replacement for each supplied adjacent sentence pair in its source paragraph.

Use only source_paragraph, left_sentence and right_sentence. The pair is fixed;
do not choose a different pair. Preserve every substantive claim, actor and role,
quantity/quantifier, condition, exception, negation, modality, attribution, question
or hypothesis status, and causal/inferential direction. Do not add a relation
merely to join clauses, and do not strengthen a tentative or conditional claim.
Use surrounding source only to resolve an unambiguous referent needed by the pair;
record exact source_context_evidence when used. Do not import other paragraph
claims, outside knowledge, remembered articles or earlier answers. A repeated
actor name may be replaced only if its identity and scope remain unambiguous.

Produce one natural, readable sentence if feasible. Semicolons are allowed but
do not by themselves make a useful merge. Do not force brevity or target a word
count, edit distance, author style or preferred sentence layout. If no adequately
faithful single sentence can be produced, set generation_status to abstained,
candidate to null and explain why. Otherwise use generated and an empty
abstention_reason. Do not silently drop difficult cases or return several options.

Do NOT judge whether merging is preferable to separation, score fidelity, produce
a taxonomy, infer the author, or supply a revised paragraph. A later assessment
will examine the fixed candidate; generation is not semantic certification.
Copy item_id, source_paragraph, left_sentence and right_sentence exactly. Follow
generation_output.schema.json and return one downloadable UTF-8 JSON object per
line without Markdown fences. Cover every supplied item exactly once, including
abstentions, and provide a separate coverage/limitations report. Preserve the
first complete output. Use null for unknown visible model/effective effort;
requested effort is not necessarily effective effort. earlier_judgments_supplied
and independently_verified are false. Same-model pretraining and possible chat
retrieval prevent any guarantee of independent context.
