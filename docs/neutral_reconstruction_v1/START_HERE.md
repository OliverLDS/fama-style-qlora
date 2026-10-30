# Start with Stage 1 only

This package selects 12 paragraph candidates from six papers: two paragraphs
per paper and four per author group. Each of the two batches contains one
paragraph per paper. The six selected papers are now committed to training-side
method development and cannot later be described as fresh held-out papers.
Existing corpus assignments remain unchanged; other new papers are unassigned.

For each batch, give ChatGPT:

1. `prompts/STAGE1_PROMPT.md`.
2. `prompts/stage1_output.schema.json`.
3. One input file: `stage1/inputs_01.jsonl` or `stage1/inputs_02.jsonl`.

Request GPT-6 with High thinking effort if available. Record the actual visible
setting and conversation arrangement in `session_notes.json`, or tell Codex when
returning the responses. Unknown/changed effort is recorded as such. Separate
chats can reduce direct carryover but cannot certify isolation from memory or
retrieval. No change to account settings or credentials is required.

Save the first downloaded responses as:

- `responses/stage1_briefs/briefs_01.jsonl`
- `responses/stage1_briefs/briefs_02.jsonl`

Any separate coverage/limitation reports can go in that same folder. Preserve
responses unchanged; use a new filename for corrections. Do not upload the
`private_key` folder. Do not run the Stage 2 templates yet.

Once both responses are ready, tell Codex. Codex will check IDs, schema,
source hashes, coverage and source/brief meaning alignment, then construct the
paired generation payloads. Cases with unresolved source/brief issues stay
blocked in both conditions, without replacements or dropped denominators.
There is no requirement for a human review panel.

For neutral reconstruction, only validated content briefs will be intentionally
supplied to the writer. Claims receive opaque IDs and are reordered without
changing supported semantic edges. No reference text, original sentences,
taxonomy, author label, paragraph outline or direct rewrite is included. Required
economic terminology, names and citations can still disclose source information;
the transport filter is not proof of no leakage or UI retrieval.

The eventual comparison has up to 24 first outputs: 12 direct rewrites and
12 neutral reconstructions. Automated fidelity/readability/structural evaluation
will follow generation; no results exist yet. Detector flags do not select the
examples, approve briefs or determine successful source construction.
