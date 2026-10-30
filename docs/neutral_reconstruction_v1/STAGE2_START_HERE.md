# Stage 2 — paired direct rewriting and neutral reconstruction

Stage 1 review approved eight cases provisionally and held four. The fixed
12-case selection and all holds remain in the audit. There are no replacements.
This is a generation-method diagnostic, not training or a result about style gains.

Run four jobs with GPT-6 and High thinking effort if available, matching the
actual settings across the two methods where possible:

| Job | Prompt | Input | Expected rows | Save response |
|---|---|---|---:|---|
| Direct 01 | `prompts/DIRECT_REWRITE_PROMPT.md` | `inputs/direct/inputs_01.jsonl` | 5 | `responses/direct/direct_01.jsonl` |
| Direct 02 | Same direct prompt | `inputs/direct/inputs_02.jsonl` | 3 | `responses/direct/direct_02.jsonl` |
| Reconstruction 01 | `prompts/RECONSTRUCTION_PROMPT.md` | `inputs/reconstruction/inputs_01.jsonl` | 5 | `responses/reconstruction/reconstruction_01.jsonl` |
| Reconstruction 02 | Same reconstruction prompt | `inputs/reconstruction/inputs_02.jsonl` | 3 | `responses/reconstruction/reconstruction_02.jsonl` |

For each job, supply its prompt, `prompts/generation_output.schema.json`, and
only its input file. Keep the first downloaded responses unchanged. Include
blocked rows rather than dropping them. Do not retry until an output looks
better or passes a detector. Tell Codex when all four files are ready.

Reconstruction jobs intentionally receive only semantic content briefs. Do not
upload original paragraphs, source PDFs, Stage 1 responses, direct outputs or
the `private_key` folder to those jobs. Separate conversations are a practical
way to avoid direct carryover; cross-chat retrieval remains uncontrolled and
is not claimed absent. Record actual model, visible effort and arrangement in
`session_notes.json` or report them to Codex; unknown is acceptable.

Three references have derived versions that remove four visually confirmed
superscript footnote markers. Both methods and later fidelity checks use those
same derived references. The original source release, Stage 1 inputs and raw
responses remain unchanged. Claim/qualifier corrections and uncertainty
resolutions are recorded in the private review audit.

The four held cases involve a missing comparative endpoint, an unbound effect
reference, an ambiguous historical-episode reference and a source item that
merged two PDF paragraphs. They do not have
generation jobs. No new source text or economic facts were invented to fill them.

The planned output is 16 paragraphs (eight per method), subject to any further
generation-time blocks. Evaluation must compare both methods against the same
reviewed reference, retain all original 12 cases in coverage, and report group
imbalance. Automated brief review is not independent semantic gold verification.
