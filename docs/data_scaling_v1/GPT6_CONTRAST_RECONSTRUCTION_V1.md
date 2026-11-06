# GPT-6 500-pair contrast audit and brief-first regeneration

Update: all six Stage 1 response files have returned. The [brief review](GPT6_CONTRAST_BRIEFS_V1.md)
retains all 175 items and prepares six actual brief-only Stage 2 attachments.
Proceed to those attachments; do not regenerate the completed Stage 1 batches.

The [Stage 2 output review](GPT6_CONTRAST_OUTPUTS_V1.md) is also complete: 173
fresh replacements are staged, including ten locally repaired paragraphs, and
two inherited source concerns remain held. Do not repeat the completed Stage 2 jobs.

The user's review found that semantically faithful inputs can still be weak
style-training examples. This audit therefore evaluates **all 500 current GPT-6
training pairs for copying and re-expression**, separately from the earlier
meaning/format audit. Existing sources, human targets, experiments and hashes stay
unchanged. No data is deleted or automatically replaced, and no GPU/API job runs.

| Disposition | Pairs |
|---|---:|
| Retain current input | 320 |
| Generate a stronger input through a neutral brief | 175 |
| Resolve inherited target problem first | 5 |
| Total screened | 500 |

The replacement queue contains 19 Fama/coauthor, 53 Shiller, 45 Holmström and 58
Baker–Wurgler/coauthor paragraphs. **153 are historical reuses; 22 are recent
manual GPT-6 counterparts.** Historical workflows and prompts differ, so this is
not evidence that model capability changed or that one teacher is universally
better. The 175 outputs will be replacement versions of existing target pairs,
not 175 new independent training examples.

## What the audit measures

Every pair has text-bound metrics for word-sequence similarity, retained ordinary
wording, shared five-word phrase coverage, longest unchanged spans, repeated word
inventory, and sentence-count/alignment patterns. Obvious numeric tokens, formulas,
acronyms, names and a finite list of technical terms are exempted from ordinary-word
diagnostics. That exemption list is incomplete; it is not a semantic parser.

High sequence similarity, extensive copied prose or combined wording/layout echoes
identify replacement candidates. A follow-up checks retained clauses even when
sentences have been split: changing punctuation can reduce a simple similarity
score without yielding a fresh paragraph. Reordering much of the same word inventory
is also inspected. Fourteen text-assessed follow-ups supplement the numerical
screen; reasons are stored per ID. Intermediate numerical screens and builder
snapshots are preserved privately as preliminary versions, not upload instructions.

This is automated contrast triage across all 500, with text follow-ups, **not 500
independent human style ratings**. The thresholds are heuristic. Neither low overlap
nor a changed sentence count proves good writing, claim consistency, or distinctive
Fama style. Retain-current means no prioritized regeneration under this audit,
not a style-quality certificate. No development/test text is used in the audit.

The user's anchors are consistent with these decisions: P001 remains; P024 and
P002 enter regeneration. The user judged both P002 and G002 insufficient, despite
an initial relative preference for G002. This unblinded feedback is retained as
qualitative evidence, without inferring a global teacher ranking. The word *hope*
alone is not a copying test; the inherited opening and rhetorical framing are the
more consequential issue.

## Correcting the generation procedure

Earlier instructions allowed unchanged wording where the original was clear and
placed disproportionate emphasis on exhaustive fidelity. That encouraged close
paraphrases. The new procedure physically separates two generation requests:

1. **Original paragraph → neutral content brief.** Extract compact economic notes,
   remove authorial voice and sentence templates, and group by content rather than
   writing one bullet for each sentence. Retain central explanations and necessary
   factual/technical anchors. Do not generate a replacement paragraph at this stage.
2. **Brief only → newly composed neutral paragraph.** The attachment contains the
   brief, ID and hash strings. It contains neither the original paragraph nor the
   previous rewrite. The model constructs prose from notes rather than editing a
   supplied paragraph sentence by sentence.

Substantial re-expression and construction are the priority. Modest changes in
emphasis or nuance are acceptable. Actual invented data, reversed economic
relationships and other major factual problems remain visible. Brief validation
checks shape, identity and actual source problems; it does not insist on verbatim
quantifiers, restore stylistic adjectives or impose a brittle lexical-change quota.
After reconstruction, contrast and material errors will be assessed separately.
Semantic repairs must not quietly restore the original's expressive wording.

The Stage 2 request should be separate, preferably in a new chat without original
attachments. This controls what the package explicitly supplies, while making no
claim of isolation from ChatGPT's cross-chat memory. It is not possible to guarantee
that no common word shared with the source will appear.

## Manual work

Use the final private folder:
`research_runs/data_scaling_v1/gpt6_track_0500_v1/contrast_rewrite_v1/`.
Start with its `START_HERE.md`. Do not use the preliminary screen folders.

- **Stage 1: six uploads**, 30/30/30/30/30/25 records. Each
  `stage_1_brief_batches/batch_NN.md` is self-contained. Select GPT-6 and request
  High thinking effort. Save `briefs_NN.jsonl` in `responses/stage_1_briefs/`.
- **Stage 2: six further uploads**, prepared from those actual brief responses.
  Tell Codex when Stage 1 is ready, or run
  `python3 scripts/data/prepare_gpt6_contrast_rewrites.py prepare-reconstruction`.
  Save the resulting `rewrites_NN.jsonl` in `responses/stage_2_rewrites/`.

Thus the planned workload is **twelve uploads across two rounds**, not six requests
that pretend to hide the original within the same prompt. P002 and P024 appear
in the first job. Each selected target is intended for completion; this is not a
request to annotate many records and use only a tiny best subset. No taxonomy,
CSV entry, detector scans or human QA form is requested. Genuine source problems
and weak reconstructions remain explicit for targeted repair rather than silent
discarding.

The five deferred targets are P250, P264, P291, P458 and P484, documented in the
[earlier quality audit](GPT6_TRACK_0500_QUALITY_V1.md). They should be repaired or
replaced consistently across the matched teacher tracks before final release.

## Evaluation and reproducibility

GPT-6 remains a training teacher. **Gemini-generated, paper-held-out inputs can
evaluate the GPT-6-trained student**, including Pangram input/output comparisons
where those inputs have detector headroom. That is a cross-teacher evaluation;
report it separately from same-teacher inputs. Do not treat copying an input
already labeled human as an AI-reduction gain. No paid API use or routine repeat
scans of published human references are requested.

The new 500-input version will be a data-quality/prompt treatment, not a larger
dataset. Preserve the earlier version as its baseline; any training comparison
needs matched training/decoding settings and explicit authorization.

`scripts/data/prepare_gpt6_contrast_rewrites.py` reproduces preparation and checks
hashes. It refuses to overwrite existing packets. CPU tests cover copied phrasing,
technical exemptions, clause-reordering blind spots, immutable source bindings,
unique job coverage, deterministic rebuilding, and a synthetic full Stage 2 build
that forwards only the supplied briefs. Schemas and all input prompts are saved;
actual model/effort settings are not independently certified by those prompts.
