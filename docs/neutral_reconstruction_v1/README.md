# Direct rewrite versus neutral reconstruction: 12-example pilot

Stage 1 responses have been received and [reviewed](STAGE1_REVIEW.md). Stage 2 now
has [16 reviewed outputs](STAGE2_RESULTS.md): reconstruction copies less and
changes claim order more, but sometimes realizes annotation guardrails as added
commentary. Four original cases remain held. The initial selection stays frozen.
The [three-case instruction check](REALIZATION_INSTRUCTION_CHECK.md) has
[reviewed results](INSTRUCTION_CHECK_RESULTS.md): the added commentary is absent
in all three new outputs; remaining local scope issues require brief repair.
Those two scope issues now have a [versioned semantic binding repair](SEMANTIC_BINDING_REPAIR.md).
The derived brief has not been tested in generation or training.

This pilot tests whether generating neutral prose from a
detailed content brief provides more useful stylistic contrast than rewriting
the human paragraph directly, without unacceptable loss of economic information.
It creates no training-ready data and launches no model, detector or GPU job.

The private package is `research_runs/neutral_reconstruction_v1/package_v1/`.
The initial package and [START_HERE](START_HERE.md) document the completed Stage 1
task. Use the current [Stage 2 instructions](STAGE2_START_HERE.md) for generation.
The [summary](preparation.json) records selection counts and paper IDs without
paragraph text. Templates and the output schema are public; source text and
response folders remain private.

## Selection and exposure

Select 12 paragraphs from the frozen 1,042-candidate curation release. Use two
papers per non-Fama group, two paragraphs per paper, with fixed SHA256 ranking
after source-only filters: 80–200 words, no recorded quality flags, a single PDF
page/reading region, no duplicate link and no explicit dangling-context pattern.
There is no selection by detector result, style score or generated output.
The sample is purposive and favors simpler native-text sources; it does not
represent all 39 papers, OCR difficulty or economics genres.

Every source is still an unverified extraction candidate. A Stage 1 source check
can block it. No replacement is made based on response quality. The six selected
papers receive explicit training-side method-development commitments in a new
versioned [paper-role file](paper_role_commitments.json). Future split builders
must respect these commitments in addition to historical inventory metadata.
Their old inventory metadata is preserved, but they are no
longer candidates for a future fresh paper-held-out evaluation. Existing Fama
and multi-author reserved sources are not read or repurposed.

## Stages and outputs

1. **Source check and content brief:** two batches of six, one paragraph per
   paper per batch. Record source uncertainty and all semantic constraints;
   do not ask for a rewrite, rhetorical taxonomy or target style plan yet.
2. **Brief assessment and transport:** check schema/coverage and compare the
   brief with the original for missing or added claims, changed qualifiers and
   unsupported relations. Structural validation alone does not establish
   semantic completeness. Unresolved cases remain blocked in both methods.
3. **Paired generation:** use the original paragraph for direct rewriting and
   only the checked brief for neutral reconstruction. Same requested model,
   audience, meaning constraints and effort; preserve actual UI provenance.
   At most 24 first outputs; missing/blocked cases stay in coverage denominators.
4. **Automated comparison:** compare output meaning against the original, using
   separate blinded assessments where feasible; measure copying/long shared
   spans, sentence grouping, claim order, readability and the same three-level
   structural specification in both methods. Meaning evaluation must check
   the original, not only the potentially incomplete brief. Report material
   claim changes separately from local expression shifts.

No GPU training is needed for this annotation-method pilot. The existing Gemma
baseline and previous training results remain separate. No rewritten paragraph
will be called human-authored merely because a detector labels it human.

## Handoff safeguards and limits

The implementation strictly allows brief fields and validates actor/claim edges.
It replaces ordered claim IDs with opaque stable IDs and reorders claims, while
preserving semantic relations and conditions. The writer must still receive
enough semantic information; a semantic dependency is not removed to force
stylistic difference. Essential terms and citations remain.

A check holds briefs containing a 12-token verbatim reference span. This is a
mechanical leakage screen, not a guarantee of wording independence; shorter
copied phrasing, semantic information and uncontrolled ChatGPT retrieval remain
possible. Structured metadata rejects original text, free paragraph plans and
author-style labels. Conversation separation does not establish independence.

The Stage 2 templates exist for review, but no real Stage 2 payloads are prepared
until Stage 1 responses have been checked. No unsupported semantic claims are
silently filled in. The source-check output is model opinion, not independent
transcription verification or certification of training readiness.
