# Structure above the paragraph: proposed issue/argument units

## Current coverage

The implemented taxonomy describes paragraph, sentence/clause and language
features. It does **not** annotate how multiple paragraphs develop a single issue.
Section/page metadata is provenance, not a discourse-structure annotation. No
empirical higher-level Fama/coauthor-style model has been constructed.

## Proposed hierarchy

Document → section/subsection → **issue or argument unit** → paragraph →
sentence/clause. Language features can be measured at several scopes rather than
constituting another text-container level.

An issue unit is a passage that develops a local question, claim or problem across
one or more paragraphs. It need not equal a heading or a fixed paragraph count.
Start with contiguous spans; link later returns to the same issue rather than
silently combining distant passages. Units can be nested when the text supports it.

An illustrative pattern is question/problem → assumptions or model → evidence →
qualification → implication. Its occurrence and frequency in Fama/coauthor papers
must be measured; this is not a verified author template.

## Representation to investigate

For each proposed unit, retain:

- Stable paper/unit IDs and ordered source paragraph IDs, with page/section
  provenance and reading-order/boundary uncertainty.
- The issue/question and its exact evidence; label inferred formulations as
  interpretations rather than stated author intent.
- Paragraph-level functions and their order: framing, claim, definition,
  assumption, analysis, evidence, qualification, implication, transition, etc.
  These are provisional candidate labels, not a frozen inventory.
- Links between paragraphs: elaboration, support, contrast, condition,
  qualification, consequence, and referential continuity, with exact endpoint
  claims and evidence. Adjacency does not establish causality or support.
- Shared assumptions and qualifications whose scope crosses paragraph boundaries.
- Entry/exit and whether the issue is resolved, left open, or handed to another
  unit. Preserve uncertainty and links to outside antecedents.

Keep presentation structure separate from the semantic graph. Reordering paragraphs
must not change an assumption's scope, detach evidence from its claim, resolve an
open question, or turn tentative results into conclusions.

## Data implications

The curated training rows are selected paragraphs. Neighboring rows in a shuffled
batch, or even rows sharing a section label, are not necessarily adjacent in the
paper. The current two-paragraph repeat check cannot answer the issue-level question.

A later pilot should recover contiguous source passages from **training papers**,
using archived paragraph IDs, reading order and page provenance, with extraction
uncertainty retained. It should include the intervening paragraphs rather than
group only the already selected examples. Check paragraph continuations and layout
against available PDF/text evidence; do not claim independent verification from
an LLM's self-check. Development and reserved-test papers must stay outside discovery.

For paired style learning, create or verify a meaning-preserving neutral version
of the WHOLE issue unit. Independent paragraph rewrites do not establish coherent
cross-paragraph claim, reference or qualification preservation. Extract both unit
structures using the same specification and claim identities, then test an oracle
issue plan before learning a transformation model.

This layer could study argument organization that a paragraph-only experiment
cannot measure. Its benefits, annotation repeatability and realization costs are
open research questions. This document is a design proposal; no source passages
were reconstructed, annotated or evaluated for it.
