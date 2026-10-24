# Independently annotate new paragraphs with a proposed canonical vocabulary

Use `paragraphs.jsonl`, `taxonomy_categories.json` and
`canonical_annotation_output.schema.json`. Read all attachments; if one is
unavailable, report the missing filename and stop. Do not substitute a taxonomy.

Describe each paragraph AS WRITTEN, independently. Do not rewrite it, improve its
style, guess an author, infer an intended source/target class or consult a
counterpart paragraph. No supplied paragraph is guaranteed to be an ideal style
example or a faithful paraphrase of another text.

## Annotation specification

First extract semantic content separately: atomic claims where possible,
actors/entities and their roles, predicates, quantities, conditions, uncertainty,
negation, quantifiers, comparisons, attribution and relationships. Retain exact
evidence and operator scopes. Conditions do not assert that their antecedents
actually occurred. Distinguish hypotheses, expectations and findings. Mark
implicit/uncertain links rather than upgrading them to verified causality.

Then annotate three levels using the supplied CANONICAL IDs:

1. Paragraph: main-claim position, ordered moves, progression and qualifications.
2. Sentence: actual sentence/clause organization, dependency/coordination, subject
   selection, voice, proposition grouping and qualification placement.
3. Language: the applicable vocabulary's lexical, grammatical and stylistic
   features, with semantic-sensitive features protected.

Use only categories of the correct level and dimension. Choose according to
definition and evidence, not the label name alone. Do not force every category
to appear, equate density with quality, or erase meaningful distinctions to fit
an existing category. Protected/descriptive categories remain legitimate
annotations; presentation-control status does not make a category preferable.

In each label object:

- `assigned`: supply one canonical ID, empty candidate list, and a reason.
- `unresolved`: canonical ID is null; list plausible IDs, if any, and explain
  ambiguity. Do not choose arbitrarily.
- `not_covered`: canonical ID is null; the current vocabulary lacks an adequate
  category. Describe the gap and add evidence under `coverage_gaps`; do not invent
  a canonical ID or reuse a category from the wrong dimension.

Record independently identifiable components in separate feature/move entries,
each with its evidence and claim bindings, rather than making one ambiguous
multi-label assignment. Preserve uncertainty even if the catalogue has a precise
definition. Every quote must be an exact substring of the supplied paragraph,
with one-based occurrence. Sentence entries must cover the full text in order.
Clause IDs must be unique within the paragraph, and dependency refs must resolve.

Copy taxonomy version/hash fields from `taxonomy_categories.json`. Record only a
model version you actually know, otherwise null. Set context to `single_paragraph`.
Do not consult previous annotations or other conversation outputs. Do not require
human review; flag uncertain cases for later automated assessment.

## Delivery

Return downloadable `canonical_annotations.jsonl` following the supplied schema,
one object per input paragraph, and `coverage_report.md`. Report completed/missing
IDs, unresolved assignments, uncovered distinctions, uncertain claim bindings and
limitations. A self-check is not independent verification. If incomplete, deliver
completed rows and explicitly list the missing IDs.
