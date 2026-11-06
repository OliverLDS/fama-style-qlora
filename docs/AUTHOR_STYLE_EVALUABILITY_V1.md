# Author-style evaluability and collection decision v1

Follow-up: the authorized [coverage and twenty-paper expansion](author_pair_expansion_v1/README.md)
is now prepared. The counts and collection recommendation below preserve the
earlier decision point; they are not the latest inventory totals.

This CPU audit separates papers, annotation selections, extraction candidates and
training pairs. It makes no new style-performance claim. No generation, annotation,
detector calls or GPU jobs were run. Existing releases and reserved texts remain intact.

## Coverage findings

| Target group | Collected papers | Current paired training coverage | Immediate unused coverage |
| --- | ---: | --- | --- |
| Fama/coauthors | 13 | 247 pairs from 7 papers in the original corpus | Four train-assigned papers contribute zero selected pairs; their exclusion must not be silently reversed. |
| Shiller | 13 | 5 admitted teacher pairs from 1 paper | 12 papers remain unassigned in the inventory. |
| Holmström/coauthors | 13 | 5 admitted teacher pairs from 1 paper | 12 papers remain unassigned in the inventory. |
| Baker–Wurgler/coauthors | 13 | 4 admitted teacher pairs from 1 paper | 12 papers remain unassigned in the inventory. |

The Fama corpus also has 21 development pairs from FF1997 and 45 reserved pairs
from FF2014. These roles are preserved. The three added groups each originally
contributed five annotation selections; selection is different from admission.
Teacher counts above were checked against the existing private
`data/teacher_pair_pilot_v2/training.private.jsonl`, not inferred from PDF counts.
That pooled release contains 46 training pairs across twelve papers overall,
including authors outside these four target groups. It is not an author-conditioned
training demonstration, and its counts must not be added to Fama's as one release.

The 39 non-Fama papers yield **1,042 extraction candidates**: Shiller 368,
Holmström 242, Baker–Wurgler 432. These have not thereby become meaning-aligned
or independently verified training pairs. Exact PDF-hash and normalized-title plus
exact-author-list checks find no duplicates in the registered inventory; these
checks cannot exclude all alternate drafts or overlapping papers.

## Exposure and holdouts

The seven contributing Fama training papers have already informed training and
style discovery. FF1997 is reused development material. Neither supplies a fresh
confirmatory evaluation. Original topic-matched controls and their replication
have also informed failed author-proxy tests; do not tune another classifier on
those same results and call its score independent validation.

The audit identifies five sealed papers from split metadata: the one Fama test
paper and four pooled-author reserved papers (65 reserved examples total).
It opens none of their paragraph bodies. All existing reservations remain sealed.
An unassigned paper is only a candidate for prospective evaluation: the inventory
does not establish that it has never appeared in an earlier prompt or conversation.
Before assigning new holdouts, check recorded project use and version aliases;
record unobservable external chat exposure as unknown. This is a bounded release
audit, not a complete reconstruction of every past model interaction.

## Collection recommendation

**Use twenty papers per group as a staged planning target; do not expand all four
to thirty immediately.** Twenty requires seven additions per group (28 overall);
thirty requires seventeen each (68 overall). These are collection budgets, not
statistically justified sample-size thresholds.

1. First assess a spread of the existing twelve unassigned papers per added group.
   Select across topics, publication periods and exact coauthor lists; avoid filling
   a group with closely related drafts. Build paired coverage over several papers
   rather than drawing more paragraphs from its current single paper.
2. For Fama, investigate why four existing papers contributed no accepted pairs.
   Preserve those decisions unless a separately audited repair supports admission.
   Targeted new papers may be preferable, especially for a prospective evaluation
   that is not another reuse of the seven familiar training papers.
3. Add papers toward twenty where existing sources leave topic, period, genre or
   evaluation gaps. Retain pre-2020 source versions where possible and exact author
   lists. A coauthor group is not evidence of an isolated individual's style.
4. Consider thirty only after the twenty-paper inventory shows remaining gaps or
   paper-level uncertainty warrants expansion. More PDFs alone cannot repair a
   weak writing-style metric or poor semantic alignment.

No downloads or annotation batches are started by this recommendation.

## Prospective comparison to freeze before execution

Start with ordinary rewriting versus one concrete name-free style specification;
include unchanged source and a generic academic rewrite as controls. Keep model,
decoding and inputs matched. Evaluate the same named author group across distinct
papers; this differs from the existing pooled unseen-author task.

For each group, provisionally seek at least three training papers and two distinct
development papers from the existing collection. This is an exploratory minimum,
not sufficient evidence of general author recognition. Choose paragraph counts,
topic strata, eligibility and all splits before generation. Record complete
paper/version and coauthor grouping; remove duplicate paragraphs and alternate
versions across splits. Preserve a new prospective confirmation pool only where
freshness can be supported; never repurpose existing reserved examples.

Freeze a small feature panel before fitting: sentence/clause organization,
qualification placement, explicit relation expression and stable economic labels.
Require executable definitions and source evidence; do not label generic formality
as Fama style. Compare regularized feature models with topic-vocabulary and
length/readability nuisance baselines. Use whole-paper validation, paper-level
label permutation, matched topic/genre strata and paper-level uncertainty intervals.
Do not split paragraphs from one paper across fitting and evaluation.

Measure meaning, copying, readability and style separately. Retain the existing
one-minor/no-material claim rubric; report uncertain judgments. A prospective style
proxy must improve over nuisance baselines on new papers and survive topic controls
before being used to select an adapter. Report paper-level effect sizes and
uncertainty, rather than treating paragraphs as independent observations. Failed or
ambiguous gates return to representation/data diagnosis, not a larger GPU sweep.

If these measurements are informative, prepare a separate matched pooled versus
author-conditioned SFT pilot. Structural mapping and DPO remain competing later
hypotheses, not established improvements. No training is authorized by this packet.

## Reproduction and limits

Run from the repository root, with private metadata present:

```sh
python3 src/fama_style/research/author_style_evaluability.py \
  --output research_runs/author_style_evaluability_v1/audit_v1/coverage.json
python3 -m unittest discover -s tests -p test_author_style_evaluability.py -v
```

The private output records four allowlisted metadata input hashes, per-paper
roles, duplicate checks and reserved-body access count. Repeating an unchanged
audit is deterministic; changed inputs require a new output version. The automated
audit reads metadata only; admitted teacher-pair counts in this report are a
separate bounded inspection of nonreserved training rows. No real paragraph,
raw response, PDF, credential or checkpoint is included in this public report.
