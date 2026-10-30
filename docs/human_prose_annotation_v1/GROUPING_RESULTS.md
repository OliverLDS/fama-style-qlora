# Explicit grouping works on the five nonidentity development cases

All 15 GPT-6 responses were received with the user-reported High thinking effort
setting. On an evidence-backed assistant reading, the oracle target-plan outputs
match the target claim grouping in **5/5 cases**, versus **1/5 ordinary rewrites**.
The source-plan control matches its own requested grouping in **4/5**, with one
composite-claim case ambiguous. This passes the previously specified small-sample
mechanistic screen: explicit grouping can change realization in these cases.
It does not establish distinctive author style, a source-only planner, or Gemma
performance. No new training or inference was executed by this review.

## Grouping and copying

These are five unused realization examples from three non-Fama economics papers,
all within the existing development split. Every target merges source sentences.
Conditions share identical source text, quoted claim evidence and meaning/style
instructions; the only treatment difference is the explicit grouping plan. The
oracle plan contains reference-derived claim grouping, not reference wording.

| Measurement | Ordinary | Source plan | Oracle target plan |
|---|---:|---:|---:|
| Outputs | 5 | 5 | 5 |
| Target sentence count matched | 4/5 | 0/5 | 5/5 |
| Target claim grouping matched | 1/5 | 0/5 | 5/5 |
| Source sentence count matched | 0/5 | 5/5 | 0/5 |
| Source claim grouping matched | 0/5 | 4/5 + 1 ambiguous | 0/5 |
| Normalized source copies | 0 | 0 | 0 |
| Median source lexical overlap | 0.643 | 0.768 | 0.735 |
| Median reference overlap, evaluation only | 0.502 | 0.541 | 0.556 |

Equal sentence counts are insufficient: ordinary rewriting gets four target
counts right but places the target claims together in only one case. The new
explicit plans demonstrate a clearer intervention than the earlier optional
plans on a mostly identity-grouping sample. The two experiments differ in both
sample selection and prompt actionability, so their difference cannot be
attributed solely to either change.

Both planned conditions retain **more source wording in every paired case** than
ordinary rewriting. None is an exact normalized copy, but the structural gain
comes with more conservative wording. Neither higher reference overlap nor
lower input overlap is itself a style-quality score. This experiment did not
measure AI detectors or establish human authorship.

All ten planned outputs report `followed`; none reports a deviation. We confirm
nine grouping assignments and abstain on one. The ambiguous source-plan case
moves the full resource-use problem into one sentence and uses only a back-reference
in the next. The source annotation assigns that composite claim to both original
sentences. A strict literal claim-set comparison misses it, while a reference-aware
reading might count the back-reference. It remains ambiguous rather than being
silently credited as full adherence. Its sentence count does match the source
plan; differing counts still definitively rule out a target-group match.

## Meaning and expressive value

The coding assistant read all 15 outputs against neutral sources and references,
with complete coverage of 37 annotated claims per condition: 111 claim instances.
Exact output quotes support each assigned claim in every assigned sentence.
These checks validate evidence consistency, not the semantic judgments themselves.
The review is unblinded, not independent or calibrated; the claim instances are
not independent statistical observations.

| Assistant claim assessment | Ordinary | Source plan | Target plan |
|---|---:|---:|---:|
| No clear additional change noticed | 30 | 33 | 34 |
| Minor qualification/degree/inferential-force shift | 6 | 4 | 3 |
| Uncertain causal-scope risk | 1 | 0 | 0 |

These are limited observations, not semantic accuracy scores. Minor shifts do
not automatically invalidate a useful rewrite. Both ordinary and target versions
of one paragraph strengthen tentative usefulness into an affirmative judgment.
Several outputs remove author conviction, weaken a described opponent's stance,
or omit emphasis that a comparison is central. The ordinary cost-survey rewrite
also moves the causal connective so that familiarity directly explains topic
selection rather than explicitly explaining brevity; this is recorded as a scope
risk, not a definite contradiction. No clear additional actor-scope broadening
was identified in this set; equivalence is not certified.

The paragraph with the known ambiguous “those people” referent retains that
ambiguity across conditions. It is a preexisting input issue, not charged as new
generation drift. Other references to omitted context remain unresolved. Original
paragraphs, source rewrites, claim inventories and prior hashes are unchanged.

The qualitative organization findings are local:

- The target plan keeps the scarcity-response restatement with its illustration
  and separates the profit-rate qualification. This matches the intended
  organization, but the merged sentence remains dense; readability is mixed.
- The cost-survey target plan keeps the familiarity-to-brevity explanation
  together. Ordinary rewriting also has two sentences but arranges those claims
  differently. The three-sentence source control is readable too.
- The cooperative-ownership target plan separates classifications that ordinary
  rewriting compresses into three sentences, yielding a locally clearer
  organization. The source plan also separates them; the target's usefulness
  statement still strengthens the input's modality.
- The Italy-data target plan groups accounting data with pre-IPO observation and
  credit data with the two directional research questions. Ordinary rewriting
  groups them differently. Both remain readable and preserve the effects as
  questions rather than findings.
- The other example already receives the target grouping from ordinary
  rewriting; the target plan supplies no clear organizational gain there.

These observations support useful control of particular presentation choices.
They do **not** demonstrate a consistent target-over-source readability advantage
or distinctive Fama/coauthor style. A model can follow a reference plan even when
another organization is equally good. No universal style score or aggregate
quality win is claimed.

## Recommended next decision

Investigate a **source-only grouping baseline**, before more QLoRA or a large
taxonomy mapper. Start with a CPU audit of the 60 proposed training annotations:
which target changes can be represented as adjacent sentence merges, which
require split/reorder operations, and which are uncertain because claims span
units. Use the existing annotations without silently repairing claims to make
mapping easier. Keep uncertainty and preexisting input defects visible.

For the eligible subset, compare a simple retrieval-based source-only merge
selector with KEEP and a fixed adjacent-merge baseline using whole-paper-held-out
checks. Neither neighbor selection nor features may use the query reference.
Only after that check should predicted plans be used in a new generation
comparison. This would test the missing deployment step; today's successful
plans are oracle information. The subsequent [CPU audit and retrieval baseline](SOURCE_ONLY_GROUPING.md)
have now run; the baseline does not beat KEEP on reference layout. A new
generation comparison has not been prepared or run. Do not use the reserved examples or start GPU training.

## Reproduction and privacy

Active private review: `research_runs/human_prose_grouping_v1/review_v2/`.
It retains raw JSONL bytes, exact output evidence, all group assignments,
per-output metrics, triplet observations and frozen hashes. `review_v1` remains
as the initial review snapshot; v2 correctly treats a differing sentence count
as a definite group mismatch even when a composite claim is ambiguous. The raw
responses and linguistic assessments are identical. Model/effort are user-reported;
current chat arrangement and hidden decoding settings are not invented. The
user's earlier cross-chat retrieval concern remains; no independence is claimed.
The session-notes file is preserved as received rather than filled speculatively.

```sh
PYTHONPATH=src python3 -m fama_style.research.human_prose_grouping_review --package research_runs/human_prose_grouping_v1/package_v2 --responses research_runs/human_prose_grouping_v1/package_v2/responses --audit research_runs/human_prose_annotation_v1/audit_v2 --assessments research_runs/human_prose_grouping_v1/assessments_v1.private.json --output /tmp/human_prose_grouping_review
python3 -m unittest discover -s tests -v
```

Use a fresh output directory. Public [aggregate results](grouping_results.json)
contain no paragraphs or per-output evidence. Private prose remains ignored by
Git. New CPU tests cover complete response transport, wrong/missing/duplicate
IDs, schema fields, deviation-note consistency, evidence for repeated claims,
composite-claim abstention, and reserved-job rejection before reading responses.
No weights, credentials, external model calls, detectors, GPU jobs or training
were accessed. All 20 reserved human-prose responses and the original Fama
reserved split remain outside development content analysis.
