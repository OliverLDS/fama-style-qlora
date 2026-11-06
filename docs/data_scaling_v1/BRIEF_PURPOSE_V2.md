# Content briefs with a communicative purpose

Superseded for new annotation by [version 3](BRIEF_PURPOSE_V3.md). The user now
requests recollecting briefs and reconstructions for all 500 targets after a
five-case pilot; the reuse-only recommendation below is historical.

The earlier brief has `task`, `key_points` and `anchors`. Its task often hints at
purpose, but that purpose is not explicit or consistently checked. Version 2 adds
a primary function, up to two secondary functions, a reader takeaway and selected
relationships between content points. The extraction and reconstruction prompts
are [Stage 1](BRIEF_EXTRACTION_V2.md) and [Stage 2](BRIEF_RECONSTRUCTION_V2.md).

The new labels reuse a **small paragraph-level subset** of the existing discovery
work. They are not a validated new three-level taxonomy. The discovery proposal
is still provisional; these functions are interpretations, not author intent.

| Brief purpose | Related discovery item(s) |
| --- | --- |
| introduce_topic | P002 Topic introduction |
| define_concept | P015 Conceptual definition; P016 Operational measure definition |
| describe_phenomenon / report_result | P022 Empirical observation; P023 Frequency result; P024 Summarized finding |
| explain_mechanism | P038 Proposed causal or functional mechanism; P039 Information transmission channel |
| explain_difference / compare_accounts | P030 Contrast or counterpoint; P031 Alternative explanation; P070 Contrastive progression |
| describe_method / introduce_data | P007 Method or measurement procedure; P008 Data source introduction |
| provide_background / illustrate_case | P009 Contextual background; P028 Illustrative case or analogy |
| state_problem / draw_implication | P047 Problem or puzzle statement; P035 Consequence or implication |
| qualify_claim | P048–P054 caveat categories |
| evaluate_argument / recommend_action | P044 Evaluation criterion; P045 Evaluative assessment; P046 Recommendation or advice |
| outline_paper | P005 Roadmap or stage |

Here `Pxxx` denotes a category in the discovery proposal, not a training pair ID.
The exact source is recorded by hash in the repair-packet manifest:
`research_runs/taxonomy_discovery_v1/responses/task_3_consolidation/taxonomy_v0_1_proposal_v2.json`.
The evidence-bound relation principle also reuses the earlier annotation prompt:
presentation labels must not introduce unsupported semantic connections.

Sentence shapes, author-specific lexical choices, quote evidence and a complete
claim graph are deliberately outside this compact brief. Keeping a paragraph's
function is useful; preserving its entire original rhetorical realization would
undermine fresh composition. Neither function labels nor relations require a fixed
wording or an original sentence sequence.

Start with the five repaired targets that still need briefs/reconstructions.
Reuse the 495 existing briefs and previous generation work. Do not ask for another
500 manual annotations merely to fill new metadata. After those five return,
assess whether purpose is useful before any larger enrichment. Legacy briefs remain
version 1; missing purpose metadata must not be silently labeled as an LLM annotation.

For future Route A inference, extract these same fields from the actual AI input,
without a human reference. A purpose inferred from a training target is oracle
information relative to that target; do not assume it will be equally recoverable
from every deployment input. Use the same serialization for training and inference
when that experiment is eventually implemented. No training change is made here.
