# Stage 2: less copying, more organization change, some annotation leakage

All four expected response files arrived: eight direct rewrites and eight neutral
reconstructions. All 16 rows pass schema, identity, method and coverage checks;
none is blocked. The user reports GPT-6 with High thinking effort for all four
files. Raw responses and reviewed references are preserved unchanged.

Neutral reconstruction produces more lexical and organizational contrast in
this small paired sample. It also introduces extra commentary in several cases.
This supports retaining it as a candidate data-construction method, not scaling
it unfiltered or claiming that it has improved a trained model.

## Descriptive results

| Measure | Direct rewriting | Neutral reconstruction |
|---|---:|---:|
| Generated outputs | 8 | 8 |
| Median word-sequence overlap with human reference | 0.704 | 0.414 |
| Median longest contiguous shared word run | 13.5 | 7.5 |
| Median word-set Jaccard similarity | 0.630 | 0.452 |
| Median generated/reference word-count ratio | 0.931 | 1.038 |
| Exact normalized word-token copies | 0 | 0 |
| Sentence-count change (surface proxy) | 3/8 | 3/8 |
| Major claim-order changes (automated review) | 0/8 | 5/8 |
| Outputs retaining first-person voice | 6/8 | 0/8 |
| Outputs with a nonmaterial review flag | 3/8 | 6/8 |
| Outputs adding epistemic metacommentary | 0/8 | 3/8 |
| Material claim errors identified in this review | 0/8 | 0/8 |

Reconstruction has lower word-sequence overlap in **all eight paired cases**.
Overlap is a case-folded word-sequence matching statistic, not a percentage of
AI content, a detector score or a writing-quality measure. Spelling, hyphenation,
voice, vocabulary and word order can all affect it. The longest-run and word-set
measurements provide complementary copying diagnostics.

The organization observations come from reading each output against the reviewed
human reference. Reconstruction moves temporal evidence into chronological order,
moves sample-period information after data selection, moves an economic mechanism
ahead of the market comparison, or relocates qualifications/history in the paragraph.
These are observed organization changes, not evidence that every new order is better.
Sentence counts alone miss changes in which information appears first or last.

## Fidelity and readability

All outputs retain the main economic claims in the qualitative automated review.
Neither method loses or introduces a distinct numeric literal. Repetition of a
number can change, so numeric mention-count differences are retained separately.
These checks and judgments do not establish complete semantic equivalence.

The nonmaterial flags are specific rather than a blanket failure label. They
include small confidence/frequency shifts, a stronger evidential formulation,
an inferred argument link, a purpose-to-function strengthening and an omitted
technical gloss. Material errors and these local changes remain separate,
consistent with the project's bounded-drift objective. Flag counts do not weight
the severity of individual edits or constitute calibrated error probabilities.

Three reconstructions turn annotation guardrails into additional prose: a
contrast with measured liquidity, an assurance that attributed views are not
established conclusions, or a statement that a model prediction is not an
empirical finding. Such distinctions can be useful instructions to a writer
without belonging in the rewritten paragraph. Their presence is a data-generation
failure to address, even where the underlying economic claims remain intact.

Both methods produce readable prose in the automated assessment. Reconstruction
sometimes becomes wordier or less direct through reporter framing; one output
returns to modern-market frequency after discussing the historical analogy.
No independent human readability ranking is available. Median words per sentence
are about 22.85 for direct rewriting and 25.04 for reconstruction, but that proxy
does not establish which prose is easier to understand.

The six references with first-person language retain it under direct rewriting
and lose it under reconstruction. Thus part of the contrast comes from changing
narrative viewpoint to an account of what the authors say. This is not proof of
distinctive author-style transformation or a desirable general pronoun rule.

## Coverage, provenance and limits

The complete starting selection remains 12 cases. Four were held before
generation; each method generated eight of the original 12. There were no
replacements, dropped generation-time failures or retries based on scores.
The eight approved pairs cover six papers, with three Holmström, three
Baker–Wurgler and two Shiller cases. This is a purposive, unbalanced development
sample, not a fresh test set or a representative author benchmark.

Method labels were masked for the first source/output text comparison, but the
reviewer knew the study setup and had seen method-level word counts. The method
can often be inferred from the prose. This is neither independent nor completely
blinded evaluation; no population accuracy or significance claim is made.

Actual generation conversation arrangement was not restated. A user-shared direct
preflight listed both input batches together; uniform isolation is not established.
Model snapshot, decoding, memory/retrieval and hidden UI settings are uncontrolled,
even with matching reported model/effort labels. The reconstruction inputs contain
41,644 bytes versus 5,721 for direct inputs, about 7.3 times as much input text by
bytes. This is not a token-billing or runtime estimate.

No Fama-style classifier, AI detector, Gemma inference or GPU training was run.
These results concern constructing synthetic counterparts, not the performance
of an adapter. Reduced copying does not establish improved humanity, reduced
detectability or distinctive Fama/coauthor alignment.

## Recommendation

Keep both methods. Before expanding reconstruction annotation, run a small
realization-instruction check on the three observed commentary cases. Keep the
reviewed briefs, references and first outputs fixed; ask the writer to use
modality/attribution fields to choose faithful wording, without appending
reviewer-style evidence disclaimers absent from the source. Preserve economic
uncertainty and attribution; do not delete those semantic protections merely
to obtain cleaner prose. This would be a targeted development repair, not a new
held-out performance test. It has not been executed or submitted by this review.

Judge whether the extra commentary disappears while copying remains lower and
claim preservation stays acceptable. Do not start fine-tuning on these outputs
yet. The present result supports a useful contrast mechanism with a fixable
realization issue; it does not certify training-ready pairs.

## Artifacts and verification

[Aggregate results](stage2_results.json) contain no paragraph text. Private files
are in `research_runs/neutral_reconstruction_v1/stage2_review_v1/`: the label-masked
bundle, mapping, recorded qualitative judgments, provenance, and `results/` with
four unchanged raw files, per-output metrics, paired differences, receipt and
manifest. Judgments were saved before the mapping was used to aggregate methods.

The implementation is `src/fama_style/research/neutral_reconstruction_results.py`.
Synthetic tests cover generated/blocked response contracts, exact-copy metrics,
order-sensitive versus vocabulary-only overlap and repeated numeric mentions.
Replaying the frozen responses and recorded judgments produces byte-identical
result files. That verifies transport and calculation reproducibility, not the
independent correctness of qualitative judgments. All existing data, sources,
reserved splits and earlier experimental results remain unchanged.
