# Five-pair reconstruction: content-only versus explicit guidance

The observed findings below remain unchanged. The user's subsequent selection is
[purpose-only reconstruction for all 500](GPT6_PURPOSE_ONLY_0500_V1.md): keep
enhanced briefs, supply topic/purpose/points/anchors, and withhold relations.
That is a new condition, not a relabeling of either observed arm.

Both returned files contain all five expected records with correct target hashes
and valid output fields. All ten responses report `rewritten` and empty review
notes. The briefs, human targets and generated paragraphs are retained unchanged.

**Finding:** explicit purpose/relationship guidance has selective benefits, but
does not show a consistent advantage for generating neutral counterparts to human
paragraphs. Content-only A is the provisional default for this data-generation
step. Keep the enhanced briefs for the separate brief-to-human training route;
this diagnostic does not establish whether either route improves a student model.

## Pairwise assessment

| Case | Effect of adding guidance B | Meaning and expression assessment |
| --- | --- | --- |
| P250 | Adds an explicit comparison: the Milwaukee mean is higher and the medians match. | Useful organization from the quantitative link; the comparison follows from the supplied numbers. B also differs more lexically from the target here. |
| P264 | Nearly the same data-source inventory, with minor sentence changes. | Both retain the sources, units, weighting, normalization and closedness construction. Neither shows a clear substantive advantage. |
| P291 | Makes the liquidity–collateral–price-discovery–disclosure chain explicit. | B preserves the least-cost comparison; A weakens it to particularly low-cost. Both are coherent, while A has lower target lexical similarity. |
| P458 | Ends with a clearer inference about how to evaluate behavioral finance. | Slight subjective organization benefit; A repeats the absence of quick-profit opportunities. Lexical similarity is almost the same. |
| P484 | Adds the caveat that an age comparison does not directly establish a retail/professional difference. | This is a reasonable methodological observation from an annotation qualification, but the human paragraph does not state that critique. B transfers commentary into the reconstructed content and weakens the original opening claim. A is leaner. |

No numerical mismatch or reversal of a central economic mechanism was identified
in this reading. The P291 comparative change and P484 added caveat are still
reported. Moderate explanatory additions and shifts in emphasis are not automatic
rejections. This is not exhaustive claim alignment or independent certification.

Both A and B return to the broad original sequence in P484: susceptibility claim,
age evidence, coordinated trading, measurement proposal. It occurs without the
explicit graph too, so it cannot be attributed uniquely to guidance leakage.
The sequence is also a natural way to discuss those studies. Alphabetical indexing
changes the supplied list order but does not guarantee a different realized
argument or demonstrate removal of an author's distinctive style.

## Surface diagnostics

Word-sequence similarity compares the reconstructed AI prose with its human
target. Lower values indicate lexical/order contrast, not better author style,
meaning preservation or downstream fine-tuning performance.

| Case | A words | B words | A target similarity | B target similarity |
| --- | ---: | ---: | ---: | ---: |
| P250 | 56 | 78 | 0.533 | 0.403 |
| P264 | 150 | 156 | 0.200 | 0.240 |
| P291 | 91 | 95 | 0.139 | 0.341 |
| P458 | 90 | 98 | 0.143 | 0.147 |
| P484 | 104 | 113 | 0.432 | 0.454 |
| Total words / median similarity | 491 | 540 | 0.200 | 0.341 |

B is longer in all five cases, about 10% more words overall. A has lower target
sequence similarity in four of five cases. Names, fixed terminology and data-source
roles contribute to overlap, particularly in P264/P484. These are surface measures;
do not label the four cases as four author-style wins or the median as an AI score.

## Interpretation and next decision

The compact v4 abstraction is useful as a content carrier: all five can be
realized coherently without supplying the human paragraph. The extra guidance
sometimes clarifies a comparison or communicative inference. It can also make
annotation critiques into new prose, and it does not consistently increase contrast
from the target. The guided metadata therefore should not be treated as purely
structural or automatically helpful.

For the planned full collection, retain the compact, purpose-aware briefs and use
content-only A as the provisional neutral-reconstruction default. That preserves
structured inputs for Route A (brief to human), while providing a stronger observed
contrast baseline for Route B (AI prose to human). Preserve the guided pilot outputs
as comparison data. This is a recommendation, not an executed full-500 collection
or new training approval. No output edits or new prompt changes are made here.

The comparison has five training-side cases from three author groups, no Fama
case, and one output per condition. Assessment was unblinded Codex text reading,
with no independent style judge, semantic model, detector or human panel.
GPT-6/High was requested; actual UI model/effort/decoding and cross-chat independence
are not independently verified. Sampling variation is not separated from guidance
effects. There is no evidence here of a general author-style or detector improvement.

## Reproducibility

Private comparisons, observed candidate pairs and checksummed response bindings:

`research_runs/data_scaling_v1/content_abstraction_pilot_v4/comparison_v1/`

The ten candidates retain their raw paragraphs and are not marked training-ready.
Verify with `python3 scripts/data/compare_abstraction_reconstructions_v4.py check`.
No provider calls, GPU training, detector scans or reserved-test reads were made.
