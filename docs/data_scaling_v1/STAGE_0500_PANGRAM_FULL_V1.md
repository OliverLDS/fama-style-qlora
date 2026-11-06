# Complete mixed-500 Pangram panel: limited improvement over the actual inputs

The [eight reference-control PDFs](STAGE_0500_REFERENCE_CONTROLS_V1.md) are now
imported: all eight are Human Written with 100% human share. This added little
to the earlier published-human calibration; routine repeat reference scans will
not be requested. The input/output findings below remain unchanged.

All 18 new PDFs have been imported and joined to the first six. The complete
panel contains eight frozen Gemini-input cases and their base and fine-tuned
outputs: **24 reports**, covering two paragraphs from one development paper per
author group (four papers). All are Pangram 4.1 reports dated 5 November 2026.
Every PDF is bound to the complete scan text and recorded by hash: eight raw
whitespace matches and sixteen layout/punctuation/whitespace matches. No report
was excluded or selected after seeing scores. Earlier two-case results remain
unchanged in their original import version.

## Paired results

Displayed **human-written document share**, retained exactly:

| Case | Gemini input | Base Gemma | Mixed-500 adapter |
| --- | ---: | ---: | ---: |
| Fama/coauthor: inflation and bill rollover | 0.1% | 0.1% | 100% |
| Fama/coauthor: CPI inflation definition | 100% | 22% | 100% |
| Shiller: Fisher's crisis theory | 67% | 0.1% | 0.1% |
| Shiller: compensated dollar and indexation | 0.1% | 0.1% | 0.1% |
| Holmström: small linear-programming models | 1% | 0.1% | 0.1% |
| Holmström: deductibles and informativeness | 20% | 0.1% | 0.1% |
| Baker–Wurgler/coauthors: fund characteristics | 4% | 0.1% | 0.1% |
| Baker–Wurgler/coauthors: Regulation FD | 0.1% | 0.1% | 0.1% |

| Endpoint | Input | Base | Fine-tuned |
| --- | ---: | ---: | ---: |
| Entirely human native label | 1/8 | 0/8 | 2/8 |
| Mean human-written share | 24.04% | 2.84% | 25.08% |
| Median human-written share | 2.5% | 0.1% | 0.1% |

Against the actual inputs, fine-tuning increases human share in **one case**,
decreases it in **four**, and leaves it unchanged in **three**. Of seven inputs
with AI-related labels, only **one** becomes entirely human-labeled. The other
human-labeled output preserves an already-human input label. Against base,
human share increases in two cases and is unchanged in six.

Pangram's authorship shares are not document AI probabilities. Display floors
such as 0.1% are retained without rescaling, even when displayed shares sum
slightly above 100%. Native labels also matter: the trained outputs comprise
two Human Written, four AI Assisted and two mixed AI/AI-assisted outcomes.
A shift from AI Generated to AI Assisted does not itself mean human-written.

## Interpretation and next diagnostic

The adapter improves on base, but provides little consistent detector advantage
over its actual Gemini inputs. The near-equal mean input/adapter shares conceal
four deteriorations and one large improvement. This panel does not support
scaling training to 1,000 examples solely to reduce Pangram detectability.

Both human-labeled outputs come from the same Fama/coauthor paper, and one of
their inputs was already human-labeled. The new conversion retains high input
similarity (0.868) and introduces literal HTML subscripts. Topic, formatting,
copying and paper-specific effects remain confounds; these two outputs do not
establish distinctive Fama-style learning or broad multi-author transfer.

The earlier automated review identified material errors in six base outputs
and three trained outputs on these eight inputs. Neither human-labeled trained
output received a material flag, a useful positive detail. The remaining trained
issues—price level becoming asset prices, an incorrect informativeness claim,
and prospective pruning becoming current underperformance—remain relevant.
Meaning judgments are provisional same-agent assessments, and small stylistic
shifts were separately allowed rather than automatically counted as failures.

Before more training, check whether Pangram recognizes the intended targets:
the **eight original human reference paragraphs**, approximately **15 credits**.
No exact reference-text matches were found in seven imported observation JSONL
files; that search does not cover unimported PDFs or outside scans. These controls
are now prepared, with exact reference hashes bound to the frozen packet, under
`research_runs/data_scaling_v1/corpus_500_pilot_4b_v1/review_v1/pangram_reference_controls_v1/`.
Open `scan_references.html` and save all eight exported Pangram PDFs in its adjacent
`pdf_reports/` folder. Default filenames are fine; no CSV or manual scores are
required. The eight texts total 1,041 whitespace-counted words and an estimated
15 credits; actual provider counts may differ. No scan has been submitted by the
agent. Builder: `scripts/data/prepare_corpus_500_reference_scans.py`.
These controls would distinguish a
model failing to reach detectable target features from a detector that also
flags those target paragraphs. They should not become a new data-selection
filter or justify editing references to satisfy a detector.

Keep the mixed-500 adapter, original datasets, reports and separate-teacher plan.
No new generation, GPU job or training was launched. Separate Gemini/GPT-6 cohort
completion remains deferred while this evaluation is interpreted; no additional
manual GPT-6 batches or author downloads are requested.

Private archives: `research_runs/data_scaling_v1/corpus_500_pilot_4b_v1/pangram_remaining_reports_v1/`.
Combined evidence and observations: `pangram_full_import_v1/` under that run.
Public aggregate: `stage_0500_pangram_full_v1.json`. Reproducible CPU importer:
`scripts/data/import_corpus_500_full_pangram.py`. Five focused parser/pairing
tests passed. No reference or generated paragraphs are published.
