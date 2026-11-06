# Matched input/output detector evaluation v2

The earlier GPT-6 controls showed that some generated inputs receive human-written
labels. The 200-pair strict-adapter result could therefore reflect retention of
already human-labeled inputs. Its six human-written outputs are observed outcomes,
but cannot yet establish improvement over the input itself.

## Complete the current eight-case comparison

The exact eight source texts are identical across both prompts and model
conditions. Searching four imported observation files (143 records) found zero
exact source-text hash matches. This checks imported local evidence, not every
possible unimported scan. A frozen packet is prepared at:

`research_runs/data_scaling_v1/prompt_comparison_pangram_v1/input_controls_v1/scan.html`

Scan those eight full inputs unchanged and save PDFs in the adjacent
`pdf_reports/` directory. Estimated cost is 15 credits; actual provider word
counting may differ. No output or reference rescans are requested. The 32 output
reports already supplied will be reused. No input score has been inferred from
an earlier, different example.

After import, show a five-column case table: input, strict base, strict adapter,
simple base and simple adapter. Retain provider-native labels and displayed
authorship shares, text hashes, dates and versions. Record changes relative to
the exact input as well as base-versus-adapter differences. A human-labeled input
remaining human-labeled after rewriting is not evidence of detector conversion.
Meaning changes, copying and readability remain separate outcomes, not automatic
exclusion gates.

## Required controls for future evaluations

- Freeze cases before generation and before seeing detector scores. Scan both
  the actual input and every output used in the detector comparison. The input
  scan also supplies the unchanged-input control.
- Add a separate **Gemma-generated input condition** to test transfer beyond
  GPT-6 rewrites. Freeze one generated passage per case and give exactly that
  text to both the base and adapted rewriter, using matched instructions and
  decoding. Keep GPT-6 and Gemma input results separate.
- Scan all preselected inputs. Do not silently retain only detector-positive
  inputs. Report results by their observed baseline label as well as over the
  full panel. Earlier success detecting other Gemma passages does not guarantee
  that every new Gemma passage will be detected.
- Record generating model/revision, generation prompt, decoding, any adapter,
  input hashes, provider version/date and available settings provenance. Report
  any changed detector version or settings across paired scans.
- Assess each rewrite against its actual input. Existing base-Gemma outputs
  include claim errors; do not label them meaning-equivalent to a historical
  paragraph or turn them into training pairs without a separate alignment audit.
  Keep generation errors distinct from errors introduced by rewriting.
- Retain all scores, missing scans, conflicting reports and meaning flags.
  Displayed authorship shares are not document AI probabilities. Avoid claiming
  author-specific style learning from detector outcomes alone.

This is an evaluation-policy update. It does not modify training data, the
executed pilot, or past results, and does not initiate Gemma generation or GPU
training. No additional annotation batches are requested.
