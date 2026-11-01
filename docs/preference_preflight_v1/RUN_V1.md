# Version 1: Kaggle completed, preflight did not execute

Update: [version 2 was recovered locally after a code fix](RUN_V2_RECOVERY.md).
The real tokenizer/mask gate now passes. The initial starter-cell audit below
is preserved as historical evidence; no further Kaggle replay is required now.

Kaggle reports `oliverlds/preference-token-preflight`, version 1, as COMPLETE.
The version-specific download contains only the log: no runtime, state, token
audit, summary, encoded pairs or tokenizer bundle was produced.

The log lists the two attached Dataset filenames and then notebook conversion.
The current saved source contains **one default Kaggle starter code cell**,
which lists input files; it does not contain the tokenizer-preflight program.
The prepared local artifact contains **one Markdown cell and four code cells**.
This is a source-import mismatch, not an observed tokenizer, mask or training
failure. A successful Kaggle commit status alone cannot pass the gate.

The CLI denied the version-specific source pull. The unversioned current source
was retrieved separately and is identified as such in the receipt; the version-1
log independently establishes that no preflight outputs were generated.
Metadata shows GPU/TPU disabled and Internet enabled. No remote source was changed,
job submitted or credential value accessed by the review tools.

## Minimal correction

Keep the existing attached Dataset. Import the prepared local file
`kaggle/preference_token_preflight.ipynb` **as notebook code** into the existing
Kaggle notebook. Adding an `.ipynb` to a Dataset does not replace notebook cells.
The first heading should be **Actual-error preference preflight — CPU only**,
followed by four code cells: CPU environment, dependencies, runtime file and launch.

Keep Accelerator **None / CPU**, Internet and the existing HF_TOKEN Secret
enabled. Save & Run All and share the new version. There is no need to recreate
the Dataset, select T4 or download model weights. Expected console progress is
`Checked 1/17 preference pairs` through `Checked 17/17 preference pairs`.

Until that execution is reviewed, real tokenization, archived-ID equality and
response masking remain unverified. The proposed training pilot remains
design-only and unapproved. No preference-training or SFT job was started.

Private raw evidence and receipts:
`research_runs/preference_token_preflight_v1/{kaggle_v1,source_current,review_v1}/`.
Public status: [run_v1_results.json](run_v1_results.json). The frozen packet and
prepared notebook are unchanged; their hashes were checked during review.
