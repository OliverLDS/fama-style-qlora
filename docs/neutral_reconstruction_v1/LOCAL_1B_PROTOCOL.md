# Local quantized 1B prompt diagnostic v1

Nine first-attempt outputs compare three existing training-side cases under
three conditions: direct rewriting of the original paragraph, reconstruction
from the detailed repaired brief, and reconstruction from a compact claim list.
No fresh or reserved examples are selected. No adapter or training is involved.
This is a small prompt diagnostic, not evidence about Gemma 4B performance or an
automatic eligibility gate for a later 4B experiment.

The user's local server reports `ggml-org/gemma-3-1b-it-GGUF:Q4_K_M`, with four
slots and 4,096 tokens per slot. The helper script was inspected for its HTTP
interface; its unrelated default model label is not used. The repository client
connects only to a loopback HTTP endpoint, bypasses proxy settings, and explicitly
checks the served model. It does not load weights or import ML libraries in the
Python process. Actual CPU/Metal placement inside the existing server is not
established by these endpoints and is not assumed.

All conditions use the same neutral-prose and meaning-preservation instruction,
prose-only response format, greedy temperature zero, seed 1729, at most 512 output
tokens, top-p one, top-k zero, min-p zero and repetition penalty one. Requests are
serial, with one user message and no conversation history. Prompt-cache reuse is
disabled. Each case rotates its condition order. All first responses, raw HTTP
response objects, finish reasons, usage and timings are retained without retries.

The compact transformation is deterministic: resolve actor IDs to their names,
retain every claim proposition, nonempty semantic qualifier, supported edge,
essential term and citation; remove empty fields and predicate labels, and render
the result as a flat claim list. Claim IDs and order remain the detailed brief's
opaque IDs and order. This jointly tests compact representation and explicit
actor names; it is not a pure character-count ablation. It does not add a target
rhetorical plan or reference wording. The confidence and media-authority repair
from the 4B packet is retained in both brief conditions.

Before generation, the client calls the server's template renderer and tokenizer
for all nine prompts and requires input plus output budget plus a small margin
to fit the actual slot context. It records model-file SHA256, server build,
chat-template SHA256, the server defaults and explicit request overrides. Defaults
are not reported as the actual requested decoding configuration. See the
[llama.cpp server API](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
for `/apply-template`, `/tokenize`, `/props` and chat completion semantics.

Source paragraphs are legitimate inputs only to the direct condition. Neither
brief condition receives original paragraphs, previous outputs, reference-informed
style plans or other conditions' conversation history. References are joined after
generation for review. All inputs and outputs remain in ignored private folders.

Assess claims, actor binding, modality and scope, causal direction, commentary,
readability, organization and the existing overlap metrics separately. Keep
format validity, generation completion and semantic quality distinct. Retain
blocked, empty and length-limited cases in the denominator. Qualitative automated
review is unblinded and not independent semantic gold. The three failure-selected
cases do not establish generalization or Fama-specific style.

The private run uses
`research_runs/neutral_reconstruction_v1/local_1b_v1/package_v2/` and `run_v1/`.
The earlier unexecuted `package/` is a draft preserved before a runner identity
comparison correction; use `package_v2`. The correction compares stable model
identity rather than volatile response metadata. The runner is
`fama_style.research.local_reconstruction`, with `prepare` and `run` subcommands.

Differences in model size, GGUF versus NF4 quantization, engine, chat template,
decoding implementation and prose-only contract prevent direct causal comparison
with the earlier Kaggle 4B JSON experiment. A compact-format benefit here, if any,
requires an independent matched inference check on 4B before post-training.
