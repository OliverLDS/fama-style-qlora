"""Export the matched 27-generation inference notebook, without paragraph data."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fama_style.rhetoric.bound_grouping import load_pack,runtime_paths,file_sha


def code(source):return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':[line+'\n' for line in source.rstrip().splitlines()]}


def build():
    packpath=ROOT/'research_runs/bound_grouping_realization_v1/input_pack/bound_grouping_pack.json'
    digest=file_sha(packpath);load_pack(packpath,digest)
    paths=runtime_paths()+sorted((ROOT/'docs/claim_verifier_calibration_v3').glob('*.json'))
    files={str(p.relative_to(ROOT)):p.read_text() for p in paths}
    setup=('import json, sys\nfrom pathlib import Path\n'
        'runtime = Path("/kaggle/working/_bound_grouping_runtime")\n'
        'files = json.loads('+repr(json.dumps(files,ensure_ascii=False))+')\n'
        'for name, source in files.items():\n    path = runtime / name\n    path.parent.mkdir(parents=True, exist_ok=True)\n    path.write_text(source, encoding="utf-8")\n'
        'sys.path.insert(0, str(runtime / "src"))\n')
    check=('import importlib\nexperiment = importlib.import_module("fama_style.rhetoric.bound_grouping")\n'
        'matches = list(Path("/kaggle/input").rglob("bound_grouping_pack.json"))\n'
        'if len(matches) != 1:\n    raise ValueError("Attach exactly one private bound_grouping_pack.json")\n'
        'arguments = ["--pack", str(matches[0]), "--expected-pack-sha", '+repr(digest)+']\n'
        'experiment.main(arguments + ["--output-dir", "/kaggle/working/bound_grouping_preflight"])\n')
    run='experiment.main(arguments + ["--execute-inference", "--output-dir", "/kaggle/working/bound_grouping_realization_v1"])\n'
    book={'cells':[{'cell_type':'markdown','metadata':{},'source':[
        '# Source-bound grouping realization: 27 inference generations\n',
        'Create a new private notebook. Enable internet, GPU T4 x2 and HF_TOKEN in Kaggle Secrets. Only GPU 0 is exposed. Attach a private Dataset containing ONLY bound_grouping_pack.json. Keep all input and output files private.\n',
        'Three selected TRAINING pairs, three sampled candidates per condition. A: concrete prompt only. B: same prompt plus source units in separate groups. C: same prompt and units, with source-bound reference-derived grouping hypotheses. Primary C versus B compares grouping assignments; A versus B also changes the unit scaffold. C is an oracle diagnostic, not deployment or learned-mapper performance.\n',
        'Pinned base Gemma 3 4B, NF4/FP32, seeds 41/42/43, temperature 0.6, top-p 0.9, 768 new tokens, no truncation. No adapters, training, forced decoding, repairs or semantic acceptance gate. The same allowance for minor drift applies to all conditions.\n',
        'The grouping arms request one sentence per source-unit group in order. Count agreement is not evidence that the right claims were grouped. Full paragraph verdicts and positional group verdicts are retained. Ambiguous sentence boundaries and count mismatches leave group alignment unresolved. Every candidate remains visible.\n',
        'Reference wording and evaluation-only semantic-obligation notes are never placed in generator prompts. Source units are duplicated in B/C to match the evidence scaffold. Gemma is unloaded before the semantic verifier loads.\n',
        'Download /kaggle/working/bound_grouping_realization_v1/: evaluation.json, generations.private.jsonl, metrics.private.jsonl, verifier_results.private.jsonl, group_verifier_results.private.jsonl, runtime.json, token_audit.json, run_status.json and manifest.json. Qualitative style/drift assessment remains pending execution. No run was submitted by this builder.\n'  ]},
        code('import os\nos.environ["CUDA_VISIBLE_DEVICES"] = "0"\nos.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"\nos.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"'),
        code('%pip install -q "transformers==5.20.1" "accelerate==1.14.0" "bitsandbytes==0.50.2" "huggingface-hub==1.33.0" "sentencepiece>=0.2"'),
        code(setup),code(check),code(run)],
        'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    path=ROOT/'kaggle/bound_grouping.ipynb';path.write_text(json.dumps(book,ensure_ascii=False,indent=2)+'\n')
    print('Built kaggle/bound_grouping.ipynb; no paragraph data, upload or execution.')


if __name__=='__main__':build()
