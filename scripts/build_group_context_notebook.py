"""Export the 18-call read-only context diagnostic without private text."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fama_style.rhetoric.group_context_restoration import load_pack,runtime_paths,file_sha


def code(source):return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':[line+'\n' for line in source.rstrip().splitlines()]}


def build():
    packpath=ROOT/'research_runs/group_context_restoration_v1/input_pack/context_pack.json'
    digest=file_sha(packpath);load_pack(packpath,digest)
    paths=runtime_paths()+sorted((ROOT/'docs/claim_verifier_calibration_v3').glob('*.json'))
    files={str(p.relative_to(ROOT)):p.read_text() for p in paths}
    setup=('import json, sys\nfrom pathlib import Path\n'
        'runtime = Path("/kaggle/working/_group_context_runtime")\n'
        'files = json.loads('+repr(json.dumps(files,ensure_ascii=False))+')\n'
        'for name, source in files.items():\n    path = runtime / name\n    path.parent.mkdir(parents=True, exist_ok=True)\n    path.write_text(source, encoding="utf-8")\n'
        'sys.path.insert(0, str(runtime / "src"))\n')
    check=('import importlib\nexperiment = importlib.import_module("fama_style.rhetoric.group_context_restoration")\n'
        'matches = list(Path("/kaggle/input").rglob("context_pack.json"))\n'
        'if len(matches) != 1:\n    raise ValueError("Attach exactly one private context_pack.json")\n'
        'arguments = ["--pack", str(matches[0]), "--expected-pack-sha", '+repr(digest)+']\n'
        'experiment.main(arguments + ["--output-dir", "/kaggle/working/group_context_preflight"])\n')
    run='experiment.main(arguments + ["--execute-inference", "--output-dir", "/kaggle/working/group_context_restoration"])\n'
    book={'cells':[{'cell_type':'markdown','metadata':{},'source':[
        '# Read-only source-context restoration: 18-call diagnostic\n',
        'Create a NEW PRIVATE notebook; enable internet, GPU T4 x2 and HF_TOKEN in Secrets. Only GPU 0 is used. Attach a PRIVATE dataset containing ONLY context_pack.json. No credentials belong in the dataset.\n',
        'Three previously failing training groups: managerial supervision, dividend rates with missing actors, and a numeric-control group. Three matched seeds, two conditions: A exact historical local-only prompt; B the same prompt plus clearly marked read-only full source context and context-use instructions. Same group token budgets and sampling seeds. No reference wording, retries, repairs, paragraph assembly, adapters or training. The full pack retains the three historical cases for frozen constructors, but only the selected three groups from two cases generate calls.\n',
        'Local-only controls should reproduce prior output hashes; changed controls require investigation. Empty/cutoff outputs remain failed samples, not fallback text. Multi-sentence outputs are retained and flagged. Both arms use the same minor-drift allowance. This checks missing referents; it does not establish arithmetic fidelity or author-style quality.\n',
        'After Gemma unloads, saved diagnostics include local-group verifier evidence and full-source-to-output NLI. Local NLI may penalize valid antecedent resolution; full-context entailment can accept extra source claims. Neither is a calibrated semantic grade or automatic selection gate. Review style, central relation/quantity changes, and leakage of other groups separately.\n',
        'Download /kaggle/working/group_context_restoration/: evaluation.json, generations.private.jsonl, metrics.private.jsonl, verifier_results.private.jsonl, jobs.private.jsonl, token_audit.json, runtime.json, run_status.json and manifest.json. No GPU job is submitted by this builder.\n']},
        code('import os\nos.environ["CUDA_VISIBLE_DEVICES"] = "0"\nos.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"\nos.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"'),
        code('%pip install -q "transformers==5.20.1" "accelerate==1.14.0" "bitsandbytes==0.50.2" "huggingface-hub==1.33.0" "sentencepiece>=0.2"'),
        code(setup),code(check),code(run)],
        'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    path=ROOT/'kaggle/group_context_restoration.ipynb';path.write_text(json.dumps(book,ensure_ascii=False,indent=2)+'\n')
    print('Built kaggle/group_context_restoration.ipynb; no paragraph data, upload or execution.')


if __name__=='__main__':build()
