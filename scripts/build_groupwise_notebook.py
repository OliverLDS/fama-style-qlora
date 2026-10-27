"""Export the matched 42-call inference notebook, without paragraph data."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fama_style.rhetoric.groupwise_realization import load_pack,runtime_paths,file_sha


def code(source):return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':[line+'\n' for line in source.rstrip().splitlines()]}


def build():
    packpath=ROOT/'research_runs/groupwise_realization_v1/input_pack/groupwise_pack.json'
    digest=file_sha(packpath);load_pack(packpath,digest)
    paths=runtime_paths()+sorted((ROOT/'docs/claim_verifier_calibration_v3').glob('*.json'))
    files={str(p.relative_to(ROOT)):p.read_text() for p in paths}
    setup=('import json, sys\nfrom pathlib import Path\n'
        'runtime = Path("/kaggle/working/_groupwise_runtime")\n'
        'files = json.loads('+repr(json.dumps(files,ensure_ascii=False))+')\n'
        'for name, source in files.items():\n    path = runtime / name\n    path.parent.mkdir(parents=True, exist_ok=True)\n    path.write_text(source, encoding="utf-8")\n'
        'sys.path.insert(0, str(runtime / "src"))\n')
    check=('import importlib\nexperiment = importlib.import_module("fama_style.rhetoric.groupwise_realization")\n'
        'matches = list(Path("/kaggle/input").rglob("groupwise_pack.json"))\n'
        'if len(matches) != 1:\n    raise ValueError("Attach exactly one private groupwise_pack.json")\n'
        'arguments = ["--pack", str(matches[0]), "--expected-pack-sha", '+repr(digest)+']\n'
        'experiment.main(arguments + ["--output-dir", "/kaggle/working/groupwise_preflight"])\n')
    run='experiment.main(arguments + ["--execute-inference", "--output-dir", "/kaggle/working/groupwise_realization_v1"])\n'
    book={'cells':[{'cell_type':'markdown','metadata':{},'source':[
        '# Groupwise realization and verbatim assembly: 42 generation calls\n',
        'Create a new private notebook. Enable internet, GPU T4 x2 and HF_TOKEN in Kaggle Secrets. Only GPU 0 is exposed. Attach a private Dataset containing ONLY groupwise_pack.json. Keep all full-text files private.\n',
        'Same three TRAINING examples and oracle groups. A: repeat whole-paragraph oracle prompting, 9 calls. B: write each group in a separate call and join returned pieces in group order, 33 calls producing 9 assembled paragraphs. Full source context is supplied to every call, without previous generated pieces. Reference wording and evaluation notes are never sent to Gemma.\n',
        'Both strategies receive 768 output tokens per paragraph. B allocates them by source-group word counts; a group has no additional retries. There are 18 paragraph candidates from 42 generation calls. Seeds 41/42/43 identify paragraph samples; deterministic group sub-seeds are recorded. Input overhead and total runtime are measured separately.\n',
        'Pinned base Gemma 3 4B, NF4/FP32, temperature 0.6, top-p 0.9, no top-k cutoff, 4096 context limit and no truncation. No adapters, training, forced one-sentence decoding, repairs, fallback or coherence rewrite. The same allowance for minor drift applies to both strategies.\n',
        'Multi-sentence, ambiguous or unpunctuated group outputs remain verbatim and are flagged; assembly does not guarantee valid sentence count or faithful claims. Empty/cutoff generations leave an incomplete run. Gemma unloads before paragraph/group verifier calls. All full verdicts, raw calls, assembled paragraphs, formats, controls and costs are retained.\n',
        'Download /kaggle/working/groupwise_realization_v1/: evaluation.json, generations.private.jsonl (42 calls), paragraphs.private.jsonl and metrics.private.jsonl (18 paragraph views), verifier_results.private.jsonl, group_verifier_results.private.jsonl, runtime.json, token_audit.json, run_status.json and manifest.json. No inference has been submitted by this builder.\n'   ]},
        code('import os\nos.environ["CUDA_VISIBLE_DEVICES"] = "0"\nos.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"\nos.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"'),
        code('%pip install -q "transformers==5.20.1" "accelerate==1.14.0" "bitsandbytes==0.50.2" "huggingface-hub==1.33.0" "sentencepiece>=0.2"'),
        code(setup),code(check),code(run)],
        'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    path=ROOT/'kaggle/groupwise_realization.ipynb';path.write_text(json.dumps(book,ensure_ascii=False,indent=2)+'\n')
    print('Built kaggle/groupwise_realization.ipynb; no paragraph data, upload or execution.')


if __name__=='__main__':build()
