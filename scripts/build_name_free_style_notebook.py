"""Export the matched 24-generation inference notebook, without paragraph data."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fama_style.rhetoric.name_free_ablation import load_pack,runtime_paths,file_sha


def code(source):return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':[line+'\n' for line in source.rstrip().splitlines()]}


def build():
    packpath=ROOT/'research_runs/name_free_style_ablation_v1/input_pack/name_free_style_pack.json'
    digest=file_sha(packpath);load_pack(packpath,digest)
    paths=runtime_paths()+sorted((ROOT/'docs/claim_verifier_calibration_v3').glob('*.json'))
    files={str(p.relative_to(ROOT)):p.read_text() for p in paths}
    setup=('import json, sys\nfrom pathlib import Path\n'
        'runtime = Path("/kaggle/working/_name_free_style_runtime")\n'
        'files = json.loads('+repr(json.dumps(files,ensure_ascii=False))+')\n'
        'for name, source in files.items():\n    path = runtime / name\n    path.parent.mkdir(parents=True, exist_ok=True)\n    path.write_text(source, encoding="utf-8")\n'
        'sys.path.insert(0, str(runtime / "src"))\n')
    check=('import importlib\nexperiment = importlib.import_module("fama_style.rhetoric.name_free_ablation")\n'
        'matches = list(Path("/kaggle/input").rglob("name_free_style_pack.json"))\n'
        'if len(matches) != 1:\n    raise ValueError("Attach exactly one private name_free_style_pack.json")\n'
        'arguments = ["--pack", str(matches[0]), "--expected-pack-sha", '+repr(digest)+']\n'
        'experiment.main(arguments + ["--output-dir", "/kaggle/working/name_free_style_preflight"])\n')
    run='experiment.main(arguments + ["--execute-inference", "--output-dir", "/kaggle/working/name_free_style_ablation_v1"])\n'
    book={'cells':[{'cell_type':'markdown','metadata':{},'source':[
        '# Author-name versus concrete name-free style: 24 generations\n',
        'Create a new private notebook. Enable internet, GPU T4 x2 and HF_TOKEN in Kaggle Secrets. Only GPU 0 is visible. Attach a private Dataset containing ONLY name_free_style_pack.json. Keep inputs and outputs private.\n',
        'Reuse the same four previously exposed development inputs. A repeats the previous author-name control exactly; B replaces only its style instruction with a concrete name-free specification. Common semantic tolerance and output instructions are identical. Three candidates per arm, paired seeds 41/42/43; 24 generations total. References are never given to the generator.\n',
        'Pinned base Gemma 3 4B, NF4/FP32, temperature 0.6, top-p 0.9, no top-k cutoff, 768 new tokens. No adapters, training, ledgers, hard sentence quotas or repairs. Small shifts in emphasis are allowed. This compares instruction strategies; it does not isolate author naming from instruction specificity.\n',
        'Gemma is unloaded before loading the pinned semantic verifier. Evaluation retains warnings and reports graded-objective proxies, copying, format, timing and per-case frontiers. Full verifier verdicts are saved incrementally and retained in metrics. Historical A output matches are reported. Qualitative style and true drift severity remain pending assessment; NLI is not a calibrated severity measurement.\n',
        'Download /kaggle/working/name_free_style_ablation_v1 after completion. Expect 24 entries in generations.private.jsonl, metrics.private.jsonl, verifier_results.private.jsonl, evaluation.json, runtime.json, run_status.json and manifest.json. This builder neither uploads nor runs the notebook.\n' ]},
        code('import os\nos.environ["CUDA_VISIBLE_DEVICES"] = "0"\nos.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"\nos.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"'),
        code('%pip install -q "transformers==5.20.1" "accelerate==1.14.0" "bitsandbytes==0.50.2" "huggingface-hub==1.33.0" "sentencepiece>=0.2"'),
        code(setup),code(check),code(run)],
        'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    path=ROOT/'kaggle/name_free_style.ipynb';path.write_text(json.dumps(book,ensure_ascii=False,indent=2)+'\n')
    print('Built kaggle/name_free_style.ipynb; no paragraph data, upload or execution.')


if __name__=='__main__':build()
