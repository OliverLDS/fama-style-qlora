"""Export the gated 21-example inference notebook, without paragraph data."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fama_style.rhetoric.ledger_ablation import load_pack,runtime_paths,file_sha


def code(source):return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':[line+'\n' for line in source.rstrip().splitlines()]}


def build():
    packpath=ROOT/'research_runs/ledger_grouping_ablation_v1/input_pack/ledger_ablation_pack.json'
    digest=file_sha(packpath);load_pack(packpath,digest)
    paths=runtime_paths()+sorted((ROOT/'docs/claim_verifier_calibration_v3').glob('*.json'))
    files={str(p.relative_to(ROOT)):p.read_text() for p in paths}
    setup=('import json, sys\nfrom pathlib import Path\n'
        'runtime = Path("/kaggle/working/_ledger_ablation_runtime")\n'
        'files = json.loads('+repr(json.dumps(files,ensure_ascii=False))+')\n'
        'for name, source in files.items():\n    path = runtime / name\n    path.parent.mkdir(parents=True, exist_ok=True)\n    path.write_text(source, encoding="utf-8")\n'
        'sys.path.insert(0, str(runtime / "src"))\n')
    check=('import importlib\nexperiment = importlib.import_module("fama_style.rhetoric.ledger_ablation")\n'
        'matches = list(Path("/kaggle/input").rglob("ledger_ablation_pack.json"))\n'
        'if len(matches) != 1:\n    raise ValueError("Attach exactly one private ledger_ablation_pack.json")\n'
        'arguments = ["--pack", str(matches[0]), "--expected-pack-sha", '+repr(digest)+']\n'
        'experiment.main(arguments + ["--output-dir", "/kaggle/working/ledger_ablation_preflight"])\n')
    run='experiment.main(arguments + ["--execute-inference", "--output-dir", "/kaggle/working/ledger_grouping_ablation_v1"])\n'
    book={'cells':[{'cell_type':'markdown','metadata':{},'source':[
        '# Ledger format x grouping guidance: eight-generation ablation\n',
        'Create a new private notebook. Enable internet, GPU T4 x2 and HF_TOKEN in Kaggle Secrets. Only GPU 0 is visible. Attach a private Dataset containing ONLY ledger_ablation_pack.json. Keep all inputs and outputs private.\n',
        'Two already-observed development failures, four conditions each: sentence-only evidence without/with the original grouping plan; cue-rich evidence without/with that same plan. Removing protected_evidence fields is the only ledger change. Rich controls exactly repeat the prior prompts. Sentence count stays free.\n',
        'Pinned base Gemma 3 4B, NF4/FP32, greedy decoding, 768 new tokens. No adapters, training or repairs. This is a post-hoc mechanism diagnostic, not fresh validation or a learned taxonomy mapper. No reference text is given to Gemma.\n',
        'The passed v3 verifier calibration is repeated before Gemma Secret access. NLI is unloaded before generation, then reloaded after Gemma unload. Evaluation reports cue artifacts, sentence inflation, old-control output hashes, semantics, copying, style and cost separately. All are imperfect diagnostic proxies.\n',
        'Outputs: /kaggle/working/ledger_grouping_ablation_v1/evaluation.json, generations.private.jsonl, metrics.private.jsonl, token_audit.json, run_status.json and manifest.json. Nothing was submitted by this builder.\n']},
        code('import os\nos.environ["CUDA_VISIBLE_DEVICES"] = "0"\nos.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"\nos.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"'),
        code('%pip install -q "transformers==5.20.1" "accelerate==1.14.0" "bitsandbytes==0.50.2" "huggingface-hub==1.33.0" "sentencepiece>=0.2"'),
        code(setup),code(check),code(run)],
        'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    path=ROOT/'kaggle/ledger_grouping_ablation.ipynb';path.write_text(json.dumps(book,ensure_ascii=False,indent=2)+'\n')
    print('Built kaggle/ledger_grouping_ablation.ipynb; no paragraph data, upload or execution.')


if __name__=='__main__':build()
