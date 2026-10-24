"""Export the gated 21-example inference notebook, without paragraph data."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fama_style.rhetoric.soft_guidance_experiment import load_pack,runtime_paths,file_sha


def code(source):return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':[line+'\n' for line in source.rstrip().splitlines()]}


def build():
    packpath=ROOT/'research_runs/soft_guidance_validation_v2/input_pack/soft_guidance_pack.json'
    digest=file_sha(packpath);load_pack(packpath,digest)
    paths=runtime_paths()+sorted((ROOT/'docs/claim_verifier_calibration_v3').glob('*.json'))
    files={str(p.relative_to(ROOT)):p.read_text() for p in paths}
    setup=('import json, sys\nfrom pathlib import Path\n'
        'runtime = Path("/kaggle/working/_soft_guidance_runtime")\n'
        'files = json.loads('+repr(json.dumps(files,ensure_ascii=False))+')\n'
        'for name, source in files.items():\n    path = runtime / name\n    path.parent.mkdir(parents=True, exist_ok=True)\n    path.write_text(source, encoding="utf-8")\n'
        'sys.path.insert(0, str(runtime / "src"))\n')
    check=('import importlib\nexperiment = importlib.import_module("fama_style.rhetoric.soft_guidance_experiment")\n'
        'matches = list(Path("/kaggle/input").rglob("soft_guidance_pack.json"))\n'
        'if len(matches) != 1:\n    raise ValueError("Attach exactly one private soft_guidance_pack.json")\n'
        'arguments = ["--pack", str(matches[0]), "--expected-pack-sha", '+repr(digest)+']\n'
        'experiment.main(arguments + ["--output-dir", "/kaggle/working/soft_guidance_preflight_v2"])\n')
    run='experiment.main(arguments + ["--execute-inference", "--output-dir", "/kaggle/working/soft_guidance_validation_v2"])\n'
    book={'cells':[{'cell_type':'markdown','metadata':{},'source':[
        '# Soft rhetorical guidance: 21-example inference comparison\n',
        'Use a new private notebook. Enable internet, GPU T4 x2 and the HF_TOKEN Kaggle Secret. Only GPU 0 is exposed. Attach a private Dataset containing ONLY the new soft_guidance_pack.json input. It is different from the old four-example pack. Keep all inputs and outputs private.\n',
        '42 first-attempt generations: A uses source-only claim guidance; B uses the identical guidance plus an optional source-bound plan derived from training-reference sentence-length evidence. Sentence count is free. This is a fixed soft prior, not a learned full-taxonomy mapper. No target wording or target schema enters the generator.\n',
        'The passed v3 verifier calibration is repeated as a runtime canary before Secret access and Gemma loading. If it fails, generation does not start. The public NLI model is unloaded before pinned base Gemma 3 4B loads in NF4 with FP32 compute. No adapters, optimizer or training.\n',
        'After generation, Gemma is unloaded and the same pinned NLI model assesses all pairs and outputs. Unknowns and malformed formats remain reported. Verifier consistency and style distance are imperfect proxies, not semantic proof or authorship accuracy. No reserved test examples are used.\n',
        'Output: /kaggle/working/soft_guidance_validation_v2/evaluation.json, generations.private.jsonl, metrics.private.jsonl, nli_probability_cache.private.jsonl, token_audit.json, run_status.json and manifest.json. No GPU job has been submitted by this builder.\n']},
        code('import os\nos.environ["CUDA_VISIBLE_DEVICES"] = "0"\nos.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"\nos.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"'),
        code('%pip install -q "transformers==5.20.1" "accelerate==1.14.0" "bitsandbytes==0.50.2" "huggingface-hub==1.33.0" "sentencepiece>=0.2"'),
        code(setup),code(check),code(run)],
        'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    path=ROOT/'kaggle/soft_guidance_validation.ipynb';path.write_text(json.dumps(book,ensure_ascii=False,indent=2)+'\n')
    print('Built kaggle/soft_guidance_validation.ipynb; no paragraph data, upload or execution.')


if __name__=='__main__':build()
