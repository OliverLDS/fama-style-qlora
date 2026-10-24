"""Export the new inference notebook only; embed code and an approved pack hash."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def cell(source):
    return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
            'source':[line+'\n' for line in source.rstrip().splitlines()]}


def build():
    pack=ROOT/'research_runs/concise_plan_v1/input_pack/concise_plan_pack.json'
    if not pack.exists():raise FileNotFoundError('First prepare the private concise_plan input pack')
    digest=hashlib.sha256(pack.read_bytes()).hexdigest()
    paths=[ROOT/'src/fama_style/__init__.py',ROOT/'src/fama_style/prompting/__init__.py',ROOT/'src/fama_style/prompting/corpus.py']
    paths+=sorted((ROOT/'src/fama_style/research').glob('*.py'))+sorted((ROOT/'src/fama_style/rhetoric').glob('*.py'))
    files={str(path.relative_to(ROOT/'src')):path.read_text() for path in paths}
    setup=('import json, sys\nfrom pathlib import Path\n'
        'runtime = Path("/kaggle/working/_claim_constrained_runtime/src")\n'
        'files = json.loads('+repr(json.dumps(files,ensure_ascii=False))+')\n'
        'for name, source in files.items():\n'
        '    path = runtime / name\n    path.parent.mkdir(parents=True, exist_ok=True)\n'
        '    path.write_text(source, encoding="utf-8")\n'
        'sys.path.insert(0, str(runtime))\n')
    check=('import importlib\n'
        'experiment = importlib.import_module("fama_style.rhetoric.claim_constrained")\n'
        'matches = list(Path("/kaggle/input").rglob("concise_plan_pack.json"))\n'
        'if len(matches) != 1:\n    raise ValueError("Attach exactly one private concise_plan_pack.json input")\n'
        'arguments = ["--pack", str(matches[0]), "--expected-pack-sha", '+repr(digest)+']\n'
        'experiment.main(arguments + ["--output-dir", "/kaggle/working/claim_constrained_preflight"])\n')
    run='experiment.main(arguments + ["--execute-inference", "--output-dir", "/kaggle/working/claim_constrained_realization_v1"])\n'
    book={'cells':[
        {'cell_type':'markdown','metadata':{},'source':[
            '# Claim-constrained realization: inference only\n',
            'Create a new private notebook. Enable internet, HF_TOKEN Secret, and GPU T4 x2; only GPU 0 is exposed. Attach the existing private concise_plan_pack.json dataset without changing its bytes.\n',
            'Four previously inspected training pairs, sixteen first attempts: sentence-count paragraph; same paragraph with source-only preservation ledger; sentence-count JSON slots; same slots with the identical ledger. All conditions receive the same target count. No reference wording enters prompts.\n',
            'Base Gemma 3 4B, pinned revision, NF4, greedy decoding, 768 new tokens per output. No adapter, optimizer, mapper training or fine-tuning. No repair attempts. Exactly one complete outer JSON fence may be removed; raw and normalized compliance are both reported.\n',
            'The ledger copies source sentence evidence and flags surface cues. It is not an independently verified claim parse. Independent NLI runs after unloading Gemma; targeted mutation controls test checker sensitivity. No score proves claim preservation. Reserved test data are unused.\n',
            'Keep all outputs private under /kaggle/working/claim_constrained_realization_v1/. Inspect evaluation.json, metrics.private.jsonl, generations.private.jsonl and run_status.json. Nothing has been uploaded or submitted by this builder.\n']},
        cell('import os\nos.environ["CUDA_VISIBLE_DEVICES"] = "0"\nos.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"'),
        cell('%pip install -q "transformers==5.20.1" "accelerate==1.14.0" "bitsandbytes==0.50.2" "huggingface-hub==1.33.0" "sentencepiece>=0.2"'),
        cell(setup),cell(check),cell(run)],
        'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},
        'nbformat':4,'nbformat_minor':5}
    path=ROOT/'kaggle/claim_constrained_realization.ipynb';path.write_text(json.dumps(book,ensure_ascii=False,indent=2)+'\n')
    print('Built kaggle/claim_constrained_realization.ipynb; code/hash only, no execution or upload')


if __name__=='__main__':build()
