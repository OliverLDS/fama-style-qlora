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
        'runtime = Path("/kaggle/working/_sentence_slot_runtime/src")\n'
        'files = json.loads('+repr(json.dumps(files,ensure_ascii=False))+')\n'
        'for name, source in files.items():\n'
        '    path = runtime / name\n    path.parent.mkdir(parents=True, exist_ok=True)\n'
        '    path.write_text(source, encoding="utf-8")\n'
        'sys.path.insert(0, str(runtime))\n')
    check=('import importlib\n'
        'experiment = importlib.import_module("fama_style.rhetoric.sentence_slots")\n'
        'matches = list(Path("/kaggle/input").rglob("concise_plan_pack.json"))\n'
        'if len(matches) != 1:\n    raise ValueError("Attach exactly one private concise_plan_pack.json input")\n'
        'arguments = ["--pack", str(matches[0]), "--expected-pack-sha", '+repr(digest)+']\n'
        'experiment.main(arguments + ["--output-dir", "/kaggle/working/sentence_slot_preflight"])\n')
    run='experiment.main(arguments + ["--execute-inference", "--output-dir", "/kaggle/working/sentence_slot_realization_v1"])\n'
    book={'cells':[
        {'cell_type':'markdown','metadata':{},'source':[
            '# Count-only versus sentence-slot realization: inference only\n',
            'Use a **new private notebook** with internet and GPU T4 x2. Only GPU 0 is exposed. Enable HF_TOKEN in Kaggle Secrets. Attach the existing private concise_plan_pack.json dataset; the input bytes are unchanged.\n',
            'Four previously inspected training pairs; twelve first-attempt generations. A: ordinary prompt. B: same prompt plus sentence count only. C: same count control plus a JSON array with one sentence per slot. No length bands, clause hints or semicolon budgets are supplied. No reference wording, adapters, optimizer or training.\n',
            'All conditions share pinned base Gemma 3 4B, greedy decoding and a 768-token output budget. This is a training-only oracle diagnostic, not held-out performance. No repair attempts or format salvage. Invalid formats remain failures. Sentence checks and NLI are imperfect proxies.\n',
            'Gemma is unloaded before independent GPU NLI. Outputs remain private under /kaggle/working/sentence_slot_realization_v1/. Main files: evaluation.json, metrics.private.jsonl, generations.private.jsonl and run_status.json. This notebook has not been submitted by its builder.\n']},
        cell('import os\nos.environ["CUDA_VISIBLE_DEVICES"] = "0"\nos.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"'),
        cell('%pip install -q "transformers==5.20.1" "accelerate==1.14.0" "bitsandbytes==0.50.2" "huggingface-hub==1.33.0" "sentencepiece>=0.2"'),
        cell(setup),cell(check),cell(run)],
        'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},
        'nbformat':4,'nbformat_minor':5}
    path=ROOT/'kaggle/sentence_slot_realization.ipynb';path.write_text(json.dumps(book,ensure_ascii=False,indent=2)+'\n')
    print('Built kaggle/sentence_slot_realization.ipynb; code/hash only, no execution or upload')


if __name__=='__main__':build()
