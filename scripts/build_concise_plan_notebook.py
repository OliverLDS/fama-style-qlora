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
        'runtime = Path("/kaggle/working/_concise_plan_runtime/src")\n'
        'files = json.loads('+repr(json.dumps(files,ensure_ascii=False))+')\n'
        'for name, source in files.items():\n'
        '    path = runtime / name\n    path.parent.mkdir(parents=True, exist_ok=True)\n'
        '    path.write_text(source, encoding="utf-8")\n'
        'sys.path.insert(0, str(runtime))\n')
    check=('import importlib\n'
        'experiment = importlib.import_module("fama_style.rhetoric.concise_plan")\n'
        'matches = list(Path("/kaggle/input").rglob("concise_plan_pack.json"))\n'
        'if len(matches) != 1:\n    raise ValueError("Attach exactly one private concise_plan_pack.json input")\n'
        'arguments = ["--pack", str(matches[0]), "--expected-pack-sha", '+repr(digest)+']\n'
        'experiment.main(arguments + ["--output-dir", "/kaggle/working/concise_plan_preflight"])\n')
    run='experiment.main(arguments + ["--execute-inference", "--output-dir", "/kaggle/working/concise_plan_realization_v1"])\n'
    book={'cells':[
        {'cell_type':'markdown','metadata':{},'source':[
            '# Concise target-plan realization: inference only\n',
            'Use a **new private notebook**. Upload the private `concise_plan_pack.json` as a private Kaggle Dataset and attach it. Enable the `HF_TOKEN` Secret, internet, and GPU T4 x2. Only GPU 0 is visible.\n',
            'Four frozen training pairs, two conditions, eight generations: ordinary instruction versus the same instruction plus a short reference-derived presentation plan. No reference wording is supplied to Gemma. No adapters or training are used. This is an oracle diagnostic, not deployable evaluation.\n',
            'The reference plan contains count, length-band, coarse clause and punctuation controls. No claim IDs are guessed across texts. Existing proposed repairs are not substituted into training data.\n',
            'Gemma is unloaded before an independent NLI model runs on the same GPU. NLI sanity controls and source/reference fidelity checks are reported. Scores remain proxies. Do not attach test data.\n',
            'Outputs: `/kaggle/working/concise_plan_realization_v1/evaluation.json`, `generations.private.jsonl`, `metrics.private.jsonl`, `nli_controls.json`, `token_audit.json`, and `run_status.json`. Keep outputs private. No upload or execution has been performed by this builder.\n']},
        cell('import os\nos.environ["CUDA_VISIBLE_DEVICES"] = "0"\nos.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"'),
        cell('%pip install -q "transformers==5.20.1" "accelerate==1.14.0" "bitsandbytes==0.50.2" "huggingface-hub==1.33.0" "sentencepiece>=0.2"'),
        cell(setup),cell(check),cell(run)],
        'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},
        'nbformat':4,'nbformat_minor':5}
    path=ROOT/'kaggle/concise_plan_realization.ipynb';path.write_text(json.dumps(book,ensure_ascii=False,indent=2)+'\n')
    print('Built kaggle/concise_plan_realization.ipynb; code/hash only, no execution or upload')


if __name__=='__main__':build()
