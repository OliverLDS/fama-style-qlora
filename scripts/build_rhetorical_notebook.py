"""Build only the new oracle notebook; existing experiments remain byte-identical."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def code(source):
    return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
            'source':[line+'\n' for line in source.rstrip().splitlines()]}


def build():
    paths = [ROOT/'src/fama_style/__init__.py', ROOT/'src/fama_style/prompting/__init__.py',
             ROOT/'src/fama_style/prompting/corpus.py']
    paths += sorted((ROOT/'src/fama_style/research').glob('*.py'))
    paths += sorted((ROOT/'src/fama_style/rhetoric').glob('*.py'))
    files = {str(path.relative_to(ROOT/'src')):path.read_text(encoding='utf-8') for path in paths}
    # Embedded files contain implementation only. No paragraph data or results.
    setup = ('import json, sys\nfrom pathlib import Path\n'
             'runtime = Path("/kaggle/working/_rhetorical_runtime/src")\n'
             'files = json.loads('+repr(json.dumps(files,ensure_ascii=False))+')\n'
             'for name, source in files.items():\n'
             '    path = runtime / name\n    path.parent.mkdir(parents=True, exist_ok=True)\n'
             '    path.write_text(source, encoding="utf-8")\n'
             'sys.path.insert(0, str(runtime))\n')
    run = ('import importlib\n'
           'oracle_runtime = importlib.import_module("fama_style.rhetoric.experiment")\n'
           'train = list(Path("/kaggle/input").rglob("train.jsonl"))\n'
           'validation = list(Path("/kaggle/input").rglob("validation.jsonl"))\n'
           'if len(train) != 1 or len(validation) != 1:\n'
           '    raise ValueError("Attach exactly the frozen corpus train.jsonl and validation.jsonl split files")\n'
           'oracle_runtime.main(["--train", str(train[0]), "--validation", str(validation[0]),\n'
           '      "--execute-inference", "--output-dir", "/kaggle/working/rhetorical_oracle_v1"])\n')
    book = {'cells':[
        {'cell_type':'markdown','metadata':{},'source':[
            '# Rhetorical oracle: inference only\n',
            'Keep this notebook and outputs private. Attach the frozen corpus v1 `train.jsonl` and `validation.jsonl` split files; do not attach the combined corpus or test file. Enable the `HF_TOKEN` Secret and internet, select GPU T4 x2. Only GPU 0 is exposed.\n',
            'This runs 63 greedy Gemma 3 4B generations: ordinary prompt, source plan, and reference-derived oracle plan, across the same 21 development examples. It performs no training. No reference wording is sent to generation. Oracle C is diagnostic, not deployable.\n',
            'After generation, Gemma is unloaded and an independent NLI model runs on CPU with synthetic sanity controls. Automatic semantic scores remain proxies. Budget 63 generations plus download and CPU evaluation time; progress is printed every 30 seconds during slow operations.\n',
            'Results: `/kaggle/working/rhetorical_oracle_v1/inference/evaluation.json`, `generations.private.jsonl`, `metrics.private.jsonl`, `nli_controls.json`, and `run_status.json`. Public code contains no private paragraphs.\n']},
        code('import os\nos.environ["CUDA_VISIBLE_DEVICES"] = "0"\nos.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"'),
        code('%pip install -q "transformers==5.20.1" "accelerate>=1.0" "bitsandbytes>=0.46" "sentencepiece>=0.2"'),
        code(setup), code(run)],
        'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},
        'nbformat':4,'nbformat_minor':5}
    destination = ROOT/'kaggle/rhetorical_oracle.ipynb'
    destination.write_text(json.dumps(book,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Built kaggle/rhetorical_oracle.ipynb (no execution or upload)')


if __name__=='__main__': build()
