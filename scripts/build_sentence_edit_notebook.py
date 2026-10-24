"""Export the gated 21-example inference notebook, without paragraph data."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fama_style.rhetoric.sentence_edit import load_pack,runtime_paths,file_sha


def code(source):return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':[line+'\n' for line in source.rstrip().splitlines()]}


def build():
    packpath=ROOT/'research_runs/sentence_preserving_edit_v1/input_pack/sentence_edit_pack.json'
    digest=file_sha(packpath);load_pack(packpath,digest)
    paths=runtime_paths()+sorted((ROOT/'docs/claim_verifier_calibration_v3').glob('*.json'))
    files={str(p.relative_to(ROOT)):p.read_text() for p in paths}
    setup=('import json, sys\nfrom pathlib import Path\n'
        'runtime = Path("/kaggle/working/_sentence_edit_runtime")\n'
        'files = json.loads('+repr(json.dumps(files,ensure_ascii=False))+')\n'
        'for name, source in files.items():\n    path = runtime / name\n    path.parent.mkdir(parents=True, exist_ok=True)\n    path.write_text(source, encoding="utf-8")\n'
        'sys.path.insert(0, str(runtime / "src"))\n')
    check=('import importlib\nexperiment = importlib.import_module("fama_style.rhetoric.sentence_edit")\n'
        'matches = list(Path("/kaggle/input").rglob("sentence_edit_pack.json"))\n'
        'if len(matches) != 1:\n    raise ValueError("Attach exactly one private sentence_edit_pack.json")\n'
        'arguments = ["--pack", str(matches[0]), "--expected-pack-sha", '+repr(digest)+']\n'
        'experiment.main(arguments + ["--output-dir", "/kaggle/working/sentence_edit_preflight"])\n')
    run='experiment.main(arguments + ["--execute-inference", "--output-dir", "/kaggle/working/sentence_preserving_edit_v1"])\n'
    book={'cells':[{'cell_type':'markdown','metadata':{},'source':[
        '# Source-sentence-preserving edits: four-generation diagnostic\n',
        'Create a new private notebook. Enable internet, GPU T4 x2 and HF_TOKEN in Kaggle Secrets; only GPU 0 is visible. Attach a private Dataset containing ONLY sentence_edit_pack.json. Keep all inputs and outputs private.\n',
        'Two exposed development cases. A repeats the previous sentence-evidence/no-grouping baseline exactly. B adds a JSON edit-script protocol: replacements keyed to source sentence IDs, with no insertion, deletion or reordering. The source sentence count is preserved by application. No target reference wording or target schema enters generation.\n',
        'Four first-attempt generations total. B produces both an all-proposed paragraph and a verifier-filtered paragraph from the same script. Rejected or unknown edits leave the original sentence untouched, and copying/fallback counts remain visible. Invalid scripts remain failures; no model repair calls. These are derived variants, not extra independent generations.\n',
        'Pinned base Gemma 3 4B, NF4/FP32, greedy decoding, 768 new tokens. No training or adapters. Verifier v3 runtime calibration runs before Secret access; NLI and Gemma load sequentially. Checks are fallible proxies, not semantic proof.\n',
        'Outputs under /kaggle/working/sentence_preserving_edit_v1/: evaluation.json, metrics.private.jsonl, edit_assessments.private.json, generations.private.jsonl, token_audit.json, run_status.json and manifest.json. No GPU job was submitted by this builder.\n']},
        code('import os\nos.environ["CUDA_VISIBLE_DEVICES"] = "0"\nos.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"\nos.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"'),
        code('%pip install -q "transformers==5.20.1" "accelerate==1.14.0" "bitsandbytes==0.50.2" "huggingface-hub==1.33.0" "sentencepiece>=0.2"'),
        code(setup),code(check),code(run)],
        'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    path=ROOT/'kaggle/sentence_preserving_edit.ipynb';path.write_text(json.dumps(book,ensure_ascii=False,indent=2)+'\n')
    print('Built kaggle/sentence_preserving_edit.ipynb; no paragraph data, upload or execution.')


if __name__=='__main__':build()
