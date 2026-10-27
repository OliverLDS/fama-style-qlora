"""Build a private-input, self-contained inference notebook without submission."""
import base64
import json
from pathlib import Path
import sys
import zlib
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fama_style.rhetoric.readable_packing import load_pack, runtime_paths, file_sha


def code(source):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [s + '\n' for s in source.rstrip().splitlines()]}


def build():
    pack = ROOT / 'research_runs/readable_packing_ablation_v1/input_pack_v3/readable_pack.json'
    digest = file_sha(pack); load_pack(pack, digest)
    files = {str(p.relative_to(ROOT)): p.read_text() for p in runtime_paths()}
    encoded = base64.b64encode(zlib.compress(json.dumps(files, ensure_ascii=False).encode('utf-8'), level=9)).decode('ascii')
    setup = ('import base64, json, sys, zlib\nfrom pathlib import Path\n'
             'runtime = Path("/kaggle/working/_readable_packing_runtime")\n'
             'files = json.loads(zlib.decompress(base64.b64decode(' + repr(encoded) + ')).decode("utf-8"))\n'
             'for name, source in files.items():\n    path = runtime / name\n    path.parent.mkdir(parents=True, exist_ok=True)\n    path.write_text(source, encoding="utf-8")\n'
             'sys.path.insert(0, str(runtime / "src"))\n')
    check = ('import importlib\nexperiment = importlib.import_module("fama_style.rhetoric.readable_packing")\n'
             'matches = list(Path("/kaggle/input").rglob("readable_pack.json"))\n'
             'if len(matches) != 1:\n    raise ValueError("Attach exactly one private readable_pack.json")\n'
             'arguments = ["--pack", str(matches[0]), "--expected-pack-sha", ' + repr(digest) + ']\n'
             'experiment.main(arguments + ["--output-dir", "/kaggle/working/readable_packing_preflight"])\n')
    run = 'experiment.main(arguments + ["--execute-inference", "--output-dir", "/kaggle/working/readable_packing_ablation_v1"])\n'
    notes = ['# Readable packing ablation: inference only\n',
             'Create a NEW PRIVATE notebook. Attach a PRIVATE Dataset containing ONLY readable_pack.json. No adapter or previous output attachment is required. Enable Internet, GPU T4 x2 and the HF_TOKEN Kaggle Secret. Only GPU 0 is used. Never put credentials in cells or data.\n',
             'Four first-attempt greedy generations: two retrospectively selected TRAINING paragraphs, each with frozen offset/JSON and readable bindings/direct-plan conditions. The cases are the hierarchy merge and market-efficiency finding/presumption split. No new annotations, development or reserved test data.\n',
             'A exactly reuses the earlier edit-guided prompt. B shows labeled source sentences, verbatim source claim/protected excerpts and the same edit slots as direct sentence instructions. Facts, claims, relationships, operation assignments, model and decoding stay fixed. Binding display and plan display change together; this does not isolate their separate effects. References and target wording are evaluation-only, never generator inputs.\n',
             'Frozen base Gemma 3 4B; pinned revisions, NF4/FP32 on one T4. 768 output tokens and 4096 total-token limit. All prompts are tokenizer-audited before Gemma weights load; no truncation. No training, adapters, retries, repairs or reference fallback. Empty/cutoff attempts remain failures.\n',
             'Gemma unloads before the smaller NLI model loads. Save full semantic and claim-localization evidence, copying, style proxies, tokens, runtime and paired results. Check whether the actual merge/split is realized; counts alone are insufficient. Minor claim drift can be acceptable for worthwhile style gains; binary NLI flags do not establish severity. This is a tiny oracle format diagnostic, not generalization or learned mapping.\n',
             'Download /kaggle/working/readable_packing_ablation_v1/ in full and keep it private. Share the notebook slug/version after completion. Phase messages and heartbeats show progress without credential values.\n']
    book = {'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source': notes},
                     code('import os\nos.environ["CUDA_VISIBLE_DEVICES"] = "0"\nos.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"\nos.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"'),
                     code('%pip install -q "transformers==5.20.1" "accelerate==1.14.0" "bitsandbytes==0.50.2" "huggingface-hub==1.33.0" "sentencepiece>=0.2"'),
                     code(setup), code(check), code(run)],
            'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}},
            'nbformat': 4, 'nbformat_minor': 5}
    serialized = json.dumps(book, indent=2, ensure_ascii=False) + '\n'
    if len(serialized.encode()) >= 900_000 or sum(len(''.join(c['source']).encode()) for c in book['cells']) >= 900_000:
        raise ValueError('Notebook exceeds Kaggle source-size budget')
    path = ROOT / 'kaggle/readable_packing.ipynb'; path.write_text(serialized)
    print('Built readable_packing.ipynb (' + str(path.stat().st_size) + ' bytes). No upload or execution.')


if __name__ == '__main__': build()
