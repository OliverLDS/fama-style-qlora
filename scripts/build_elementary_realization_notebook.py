"""Build a private-input, self-contained inference notebook without submission."""
import base64
import json
from pathlib import Path
import sys
import zlib
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fama_style.rhetoric.elementary_realization import load_pack, runtime_paths, file_sha


def code(source):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': [s + '\n' for s in source.rstrip().splitlines()]}


def build():
    pack = ROOT / 'research_runs/elementary_packing_realization_v1/input_pack/elementary_pack.json'
    digest = file_sha(pack); load_pack(pack, digest)
    files = {str(p.relative_to(ROOT)): p.read_text() for p in runtime_paths()}
    encoded = base64.b64encode(zlib.compress(json.dumps(files, ensure_ascii=False).encode('utf-8'), level=9)).decode('ascii')
    setup = ('import base64, json, sys, zlib\nfrom pathlib import Path\n'
             'runtime = Path("/kaggle/working/_elementary_runtime")\n'
             'files = json.loads(zlib.decompress(base64.b64decode(' + repr(encoded) + ')).decode("utf-8"))\n'
             'for name, source in files.items():\n    path = runtime / name\n    path.parent.mkdir(parents=True, exist_ok=True)\n    path.write_text(source, encoding="utf-8")\n'
             'sys.path.insert(0, str(runtime / "src"))\n')
    check = ('import importlib\nexperiment = importlib.import_module("fama_style.rhetoric.elementary_realization")\n'
             'matches = list(Path("/kaggle/input").rglob("elementary_pack.json"))\n'
             'if len(matches) != 1:\n    raise ValueError("Attach exactly one private elementary_pack.json")\n'
             'arguments = ["--pack", str(matches[0]), "--expected-pack-sha", ' + repr(digest) + ']\n'
             'experiment.main(arguments + ["--output-dir", "/kaggle/working/elementary_preflight"])\n')
    run = 'experiment.main(arguments + ["--execute-inference", "--output-dir", "/kaggle/working/elementary_packing_realization_v1"])\n'
    notes = ['# Elementary packing edits: inference-only check\n',
             'Create a NEW PRIVATE notebook. Upload this notebook and attach a PRIVATE Dataset containing ONLY elementary_pack.json. No previous notebook output or adapter is required. Enable Internet, GPU T4 x2 and the HF_TOKEN Kaggle Secret. Only GPU 0 is exposed. Never paste the token into a cell or Dataset.\n',
             '12 first-attempt greedy generations on frozen base Gemma 3 4B: two conditions on six previously annotated training paragraphs from six papers. Five new oracle candidates and one reserved diagnostic probe are reported separately. The split case and reserved probe are mandatory; other papers use the shortest compiled prompt. No selection used model outputs or development/test data.\n',
             'A: concrete name-free style instruction plus source evidence. B: exactly the same instruction and evidence plus reference-derived sentence-packing edits. Claim IDs and grouping are oracle annotation information; this is neither a learned mapper nor deployment/generalization performance. References and target wording are evaluation-only, never generator inputs.\n',
             'NF4/FP32 on one T4, pinned Gemma/NLI revisions, 768 output tokens and a 4096 total-token budget. The tokenizer audits every prompt before Gemma weights load; overflow fails without truncation. All model parameters are frozen. No training, adapter, retries, generation repairs or reference fallback. Empty and cutoff outputs remain failures.\n',
             'After generation Gemma unloads before the smaller NLI model loads. Save full semantic diagnostics, source-claim localization matrices, copying, surface style proxies and paired organization results. NLI is fallible, sentence-count matches are not full adherence, and surface style distance is not an author classifier. Minor drift may justify meaningful style gain; binary verifier status does not establish severity or acceptability.\n',
             'Download the complete elementary_packing_realization_v1 output folder. Keep generations, references and claim evidence PRIVATE. Share only the notebook slug/version for analysis. Phase messages and 30-second heartbeats show progress without revealing credentials.\n']
    book = {'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source': notes},
                     code('import os\nos.environ["CUDA_VISIBLE_DEVICES"] = "0"\nos.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"\nos.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"'),
                     code('%pip install -q "transformers==5.20.1" "accelerate==1.14.0" "bitsandbytes==0.50.2" "huggingface-hub==1.33.0" "sentencepiece>=0.2"'),
                     code(setup), code(check), code(run)],
            'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}},
            'nbformat': 4, 'nbformat_minor': 5}
    serialized = json.dumps(book, indent=2, ensure_ascii=False) + '\n'
    if len(serialized.encode()) >= 900_000 or sum(len(''.join(c['source']).encode()) for c in book['cells']) >= 900_000:
        raise ValueError('Notebook exceeds Kaggle source-size budget')
    path = ROOT / 'kaggle/elementary_realization.ipynb'; path.write_text(serialized)
    print('Built elementary_realization.ipynb (' + str(path.stat().st_size) + ' bytes). No upload or execution.')


if __name__ == '__main__': build()
