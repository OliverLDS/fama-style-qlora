"""Export the 27-call minimal antecedent diagnostic without private text."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fama_style.rhetoric.minimal_antecedent import load_pack,runtime_paths,file_sha


def code(source):return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':[line+'\n' for line in source.rstrip().splitlines()]}


def build():
    packpath=ROOT/'research_runs/minimal_antecedent_v1/input_pack/antecedent_pack.json'
    digest=file_sha(packpath);load_pack(packpath,digest)
    paths=runtime_paths()+sorted((ROOT/'docs/claim_verifier_calibration_v3').glob('*.json'))
    files={str(p.relative_to(ROOT)):p.read_text() for p in paths}
    setup=('import json, sys\nfrom pathlib import Path\n'
        'runtime = Path("/kaggle/working/_minimal_antecedent_runtime")\n'
        'files = json.loads('+repr(json.dumps(files,ensure_ascii=False))+')\n'
        'for name, source in files.items():\n    path = runtime / name\n    path.parent.mkdir(parents=True, exist_ok=True)\n    path.write_text(source, encoding="utf-8")\n'
        'sys.path.insert(0, str(runtime / "src"))\n')
    check=('import importlib\nexperiment = importlib.import_module("fama_style.rhetoric.minimal_antecedent")\n'
        'matches = list(Path("/kaggle/input").rglob("antecedent_pack.json"))\n'
        'if len(matches) != 1:\n    raise ValueError("Attach exactly one private antecedent_pack.json")\n'
        'arguments = ["--pack", str(matches[0]), "--expected-pack-sha", '+repr(digest)+']\n'
        'experiment.main(arguments + ["--output-dir", "/kaggle/working/minimal_antecedent_preflight"])\n')
    run='experiment.main(arguments + ["--execute-inference", "--output-dir", "/kaggle/working/minimal_antecedent"])\n'
    book={'cells':[{'cell_type':'markdown','metadata':{},'source':[
        '# Minimal antecedent expansion: 27 inference calls\n',
        'Upload as a NEW PRIVATE notebook. Attach a PRIVATE Dataset containing ONLY antecedent_pack.json. Enable internet, GPU T4 x2 and HF_TOKEN in Kaggle Secrets. Only GPU 0 is used; token values are never printed.\n',
        'Three training groups, three seeds, three conditions: A exact historical local-only controls; B exact historical read-only full-context controls; C the A prompt with source-supported actors and cohort made explicit only inside its assigned group. No surrounding passage in C; no target wording in any prompt. All conditions keep the same sampling seeds and original group token caps. Rotation balances condition positions within triples.\n',
        'C expansions are deterministic, case-specific antecedent resolutions from the plain-English source, with saved literal edits and source-span provenance. This is not a learned resolver or new training data. Both dividend groups make the never-payer cohort explicit; the numeric group therefore tests scoped realization, not an unchanged input. Numbers and explicit comparisons are retained.\n',
        'Base Gemma 3 4B, pinned NF4/FP32 setup; no adapter, training, retries, repairs or paragraph assembly. Empty/cutoff samples stay failures. Format violations stay verbatim. All outputs retain local-group and expanded-group verifier evidence plus full-context forward NLI; these are uncalibrated diagnostics, not semantic gates. Style and drift must be assessed separately with the existing minor-drift allowance.\n',
        'Download /kaggle/working/minimal_antecedent/: evaluation.json, generations.private.jsonl, metrics.private.jsonl, verifier_results.private.jsonl, jobs.private.jsonl, token_audit.json, runtime.json, run_status.json, manifest.json. All 18 A/B controls should reproduce previous output hashes before interpreting C. No inference was submitted by this builder.\n']},
        code('import os\nos.environ["CUDA_VISIBLE_DEVICES"] = "0"\nos.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"\nos.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"'),
        code('%pip install -q "transformers==5.20.1" "accelerate==1.14.0" "bitsandbytes==0.50.2" "huggingface-hub==1.33.0" "sentencepiece>=0.2"'),
        code(setup),code(check),code(run)],
        'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    path=ROOT/'kaggle/minimal_antecedent.ipynb';path.write_text(json.dumps(book,ensure_ascii=False,indent=2)+'\n')
    print('Built kaggle/minimal_antecedent.ipynb; no paragraph data, upload or execution.')


if __name__=='__main__':build()
