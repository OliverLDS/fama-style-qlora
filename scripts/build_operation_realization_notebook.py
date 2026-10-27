"""Build the self-contained, inference-only grounded-operation comparison."""
import base64
import json
import zlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fama_style.rhetoric.operation_realization import load_pack,runtime_paths,file_sha


def code(source):
    return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':[s+'\n' for s in source.rstrip().splitlines()]}


def build():
    pack=ROOT/'research_runs/grounded_operation_realization_v1/input_pack/operation_pack.json';digest=file_sha(pack);load_pack(pack,digest)
    paths=runtime_paths()+sorted((ROOT/'docs/claim_verifier_calibration_v3').glob('*.json'))
    files={str(p.relative_to(ROOT)):p.read_text() for p in paths}
    encoded=base64.b64encode(zlib.compress(json.dumps(files,ensure_ascii=False).encode('utf-8'),level=9)).decode('ascii')
    setup=('import base64, json, sys, zlib\nfrom pathlib import Path\n'
        'runtime = Path("/kaggle/working/_operation_runtime")\n'
        'files = json.loads(zlib.decompress(base64.b64decode('+repr(encoded)+')).decode("utf-8"))\n'
        'for name, source in files.items():\n    path = runtime / name\n    path.parent.mkdir(parents=True, exist_ok=True)\n    path.write_text(source, encoding="utf-8")\n'
        'sys.path.insert(0, str(runtime / "src"))\n')
    check=('import importlib\nexperiment = importlib.import_module("fama_style.rhetoric.operation_realization")\n'
        'matches = list(Path("/kaggle/input").rglob("operation_pack.json"))\n'
        'if len(matches) != 1:\n    raise ValueError("Attach exactly one private operation_pack.json")\n'
        'arguments = ["--pack", str(matches[0]), "--expected-pack-sha", '+repr(digest)+', \"--adapter-root\", \"/kaggle/input\"]\n'
        'experiment.main(arguments + ["--output-dir", "/kaggle/working/operation_preflight"])\n')
    run='experiment.main(arguments + ["--execute-inference", "--output-dir", "/kaggle/working/grounded_operation_realization_v1"])\n'
    notes=['# Grounded rhetorical-operation realization: inference only\n',
        'NEW PRIVATE notebook. Attach a PRIVATE Dataset containing ONLY operation_pack.json, plus the outputs of oliverlds/plan_realizer_pilot version 1 (must include adapter_planned/adapter_config.json and adapter_model.safetensors). Enable internet, GPU T4 x2 and HF_TOKEN in Kaggle Secrets. Only GPU 0 is used. No credentials in files.\n',
        '30 first-attempt greedy calls: base Gemma 3 4B and the frozen planned adapter, each under ordinary, shallow count/length plan and grounded-operation conditions on the same five paragraphs. The four transformation cases and one already-parallel preservation control are reported separately. ALL FIVE WERE TRAINING EXAMPLES SEEN BY THE ADAPTER. No development/test rows, new training, adapter updates or learned mapping.\n',
        'All conditions share byte-identical source claim evidence, supplemental facts and existing relationship evidence. Ordinary therefore means no additional plan, with this common source scaffold. This differs from earlier pilot prompts; no exact historical generation reproduction is asserted. Only the plan guidance changes. The reference wording/annotations/cues are never passed to the generator. Plans are reference-informed oracle side information.\n',
        'The saved adapter was trained on shallow plans, not these new operations or the shared evidence scaffold. Its operation response is an out-of-format realization diagnostic, not evaluation of an operation-trained adapter or generalization. Adapter files and loaded tensor hashes are checked against the reviewed pilot. No optimizer or backward pass exists here.\n',
        'Pinned base/NLI revisions, NF4/FP32 on one T4, 512 output-token cap and 4096 total context limit. Audit all rendered prompts before loading weights; never truncate. Empty/cutoff responses stay failures and remaining calls continue; format issues remain verbatim. No retries, repairs or reference fallback. Gemma unloads before NLI.\n',
        'Save full semantic verdicts and claim-to-output-sentence NLI matrices. Location hypotheses cannot prove operation adherence, resolve every composite claim, or establish author style. Assess claim drift, meaningful style change, copying and rhetorical realization separately, with the existing allowance for minor drift. A preservation-control match is not a style success.\n',
        'Download /kaggle/working/grounded_operation_realization_v1/: raw generations, jobs, metrics, full verifier/localization evidence, evaluation, token audit, adapter identity, runtime, run status and manifest. No GPU run was submitted by the builder.\n']
    book={'cells':[{'cell_type':'markdown','metadata':{},'source':notes},
        code('import os\nos.environ["CUDA_VISIBLE_DEVICES"] = "0"\nos.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"\nos.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"'),
        code('%pip install -q "transformers==5.20.1" "peft==0.21.2" "accelerate==1.14.0" "bitsandbytes==0.50.2" "huggingface-hub==1.33.0" "sentencepiece>=0.2"'),
        code(setup),code(check),code(run)],'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    serialized=json.dumps(book,indent=2,ensure_ascii=False)+'\n'
    source_bytes=sum(len(''.join(cell['source']).encode('utf-8')) for cell in book['cells'])
    if len(serialized.encode('utf-8'))>=900_000 or source_bytes>=900_000:raise ValueError('Notebook exceeds conservative Kaggle source-size budget')
    path=ROOT/'kaggle/operation_realization.ipynb';path.write_text(serialized,encoding='utf-8')
    print('Built operation_realization.ipynb ('+str(len(serialized.encode('utf-8')))+' bytes); unchanged runtime and pack, no upload or execution.')


if __name__=='__main__':build()
