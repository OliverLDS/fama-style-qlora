"""Build a self-contained pilot notebook, with training disabled by default."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fama_style.training.plan_realizer import load_pack,runtime_paths,file_sha


def code(source):
    return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':[s+'\n' for s in source.rstrip().splitlines()]}


def build():
    pack=ROOT/'research_runs/plan_realizer_pilot_v1/input_pack/realizer_pack.json';digest=file_sha(pack);load_pack(pack,digest)
    paths=runtime_paths()+sorted((ROOT/'docs/claim_verifier_calibration_v3').glob('*.json'))
    files={str(p.relative_to(ROOT)):p.read_text() for p in paths}
    setup=('import json, sys\nfrom pathlib import Path\n'
        'runtime = Path("/kaggle/working/_plan_realizer_runtime")\n'
        'files = json.loads('+repr(json.dumps(files,ensure_ascii=False))+')\n'
        'for name, source in files.items():\n    path = runtime / name\n    path.parent.mkdir(parents=True, exist_ok=True)\n    path.write_text(source, encoding="utf-8")\n'
        'sys.path.insert(0, str(runtime / "src"))\n')
    check=('import importlib\nexperiment = importlib.import_module("fama_style.training.plan_realizer")\n'
        'matches = list(Path("/kaggle/input").rglob("realizer_pack.json"))\n'
        'if len(matches) != 1:\n    raise ValueError("Attach exactly one private realizer_pack.json")\n'
        'arguments = ["--pack", str(matches[0]), "--expected-pack-sha", '+repr(digest)+']\n'
        'experiment.main(arguments + ["--output-dir", "/kaggle/working/plan_realizer_preflight"])\n')
    run=('RUN_TRAINING = False  # Change only when you choose to authorize this bounded pilot.\n'
        'if RUN_TRAINING:\n    experiment.main(arguments + ["--execute-training", "--output-dir", "/kaggle/working/plan_realizer_pilot_v1"])\n'
        'else:\n    print("Prepared and CPU-checked only. Training is disabled; no model or Secret accessed.")\n')
    notes=['# Matched direct versus plan-conditioned supervised realization pilot\n',
        'PREPARED, NOT EXECUTED. Training is disabled in the final cell. Save & Run All with the default flag only performs CPU preflight. Change RUN_TRAINING to True only when authorizing the specified bounded pilot. No automatic continuation or full-corpus training.\n',
        'Upload as a new PRIVATE notebook. Attach a PRIVATE Dataset containing ONLY realizer_pack.json. For the authorized GPU run enable internet, T4 x2 and HF_TOKEN in Kaggle Secrets. Only GPU 0 is used. Never put credentials in the Dataset or notebook.\n',
        '12 existing retained-source training pairs across six papers, identical unchanged author targets. Train two fresh rank-8 q/v adapters from the same pinned Gemma 3 4B base and initial adapter state: direct and plan-conditioned. Four epochs each, 24 optimizer updates per arm, accumulation 2, LR 5e-5. Same shuffled example schedules and seed; planned prompts contain extra structural tokens. No trained adapter is inherited by the second arm.\n',
        'Plans contain only sentence count, per-sentence word bands and a semicolon budget. They are a bounded structural realization probe, not the full discovered taxonomy or learned rhetorical mapping. The same deterministic plan extractor is used for training and development. The prior LLM reference annotations validate coverage/count only. No reference wording is placed in prompts. Development plans come from references: this is an oracle diagnostic, not deployment.\n',
        'Four existing development-validation pairs selected by fixed feasible-plan/shortest-word-budget rules, plus two clearly labeled training probes. Generate base, direct adapter and planned adapter under BOTH prompt formats: 36 total greedy generations. The four development rows are shared by every condition. No reserved test examples. The old 247-row pilot is historical context; the newly matched direct arm is the causal control.\n',
        'Prompt labels are masked exactly; targets include their terminal chat tokens. No training/generation truncation. 1536 training-token ceiling and 4096 generation context ceiling. Gemma unloads before independent NLI. Save adapters, tensor hashes and verified reloads, full generation/verifier evidence, losses, costs and manifests. Empty/cutoff samples stay failed outputs; no repairs or retries. NLI and surface checks do not establish claim fidelity or author style.\n',
        'Download /kaggle/working/plan_realizer_pilot_v1/ after an authorized run. Lower loss is not the success criterion: compare style, claim-drift severity, copying and plan adherence separately. Minor drift remains eligible when substantial style gain justifies it. This small pilot may be underpowered; no full mapper is trained.\n']
    book={'cells':[{'cell_type':'markdown','metadata':{},'source':notes},
        code('import os\nos.environ["CUDA_VISIBLE_DEVICES"] = "0"\nos.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"\nos.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"'),
        code('%pip install -q "transformers==5.20.1" "peft==0.21.2" "accelerate==1.14.0" "bitsandbytes==0.50.2" "huggingface-hub==1.33.0" "sentencepiece>=0.2"'),
        code(setup),code(check),code(run)],'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    path=ROOT/'kaggle/plan_realizer_pilot.ipynb';path.write_text(json.dumps(book,indent=2,ensure_ascii=False)+'\n')
    print('Built plan_realizer_pilot.ipynb with training disabled. No upload, inference or training.')


if __name__=='__main__':build()
