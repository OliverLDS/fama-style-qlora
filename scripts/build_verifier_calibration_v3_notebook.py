"""Export synthetic verifier calibration; no private data, Secret or GPU submission."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def code(source):return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':[line+'\n' for line in source.rstrip().splitlines()]}


def build():
    paths=[ROOT/'src/fama_style/__init__.py',ROOT/'src/fama_style/prompting/__init__.py',ROOT/'src/fama_style/prompting/corpus.py']
    paths+=sorted((ROOT/'src/fama_style/research').glob('*.py'))+sorted((ROOT/'src/fama_style/rhetoric').glob('*.py'))
    paths+=sorted((ROOT/'docs/claim_verifier_calibration_v3').glob('*.json'))
    files={str(p.relative_to(ROOT)):p.read_text() for p in paths}
    setup=('import json, sys\nfrom pathlib import Path\n'
        'runtime = Path("/kaggle/working/_verifier_v3_runtime")\n'
        'files = json.loads('+repr(json.dumps(files,ensure_ascii=False))+')\n'
        'for name, source in files.items():\n    path = runtime / name\n    path.parent.mkdir(parents=True, exist_ok=True)\n    path.write_text(source, encoding="utf-8")\n'
        'sys.path.insert(0, str(runtime / "src"))\n')
    preflight=('import importlib\nexperiment = importlib.import_module("fama_style.research.claim_verifier_v3")\n'
        'experiment.main(["--output-dir", "/kaggle/working/verifier_preflight_v3"])\n')
    run='experiment.main(["--execute-calibration", "--output-dir", "/kaggle/working/verifier_calibration_v3"])\n'
    book={'cells':[{'cell_type':'markdown','metadata':{},'source':[
        '# Claim-verifier calibration v3: pretrained NLI inference only\n',
        'Use a new Kaggle notebook. Enable internet and GPU T4 x2; the first cell exposes GPU 0 only. No HF_TOKEN Secret, private dataset or other attachment is required. All embedded cases are synthetic.\n',
        'This loads only the pinned public DeBERTa NLI model. It performs no Gemma generation, fine-tuning, optimizer updates or structural-mapper training.\n',
        'All 120 exposed v2 cases are development-only. 48 fresh v3 controls were authored after freezing the revision. New scoped guards distinguish strict/inclusive numeric bounds and association from causal assertions. Unmatched or unsupported scopes abstain. The unchanged v2 verifier and global NLI remain baselines. Synthetic controls are not domain accuracy or semantic proof.\n',
        'The evaluation reports detection, false alarms, hard-paraphrase acceptance, false acceptances and abstentions. Thresholds and fixtures are frozen. The 21-example style comparison remains blocked pending verifier evidence.\n',
        'Outputs: /kaggle/working/verifier_calibration_v3/evaluation.json, results.json, runtime.json, run_status.json, manifest.json. No model-quality result has been measured by this builder.\n']},
        code('import os\nos.environ["CUDA_VISIBLE_DEVICES"] = "0"\nos.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"\nos.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"'),
        code('%pip install -q "transformers==5.20.1" "huggingface-hub==1.33.0" "sentencepiece>=0.2"'),
        code(setup),code(preflight),code(run)],
        'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    (ROOT/'kaggle/verifier_calibration_v3.ipynb').write_text(json.dumps(book,ensure_ascii=False,indent=2)+'\n')
    print('Built kaggle/verifier_calibration_v3.ipynb; synthetic cases only; no upload or execution.')


if __name__=='__main__':build()
