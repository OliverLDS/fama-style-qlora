"""Build the approved matched-input SFT notebook for manual Kaggle execution."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def code(text):
    return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
            'source':text.rstrip().splitlines(keepends=True)}


def validate_authorization(plan,approved,source,packet_manifest_sha):
    expected={'protocol':'input_replacement_execution_v1','user_authorized':True,'training_rows':46,
        'development_rows':8,'replaced_training_inputs':12,'epochs':2,'optimizer_steps':46,
        'fresh_adapter_from_base':True,'full_training_authorized':False,'automatic_retries_authorized':False,
        'additional_experiments_authorized':False,'model_id':'google/gemma-3-4b-it',
        'model_revision':'093f9f388b31de276ce2de164bdc2081324b9767',
        'dataset_sha256':plan['dataset_sha256'],'config_sha256':plan['config_sha256'],
        'runtime_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'packet_manifest_sha256':packet_manifest_sha,'baseline_evaluation_sha256':plan['control_baseline_sha256'],
        'baseline_slug':'oliverlds/teacher-pair-expanded-base-4b','baseline_version':1}
    if (plan['runtime_source_sha256']!=expected['runtime_source_sha256'] or plan['optimizer_updates']!=46
            or plan['training_targets_unchanged']!=46 or plan['replaced_training_inputs']!=12
            or any(approved.get(k)!=v for k,v in expected.items())):
        raise ValueError('Approval does not match the one-for-one 46-update experiment')


def build():
    packet=ROOT/'data/teacher_pair_input_replacement_v1'
    plan=json.loads((packet/'plan.json').read_text())
    approved=json.loads((ROOT/'docs/input_replacement_v1/execution.json').read_text())
    source=(ROOT/'src/fama_style/training/teacher_pair_input_replacement.py').read_text()
    validate_authorization(plan,approved,source,hashlib.sha256((packet/'manifest.json').read_bytes()).hexdigest())
    expected={key:approved[key] for key in ('dataset_sha256','config_sha256','baseline_evaluation_sha256')}
    launch='''from pathlib import Path
import hashlib
import subprocess
import sys
data=list(Path('/kaggle/input').rglob('teacher_pair_replacement.jsonl'))
config=list(Path('/kaggle/input').rglob('teacher_pair_replacement_config.json'))
baseline=[p for p in Path('/kaggle/input').rglob('evaluation.json') if 'teacher_pair_base_v2' in p.parts]
if len(data)!=1 or len(config)!=1 or len(baseline)!=1:
    raise RuntimeError('Attach one private input-replacement Dataset and the expanded BASELINE version 1 outputs')
expected = '''+repr(expected)+'''
for path,key in ((data[0],'dataset_sha256'),(config[0],'config_sha256'),(baseline[0],'baseline_evaluation_sha256')):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected[key]:
        raise RuntimeError('Attached file differs from the approved data/config/baseline')
command=[sys.executable,'/kaggle/working/teacher_pair_input_replacement.py',
         '--data',str(data[0]),'--config',str(config[0])]
subprocess.run(command+['--check'],check=True)
# User approved one fresh two-epoch / 46-update matched SFT experiment.
RUN_PILOT = True
if RUN_PILOT:
    subprocess.run(command+['--mode','pilot','--baseline',str(baseline[0]),'--train-authorized',
        '--output','/kaggle/working/input_replacement_pilot_v1'],check=True)
else:
    print('CPU checks only; no model loaded or trained.')
'''
    book={'nbformat':4,'nbformat_minor':5,
        'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}},
        'cells':[{'cell_type':'markdown','metadata':{},'source':[
            '# Controlled training-input replacement\n',
            'ONE APPROVED LIMITED RUN: 46 targets, 12 replaced inputs, two epochs / 46 updates. Fresh rank-8 adapter from pinned base Gemma 3 4B. No previous adapter is loaded for training.\n',
            'NEW PRIVATE notebook. Attach the two upload files in data/teacher_pair_input_replacement_v1/upload/ and outputs of oliverlds/teacher-pair-expanded-base-4b version 1. Attach the BASELINE output, not the trained pilot adapter.\n',
            'Enable HF_TOKEN in Secrets, Internet and GPU T4 x2. GPU 0 only. RUN_PILOT=True. All 54 rows are tokenized without truncation; the common base must reproduce ten outputs and eight losses before a fresh adapter is created.\n',
            'Targets, row order, two training probes, eight development cases, seed, decoding and optimizer settings match the original-input 46-update control. Checkpoints/evaluations at updates 23 and 46, followed by exact reload verification.\n',
            'Source variants remain provisionally reviewed; this single-seed experiment does not establish a Fama signature. Download input_replacement_pilot_v1 in full. No extra training or automatic retries authorized.\n']},
            code("import os\nimport sys\nif 'torch' in sys.modules:\n    raise RuntimeError('Restart the session and run this cell first')\nos.environ['CUDA_VISIBLE_DEVICES']='0'\nos.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'\nos.environ['TOKENIZERS_PARALLELISM']='false'"),
            code('%pip install -q -U "transformers==5.20.1" "peft==0.21.2" "accelerate==1.15.0" "bitsandbytes==0.50.2" "huggingface-hub==1.33.0" "sentencepiece" "Pillow"'),
            code('%%writefile /kaggle/working/teacher_pair_input_replacement.py\n'+source),code(launch)]}
    path=ROOT/'kaggle/input_replacement_4b.ipynb';path.write_text(json.dumps(book,indent=2,ensure_ascii=False)+'\n')
    if path.stat().st_size>=1_000_000:raise ValueError('Kaggle source size exceeded')
    print('Prepared kaggle/input_replacement_4b.ipynb; no upload or execution.')


if __name__=='__main__':build()
