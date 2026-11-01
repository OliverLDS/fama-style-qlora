"""Build new self-contained expanded-pair notebooks without submission or execution."""
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]


def code(text):
    return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
            'source':text.rstrip().splitlines(keepends=True)}


def validate_authorization(plan,authorization,source,packet):
    expected={'protocol':'teacher_pair_reverse_pilot_execution_v2','user_authorized':True,
        'epochs':2,'optimizer_steps':46,'training_rows':46,'development_rows':8,
        'model_id':'google/gemma-3-4b-it',
        'model_revision':'093f9f388b31de276ce2de164bdc2081324b9767',
        'full_training_authorized':False,'automatic_retries_authorized':False,
        'additional_experiments_authorized':False,
        'runtime_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'packet_manifest_sha256':hashlib.sha256((packet/'manifest.json').read_bytes()).hexdigest(),
        'dataset_sha256':plan['dataset_sha256'],'config_sha256':plan['config_sha256'],
        'baseline_slug':'oliverlds/teacher-pair-expanded-base-4b','baseline_version':1}
    if (any(authorization.get(key)!=value for key,value in expected.items())
            or re.fullmatch('[0-9a-f]{64}',authorization.get('baseline_evaluation_sha256','')) is None):
        raise ValueError('Approval does not match the frozen 46-update pilot')


def build():
    packet=ROOT/'data/teacher_pair_expanded_experiment_v2'
    plan=json.loads((packet/'plan.json').read_text())
    source=(ROOT/'src/fama_style/training/teacher_pair_expanded_pilot.py').read_text()
    if (plan['runtime_source_sha256']!=hashlib.sha256(source.encode()).hexdigest()
            or plan['training_authorized'] is not False or plan['optimizer_updates']!=46):
        raise ValueError('Prepared plan/runtime mismatch')
    expected={key:plan[key] for key in ('dataset_sha256','config_sha256')}
    authorization_path=ROOT/'docs/teacher_pair_pilot_v2/pilot_execution.json'
    authorization=json.loads(authorization_path.read_text()) if authorization_path.exists() else None
    if authorization is not None:validate_authorization(plan,authorization,source,packet)
    locate='''from pathlib import Path
import hashlib
import subprocess
import sys
data=list(Path('/kaggle/input').rglob('teacher_pair_expanded.jsonl'))
config=list(Path('/kaggle/input').rglob('teacher_pair_expanded_config.json'))
if len(data)!=1 or len(config)!=1:
    raise RuntimeError('Attach exactly one NEW private expanded teacher-pair Dataset')
expected = '''+repr(expected)+'''
for path,key in ((data[0],'dataset_sha256'),(config[0],'config_sha256')):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected[key]:
        raise RuntimeError('Attached file differs from frozen expanded-pair packet')
command=[sys.executable,'/kaggle/working/teacher_pair_expanded_pilot.py',
         '--data',str(data[0]),'--config',str(config[0])]
subprocess.run(command+['--check'],check=True)
'''
    for mode in ('base','pilot'):
        launch=locate
        if mode=='base':
            launch+="subprocess.run(command+['--mode','baseline','--output','/kaggle/working/teacher_pair_base_v2'],check=True)\n"
        else:
            launch+='''# Limited scope approved after the expanded baseline version 1 review.
RUN_PILOT = '''+str(authorization is not None)+'''
if RUN_PILOT:
    baseline=[p for p in Path('/kaggle/input').rglob('evaluation.json')
              if 'teacher_pair_base_v2' in p.parts]
    if len(baseline)!=1:
        raise RuntimeError('Attach exactly one complete teacher_pair_base_v2 baseline output')
    expected_baseline_sha256 = '''+repr(authorization['baseline_evaluation_sha256'] if authorization else None)+'''
    if hashlib.sha256(baseline[0].read_bytes()).hexdigest()!=expected_baseline_sha256:
        raise RuntimeError('Attached baseline differs from the exact reviewed version 1 report')
    subprocess.run(command+['--mode','pilot','--baseline',str(baseline[0]),
        '--train-authorized','--output','/kaggle/working/teacher_pair_pilot_v2'],check=True)
else:
    print('CPU file checks complete. No model loaded and no training started; RUN_PILOT=False.')
'''
        notes=['# Expanded teacher pairs: '+mode+' v2\n',
            'NEW PRIVATE notebook and Dataset; never replace the frozen v1 artifacts.\n',
            'Attach only the two files from data/teacher_pair_expanded_experiment_v2/upload/. Enable HF_TOKEN in Kaggle Secrets, Internet and GPU T4 x2. GPU 0 only.\n',
            'Gemma 3 4B, pinned revision, 46 training pairs and the unchanged eight development pairs. No reserved test data.\n',
            ('Run this baseline first: all 54 tokenizer/label checks, eight development losses and ten generations; no training.\n'
             if mode=='base' else ('LIMITED PILOT AUTHORIZED. RUN_PILOT=True. Two epochs / 46 updates, evaluations at 0/23/46, plus reload check. Attach outputs from oliverlds/teacher-pair-expanded-base-4b version 1.\n'
                                  if authorization else 'TRAINING DISABLED. Two epochs / 46 updates prepared; explicit approval required.\n')),
            'More data AND more optimizer updates than v1; not a causal test of data volume. No Fama-style or detector improvement established.\n']
        book={'nbformat':4,'nbformat_minor':5,
            'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}},
            'cells':[{'cell_type':'markdown','metadata':{},'source':notes},
                code("import os\nimport sys\nif 'torch' in sys.modules:\n    raise RuntimeError('Restart the session and run this cell first')\nos.environ['CUDA_VISIBLE_DEVICES']='0'\nos.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'\nos.environ['TOKENIZERS_PARALLELISM']='false'"),
                code('%pip install -q -U "transformers==5.20.1" "peft==0.21.2" "accelerate==1.15.0" "bitsandbytes==0.50.2" "huggingface-hub==1.33.0" "sentencepiece" "Pillow"'),
                code('%%writefile /kaggle/working/teacher_pair_expanded_pilot.py\n'+source),code(launch)]}
        path=ROOT/'kaggle'/('teacher_pair_expanded_'+mode+'_4b.ipynb')
        path.write_text(json.dumps(book,indent=2,ensure_ascii=False)+'\n')
        if path.stat().st_size>=1_000_000:raise ValueError('Kaggle source size exceeded')
        print('Prepared '+str(path.relative_to(ROOT))+'; no upload or execution.')


if __name__=='__main__':build()
