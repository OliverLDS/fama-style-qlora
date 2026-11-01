"""Build separate base and explicitly authorized pilot notebooks; never submit them."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def code(text):return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':text.rstrip().splitlines(keepends=True)}


def build():
    source=(ROOT/'src/fama_style/training/teacher_pair_pilot.py').read_text()
    authorization=json.loads((ROOT/'docs/teacher_pair_pilot_v1/pilot_execution.json').read_text())
    if (authorization['user_authorized'] is not True or authorization['epochs']!=2
            or authorization['optimizer_steps']!=16 or authorization['full_training_authorized'] is not False
            or authorization['runtime_source_sha256']!=hashlib.sha256(source.encode()).hexdigest()):
        raise ValueError('Limited pilot authorization does not match runtime/scope')
    locate='''from pathlib import Path
import subprocess
import sys
data=list(Path('/kaggle/input').rglob('teacher_pair_pilot.jsonl'))
config=list(Path('/kaggle/input').rglob('teacher_pair_config.json'))
if len(data)!=1 or len(config)!=1:
    raise RuntimeError('Attach exactly one private teacher-pair Dataset with the two upload files')
command=[sys.executable,'/kaggle/working/teacher_pair_pilot.py','--data',str(data[0]),'--config',str(config[0])]
subprocess.run(command+['--check'],check=True)
'''
    for mode in ('base','pilot'):
        launch=locate
        if mode=='base':
            launch+="subprocess.run(command+['--mode','baseline','--output','/kaggle/working/teacher_pair_base_v1'],check=True)\n"
        else:
            launch+='''# User authorized this limited pilot after baseline v1 review (2026-10-31).
RUN_PILOT = True
if RUN_PILOT:
    baseline=[p for p in Path('/kaggle/input').rglob('evaluation.json')
              if 'teacher_pair_base_v1' in p.parts]
    if len(baseline)!=1:
        raise RuntimeError('Attach exactly one complete teacher_pair_base_v1 baseline output')
    import hashlib
    expected = '''+repr({key:authorization[key] for key in ('baseline_evaluation_sha256','dataset_sha256','config_sha256')})+'''
    for path,key in ((baseline[0],'baseline_evaluation_sha256'),(data[0],'dataset_sha256'),(config[0],'config_sha256')):
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected[key]:
            raise RuntimeError('Attached '+key+' differs from the frozen, reviewed pilot inputs')
    subprocess.run(command+['--mode','pilot','--baseline',str(baseline[0]),'--train-authorized',
                            '--output','/kaggle/working/teacher_pair_pilot_v1'],check=True)
else:
    print('CPU preflight only; RUN_PILOT was manually disabled.')
'''
        notes=['# Teacher-pair reverse rewrite: '+('base comparison\n' if mode=='base' else 'limited QLoRA pilot\n'),
            'Gemma 3 4B, pinned revision, 16 training and 8 paper/author-held-out development pairs. All data and outputs stay private.\n',
            'Import as a NEW PRIVATE notebook. Attach the two-file Teacher Pair Reverse Pilot Dataset. Enable HF_TOKEN in Secrets, Internet, and GPU T4 x2. Only GPU 0 is used.\n']
        notes.append('Inference and loss evaluation only: 8 development outputs plus 2 labelled training probes. No training. Run this notebook first.\n' if mode=='base'
                     else 'LIMITED PILOT AUTHORIZED after baseline v1 review. RUN_PILOT is True. Attach the existing private Dataset AND outputs of oliverlds/teacher-pair-base-4b version 1 before Save & Run All. Cap: two epochs / sixteen updates; no full training.\n')
        notes.append('This new notebook has not been GPU-executed. A successful run is not evidence of claim preservation, Fama style or human authorship.\n')
        book={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}},
            'cells':[{'cell_type':'markdown','metadata':{},'source':notes},
                code("import os\nimport sys\nif 'torch' in sys.modules:\n    raise RuntimeError('Restart the session and run this cell first')\nos.environ['CUDA_VISIBLE_DEVICES']='0'\nos.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'\nos.environ['TOKENIZERS_PARALLELISM']='false'"),
                code('%pip install -q -U "transformers==5.20.1" "peft==0.21.2" "accelerate==1.15.0" "bitsandbytes==0.50.2" "huggingface-hub==1.33.0" "sentencepiece" "Pillow"'),
                code('%%writefile /kaggle/working/teacher_pair_pilot.py\n'+source),code(launch)]}
        name='teacher_pair_base_4b.ipynb' if mode=='base' else 'teacher_pair_pilot_4b.ipynb'
        path=ROOT/'kaggle'/name;path.write_text(json.dumps(book,indent=2,ensure_ascii=False)+'\n')
        if path.stat().st_size>=1_000_000:raise ValueError('Kaggle source size exceeded')
        print('Prepared '+str(path.relative_to(ROOT))+'; no upload or execution.')


if __name__=='__main__':build()
