"""Build a self-contained fourteen-output milestone notebook; never submit it."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def code(text):return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
                       'source':text.rstrip().splitlines(keepends=True)}


def build():
    plan=json.loads((ROOT/'data/milestone_checkpoint_4b_v1/plan.json').read_text())
    source=(ROOT/'src/fama_style/evaluation/milestone_checkpoint_4b.py').read_text()
    if plan['runtime_source_sha256']!=hashlib.sha256(source.encode()).hexdigest() or plan['new_outputs_planned']!=14:
        raise ValueError('Frozen missing-job plan changed')
    expected={k:plan[k] for k in ('jobs_sha256','config_sha256')}
    launch='''from pathlib import Path
import hashlib
import subprocess
import sys
jobs=list(Path('/kaggle/input').rglob('milestone_checkpoint_jobs.jsonl'))
configs=list(Path('/kaggle/input').rglob('milestone_checkpoint_config.json'))
weights=[p for p in Path('/kaggle/input').rglob('adapter_model.safetensors')
         if 'teacher_pair_pilot_v2' in p.parts and p.parent.name=='adapter' and p.parent.parent.name=='epoch_2']
if len(jobs)!=1 or len(configs)!=1 or len(weights)!=1:
    raise RuntimeError('Attach the new Milestone Checkpoint Dataset and expanded-pilot version 1 outputs (epoch-2 adapter)')
expected = '''+repr(expected)+'''
for path,key in ((jobs[0],'jobs_sha256'),(configs[0],'config_sha256')):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected[key]:
        raise RuntimeError('Attached checkpoint jobs/config differ from the frozen packet')
command=[sys.executable,'/kaggle/working/milestone_checkpoint_4b.py','--jobs',str(jobs[0]),
         '--config',str(configs[0]),'--adapter',str(weights[0].parent)]
subprocess.run(command+['--check'],check=True)
subprocess.run(command+['--run-inference','--output','/kaggle/working/milestone_checkpoint_4b_v1'],check=True)
'''
    book={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}},
        'cells':[{'cell_type':'markdown','metadata':{},'source':[
            '# Multi-author milestone checkpoint: missing 4B outputs\n',
            'INFERENCE ONLY. Original frozen eight cases, three arms: base ordinary, base source-only structural prior, original-input expanded adapter ordinary. Ten outputs archived; generate only 14 missing outputs.\n',
            'NEW PRIVATE notebook. Attach the two files in data/milestone_checkpoint_4b_v1/upload/ and OUTPUTS of oliverlds/teacher-pair-expanded-pilot-4b version 1, including epoch_2/adapter. Do not attach the input-replacement adapter.\n',
            'Enable HF_TOKEN in Secrets, Internet, GPU T4 x2. GPU 0 only. Exact adapter, environment and generation settings checked before reuse. No training or detector calls.\n',
            'Structural prompts repeat source text and use a longer safety guard; this is a static optional prior, not a learned mapper or a matched-length structure ablation. References/author labels never enter generator payloads.\n',
            'Download milestone_checkpoint_4b_v1, emitted runtime and notebook log. Detector export remains blocked until all missing outputs are complete.\n']},
            code("import os\nimport sys\nif 'torch' in sys.modules:\n    raise RuntimeError('Restart the session and run this cell first')\nos.environ['CUDA_VISIBLE_DEVICES']='0'\nos.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'\nos.environ['TOKENIZERS_PARALLELISM']='false'"),
            code('%pip install -q -U "transformers==5.20.1" "peft==0.21.2" "accelerate==1.15.0" "bitsandbytes==0.50.2" "huggingface-hub==1.33.0" "sentencepiece" "Pillow"'),
            code('%%writefile /kaggle/working/milestone_checkpoint_4b.py\n'+source),code(launch)]}
    path=ROOT/'kaggle/milestone_checkpoint_4b.ipynb';path.write_text(json.dumps(book,indent=2,ensure_ascii=False)+'\n')
    if path.stat().st_size>=1_000_000:raise ValueError('Kaggle source limit exceeded')
    print('Prepared kaggle/milestone_checkpoint_4b.ipynb; no upload or execution.')


if __name__=='__main__':build()
