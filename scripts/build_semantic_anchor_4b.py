"""Self-contained inference-only anchor notebook; no submission or training."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def code(text):
    return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
            'source':text.rstrip().splitlines(keepends=True)}


def build():
    plan=json.loads((ROOT/'data/semantic_anchor_inference_v1/plan.json').read_text())
    source=(ROOT/'src/fama_style/evaluation/semantic_anchor_4b.py').read_text()
    if (plan['runtime_source_sha256']!=hashlib.sha256(source.encode()).hexdigest()
            or plan['training_authorized'] is not False or plan['optimizer_steps']!=0):
        raise ValueError('Frozen inference plan changed')
    expected={k:plan[k] for k in ('jobs_sha256','config_sha256')}
    launch='''from pathlib import Path
import hashlib
import subprocess
import sys
jobs=list(Path('/kaggle/input').rglob('semantic_anchor_jobs.jsonl'))
config=list(Path('/kaggle/input').rglob('semantic_anchor_config.json'))
if len(jobs)!=1 or len(config)!=1:
    raise RuntimeError('Attach exactly one new private Semantic Anchor Inference Dataset')
expected = '''+repr(expected)+'''
for path,key in ((jobs[0],'jobs_sha256'),(config[0],'config_sha256')):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected[key]:
        raise RuntimeError('Attached inference jobs/config differ from the frozen packet')
weights=[p for p in Path('/kaggle/input').rglob('adapter_model.safetensors')
         if 'teacher_pair_pilot_v2' in p.parts and p.parent.name=='adapter' and p.parent.parent.name=='epoch_2']
if len(weights)!=1:
    raise RuntimeError('Attach exactly one output of oliverlds/teacher-pair-expanded-pilot-4b version 1, including epoch_2/adapter')
command=[sys.executable,'/kaggle/working/semantic_anchor_4b.py',
         '--jobs',str(jobs[0]),'--config',str(config[0]),'--adapter',str(weights[0].parent)]
subprocess.run(command+['--check'],check=True)
subprocess.run(command+['--run-inference','--output','/kaggle/working/semantic_anchor_4b_v1'],check=True)
'''
    book={'nbformat':4,'nbformat_minor':5,
        'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}},
        'cells':[{'cell_type':'markdown','metadata':{},'source':[
            '# Semantic anchors: original vs variant inputs\n',
            'INFERENCE ONLY. Twelve training-side cases x two input conditions x base/expanded epoch-2 Gemma: 48 first rewrites and 96 teacher-forced completion scores. Zero optimizer updates.\n',
            'Import as a NEW PRIVATE notebook. Attach only the two upload files from data/semantic_anchor_inference_v1/upload/ and notebook OUTPUTS of oliverlds/teacher-pair-expanded-pilot-4b version 1. The exact epoch-2 adapter hashes are checked.\n',
            'Enable HF_TOKEN in Secrets, Internet and GPU T4 x2. Only GPU 0 is used. Run a fresh session. Full context/target checks occur before model loading, with no truncation.\n',
            'References and rejected targets are scored separately and never supplied to rewrite generation. Targets were seen during adapter training; this is not held-out validation or a Fama-style benchmark.\n',
            'Download semantic_anchor_4b_v1 in full, emitted runtime and log. All outputs stay private. This notebook has not been GPU-executed.\n']},
            code("import os\nimport sys\nif 'torch' in sys.modules:\n    raise RuntimeError('Restart the session and run this cell first')\nos.environ['CUDA_VISIBLE_DEVICES']='0'\nos.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'\nos.environ['TOKENIZERS_PARALLELISM']='false'"),
            code('%pip install -q -U "transformers==5.20.1" "peft==0.21.2" "accelerate==1.15.0" "bitsandbytes==0.50.2" "huggingface-hub==1.33.0" "sentencepiece" "Pillow"'),
            code('%%writefile /kaggle/working/semantic_anchor_4b.py\n'+source),code(launch)]}
    path=ROOT/'kaggle/semantic_anchor_4b.ipynb';path.write_text(json.dumps(book,indent=2,ensure_ascii=False)+'\n')
    if path.stat().st_size>=1_000_000:raise ValueError('Kaggle source size exceeded')
    print('Prepared kaggle/semantic_anchor_4b.ipynb; no upload or execution.')


if __name__=='__main__':build()
