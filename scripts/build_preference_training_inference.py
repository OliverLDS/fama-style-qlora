"""Build a self-contained inference-only notebook. Never upload or submit."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def cell(source):return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':source.splitlines(keepends=True)}


def build(packet=None,output=None):
    packet=packet or ROOT/'data/preference_training_inference_v1'
    output=output or ROOT/'kaggle/preference_training_inference_4b.ipynb'
    plan=json.loads((packet/'plan.json').read_text());source=(ROOT/'src/fama_style/evaluation/preference_training_inference.py').read_text()
    if (hashlib.sha256(source.encode()).hexdigest()!=plan['runtime_source_sha256'] or plan['optimizer_steps']!=0
            or plan['training_authorized'] is not False):raise ValueError('frozen_inference_plan_changed')
    setup='''import os
import sys
import time
import subprocess
from pathlib import Path
NOTEBOOK_STARTED=time.monotonic()
if 'torch' in sys.modules:raise RuntimeError('Restart a fresh session and execute this cell first')
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['DO_NOT_TRACK']='1'
def remaining():
    value=1800-(time.monotonic()-NOTEBOOK_STARTED)
    if value<1:raise TimeoutError('Thirty-minute inference budget exhausted')
    return value
dependencies=['transformers==5.20.1','peft==0.21.2','accelerate==1.15.0',
              'bitsandbytes==0.50.2','huggingface-hub==1.33.0','sentencepiece','Pillow','jinja2==3.1.6']
subprocess.run([sys.executable,'-m','pip','install','-q','-U']+dependencies,check=True,timeout=remaining())
'''
    write="from pathlib import Path\n(Path('/kaggle/working')/'preference_training_inference.py').write_text("+repr(source)+")\n"
    launch='''import hashlib
import json
inputs=Path('/kaggle/input')
jobs=list(inputs.rglob('preference_training_jobs.jsonl'))
configs=list(inputs.rglob('preference_training_config.json'))
if len(jobs)!=1 or len(configs)!=1:raise RuntimeError('Attach exactly one new private training-inference Dataset with its two upload files')
expected='''+repr({k:plan[k] for k in ('jobs_sha256','config_sha256')})+'''
for path,key in ((jobs[0],'jobs_sha256'),(configs[0],'config_sha256')):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected[key]:raise RuntimeError('Frozen inference payload changed')
config=json.loads(configs[0].read_text())
command=[sys.executable,'/kaggle/working/preference_training_inference.py','--jobs',str(jobs[0]),'--config',str(configs[0])]
for arm in ('sft','dpo'):
    matches=[]
    for weights in inputs.rglob('adapter_model.safetensors'):
        folder=weights.parent
        if all((folder/name).is_file() and hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest
               for name,digest in config['adapter_sha256'][arm].items()):matches.append(folder)
    if len(matches)!=1:raise RuntimeError('Attach outputs of oliverlds/preference-pair-pilot-4b version 1, including both final epoch_2 adapters')
    command+=['--'+arm+'-adapter',str(matches[0])]
subprocess.run(command+['--check'],check=True,timeout=remaining())
subprocess.run(command+['--run-inference','--max-runtime-seconds',str(int(remaining())),
    '--output','/kaggle/working/preference_training_inference_v1'],check=True,timeout=remaining())
'''
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},
        'language_info':{'name':'python'}},'cells':[{'cell_type':'markdown','metadata':{},'source':[
        '# Training-side free generation after the preference pilot\n',
        'INFERENCE ONLY: eleven fixed training paragraphs x two saved epoch-2 adapters = 22 new generations. Zero optimizer steps. Eleven exact archived control responses are reused in local analysis.\n',
        'Create a NEW PRIVATE notebook. Attach only the two upload files from data/preference_training_inference_v1/upload/ as a private Dataset. Do not upload private_key/.\n',
        'Attach notebook OUTPUTS of oliverlds/preference-pair-pilot-4b version 1, including sft/epoch_2/adapter and dpo/epoch_2/adapter. No original expanded-pilot adapter needed.\n',
        'Select GPU T4 x2; only GPU 0 is exposed. Enable Internet and HF_TOKEN in Secrets before Save & Run All. Keep credentials outside code.\n',
        'Use a fresh session with Torch 2.11.0+cu128. All model/package/adapter hashes and exact training prompt tokens are checked. No truncation or prompt changes.\n',
        'The upload contains source prompts only, with no targets or archived outputs. All parameters are frozen; there is no training or checkpoint saving.\n',
        'The thirty-minute cap includes setup. Download preference_training_inference_v1/ and the notebook log after completion, including partial outputs on failure.\n',
        'This is a seen-training diagnostic, not held-out evaluation or evidence of Fama style/human authorship. No detector calls or automatic submission.\n']},cell(setup),cell(write),cell(launch)]}
    output.write_text(json.dumps(notebook,indent=2,ensure_ascii=False)+'\n')
    if output.stat().st_size>=1_000_000:raise ValueError('Kaggle source exceeds limit')
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--packet',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();print(build(a.packet,a.output))
