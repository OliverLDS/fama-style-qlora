"""Build the approved single-arm bounded SFT notebook; never upload or submit."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def cell(source):return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':source.splitlines(keepends=True)}


def build(packet=None,output=None):
    packet=packet or ROOT/'data/sft_strength_v1';output=output or ROOT/'kaggle/sft_strength_4b.ipynb'
    plan=json.loads((packet/'plan.json').read_text());approval=json.loads((packet/'authorization.json').read_text())
    if (not approval['user_authorized'] or approval['optimizer_updates']!=22 or approval['maximum_runtime_seconds']!=1800
            or approval['dpo_training_authorized'] is not False or approval['full_training_authorized'] is not False
            or approval['config_sha256']!=plan['config_sha256'] or approval['dataset_sha256']!=plan['dataset_sha256']
            or approval['runtime_source_sha256']!=plan['runtime_source_sha256']):raise ValueError('bounded_authorization_mismatch')
    sources={}
    for name,path,key in (
        ('preference_token_preflight.py','src/fama_style/evaluation/preference_token_preflight.py','token_helper_sha256'),
        ('preference_pair_pilot.py','src/fama_style/training/preference_pair_pilot.py','prior_runtime_sha256'),
        ('sft_strength_pilot.py','src/fama_style/training/sft_strength_pilot.py','runtime_source_sha256')):
        source=(ROOT/path).read_text()
        if hashlib.sha256(source.encode()).hexdigest()!=plan[key]:raise ValueError('frozen_runtime_changed')
        sources[name]=source
    setup='''import os
import sys
import time
import subprocess
from pathlib import Path
NOTEBOOK_STARTED=time.monotonic()
if 'torch' in sys.modules:raise RuntimeError('Restart a fresh session; run this cell first')
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
os.environ['DO_NOT_TRACK']='1'
def remaining():
    value=1800-(time.monotonic()-NOTEBOOK_STARTED)
    if value<1:raise TimeoutError('Thirty-minute bounded SFT budget exhausted')
    return value
dependencies=['transformers==5.20.1','peft==0.21.2','accelerate==1.15.0',
              'bitsandbytes==0.50.2','huggingface-hub==1.33.0','sentencepiece','Pillow','jinja2==3.1.6']
subprocess.run([sys.executable,'-m','pip','install','-q','-U']+dependencies,check=True,timeout=remaining())
'''
    write='from pathlib import Path\nSOURCES='+repr(sources)+"\nfor name,source in SOURCES.items():\n    (Path('/kaggle/working')/name).write_text(source)\n"
    launch='''import hashlib
import json
inputs=Path('/kaggle/input')
data=list(inputs.rglob('sft_strength_data.json'));configs=list(inputs.rglob('sft_strength_config.json'))
if len(data)!=1 or len(configs)!=1:raise RuntimeError('Attach only the new private SFT Strength Dataset with its two upload files')
expected='''+repr({k:plan[k] for k in ('dataset_sha256','config_sha256')})+'''
for path,key in ((data[0],'dataset_sha256'),(configs[0],'config_sha256')):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected[key]:raise RuntimeError('Frozen payload changed')
config=json.loads(configs[0].read_text());matches=[]
for weights in inputs.rglob('adapter_model.safetensors'):
    folder=weights.parent
    if all((folder/name).is_file() and hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest
           for name,digest in config['adapter_sha256'].items()):matches.append(folder)
if len(matches)!=1:raise RuntimeError('Attach original teacher-pair-expanded-pilot-4b version 1 outputs, including epoch_2/adapter; not the weak SFT/DPO adapters')
command=[sys.executable,'/kaggle/working/sft_strength_pilot.py','--data',str(data[0]),'--config',str(configs[0]),'--adapter',str(matches[0])]
subprocess.run(command+['--check'],check=True,timeout=remaining())
subprocess.run(command+['--train-authorized','--max-runtime-seconds',str(int(remaining())),
    '--output','/kaggle/working/sft_strength_v1'],check=True,timeout=remaining())
'''
    book={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},
        'language_info':{'name':'python'}},'cells':[{'cell_type':'markdown','metadata':{},'source':[
        '# Approved bounded SFT strength check\n',
        'TRAINING ENABLED for one SFT continuation: same 17 pairs, 22 updates, learning rate 1e-5 rather than 1e-6. No DPO or full-scale training.\n',
        'Create a NEW PRIVATE notebook. Attach only the two files in data/sft_strength_v1/upload/ as a private Dataset; do not publish or attach private_key/.\n',
        'Attach OUTPUTS of oliverlds/teacher-pair-expanded-pilot-4b version 1, including its original epoch_2 adapter. Do not start from the weak preference-pilot adapter.\n',
        'Select GPU T4 x2, enable Internet and HF_TOKEN in Secrets, then Save & Run All. GPU 0 alone is exposed. No credential values belong in code.\n',
        'The unchanged adapter must reproduce eight development outputs and seventeen initial target likelihoods before updates. Original data, masks and group order are frozen.\n',
        'Epoch 2 is primary. Epoch 1 is saved only; no selection or extra evaluation. After final adapter reload, generate eleven seen training and eight development outputs.\n',
        'The frozen prior runtime is embedded only for CPU validators and pure helpers; its old training/DPO entrypoint is never invoked.\n',
        'Require recorded Torch 2.11.0+cu128. The thirty-minute wall cap includes dependency setup; keep partial outputs if a gate fails.\n',
        'Download sft_strength_v1/ and the notebook log. No detector calls, reserved-test access or automatic submission. This notebook has not yet been GPU-executed.\n']},cell(setup),cell(write),cell(launch)]}
    output.write_text(json.dumps(book,indent=2,ensure_ascii=False)+'\n')
    if output.stat().st_size>=1_000_000:raise ValueError('Kaggle source limit exceeded')
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--packet',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();print(build(a.packet,a.output))
