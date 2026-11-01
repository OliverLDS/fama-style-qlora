"""Build a self-contained, check-only-by-default notebook. Never submit jobs."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def cell(source):return {'cell_type':'code','metadata':{},'source':source.splitlines(keepends=True),'outputs':[],'execution_count':None}


def build(packet=None,output=None,approval=None):
    packet=packet or ROOT/'data/actual_error_pilot_v1'
    output=output or ROOT/'kaggle/preference_pair_pilot_4b.ipynb'
    plan=json.loads((packet/'plan.json').read_text())
    approved=False
    if approval is not None:
        record=json.loads(approval.read_text())
        if (record.get('explicit_user_approval') is not True or record.get('total_updates')!=44
                or record.get('maximum_runtime_seconds')!=3600 or record.get('visible_gpu_count')!=1
                or record.get('packet_manifest_sha256')!=hashlib.sha256((packet/'manifest.json').read_bytes()).hexdigest()):
            raise ValueError('Approval must match this bounded frozen pilot')
        approved=True
    sources={
        'preference_token_preflight.py':(ROOT/'src/fama_style/evaluation/preference_token_preflight.py').read_text(),
        'preference_pair_pilot.py':(ROOT/'src/fama_style/training/preference_pair_pilot.py').read_text()}
    for name,key in (('preference_token_preflight.py','token_helper_sha256'),('preference_pair_pilot.py','runtime_source_sha256')):
        if hashlib.sha256(sources[name].encode()).hexdigest()!=plan[key]:raise ValueError('Frozen pilot runtime changed')
    setup='''import os
import time
import sys
import subprocess
from pathlib import Path
NOTEBOOK_STARTED=time.monotonic()
# Leave False until the bounded GPU pilot has explicit approval.
RUN_APPROVED_PILOT=False
if 'torch' in sys.modules:
    raise RuntimeError('Restart with a fresh session before executing this notebook')
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
os.environ['DO_NOT_TRACK']='1'
def remaining():
    seconds=3600-(time.monotonic()-NOTEBOOK_STARTED)
    if seconds<1:raise TimeoutError('One-hour notebook budget exhausted')
    return seconds
if RUN_APPROVED_PILOT:
    dependencies=['transformers==5.20.1','peft==0.21.2','accelerate==1.15.0',
                  'bitsandbytes==0.50.2','huggingface-hub==1.33.0','sentencepiece','Pillow','jinja2==3.1.6']
    subprocess.run([sys.executable,'-m','pip','install','-q','-U']+dependencies,check=True,timeout=remaining())
else:
    print('Check-only mode: no packages installed, model loaded, GPU work or training.')
'''
    if approved:
        setup=setup.replace('# Leave False until the bounded GPU pilot has explicit approval.\nRUN_APPROVED_PILOT=False',
            '# Explicit user approval recorded for this frozen 44-update / one-hour pilot.\nRUN_APPROVED_PILOT=True')
    write="from pathlib import Path\nSOURCES="+repr(sources)+"\nfor name,source in SOURCES.items():\n    (Path('/kaggle/working')/name).write_text(source)\n"
    launch='''import hashlib
import json
inputs=Path('/kaggle/input')
data=list(inputs.rglob('actual_error_pilot_data.json'))
configs=list(inputs.rglob('actual_error_pilot_config.json'))
if len(data)!=1 or len(configs)!=1:
    raise RuntimeError('Attach exactly one private pilot Dataset with its two upload files')
expected='''+repr({k:plan[k] for k in ('data_sha256','config_sha256')})+'''
for path,key in ((data[0],'data_sha256'),(configs[0],'config_sha256')):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected[key]:raise RuntimeError('Frozen pilot payload changed')
command=[sys.executable,'/kaggle/working/preference_pair_pilot.py','--data',str(data[0]),'--config',str(configs[0])]
if RUN_APPROVED_PILOT:
    adapter_hashes=json.loads(configs[0].read_text())['adapter_sha256']
    adapters=[]
    for weights in inputs.rglob('adapter_model.safetensors'):
        folder=weights.parent
        if all((folder/name).is_file() and hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest
               for name,digest in adapter_hashes.items()):adapters.append(folder)
    if len(adapters)!=1:
        raise RuntimeError('Attach the original teacher-pair-expanded-pilot-4b version 1 outputs; epoch_2 adapter must match exactly')
    command+=['--adapter',str(adapters[0])]
subprocess.run(command+['--check'],check=True,timeout=remaining())
if RUN_APPROVED_PILOT:
    subprocess.run(command+['--train-authorized','--max-runtime-seconds',str(int(remaining())),
        '--output','/kaggle/working/actual_error_pilot_v1'],check=True,timeout=remaining())
else:
    print('Checks complete. Training was not authorized or started.')
'''
    book={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},
        'language_info':{'name':'python'}},'cells':[{'cell_type':'markdown','metadata':{},'source':[
        '# Bounded preference versus SFT pilot — prepared, not executed\n',
        ('Approved bounded GPU pilot: training is ENABLED. Select GPU T4 x2; GPU 0 alone is exposed.\n' if approved else
         'NEW PRIVATE notebook and Dataset. Default is check-only; Accelerator None is sufficient for that mode.\n'),
        'Attach both files in data/actual_error_pilot_v1/upload/. They contain private copyrighted references. Do not publish them.\n',
        'Only after explicit GPU approval: enable RUN_APPROVED_PILOT, select GPU T4 x2, enable Internet and HF_TOKEN in Secrets. GPU 0 alone is exposed.\n',
        'Attach outputs of oliverlds/teacher-pair-expanded-pilot-4b version 1, including its original epoch_2 adapter. Not the input-replacement adapter.\n',
        'Two independent 22-update continuations from that adapter: chosen-only SFT and DPO. Epoch 2 is primary; epoch 1 diagnostic.\n',
        'The unchanged control must reproduce eight archived outputs before training. Frozen reference scores, processor masks and real likelihoods must pass first.\n',
        'The existing Kaggle Torch must be 2.11.0+cu128; mismatches stop execution. Memory fit and runtime are unmeasured.\n',
        'One-hour wall-clock cap starts in the first code cell and includes dependency setup. Keep partial outputs on failure; do not silently change settings.\n',
        'Download actual_error_pilot_v1/ and the notebook log after an approved run. No full training, detector calls or automatic submission.\n']},cell(setup),cell(write),cell(launch)]}
    output.write_text(json.dumps(book,indent=2,ensure_ascii=False)+'\n')
    if output.stat().st_size>=1_000_000:raise ValueError('Kaggle notebook source exceeds limit')
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--packet',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--approval',type=Path)
    a=p.parse_args();print(build(a.packet,a.output,a.approval))
