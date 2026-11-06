"""Build a private capped 200-pair Kaggle packet; never upload or submit it."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fama_style.research.corpus_200_packet import build as build_packet
from fama_style.training import corpus_200_pilot as runtime


def cell(text):return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
                       'source':text.rstrip().splitlines(keepends=True)}


def build_notebook(packet,output):
    manifest=json.loads((packet/'manifest.json').read_text())
    for name,h in manifest['files'].items():
        if hashlib.sha256((packet/name).read_bytes()).hexdigest()!=h:raise ValueError('frozen_packet_changed')
    config=json.loads((packet/'upload/corpus_200_config.json').read_text())
    source=Path(runtime.__file__).read_text()
    if runtime.sha(source)!=config['runtime_source_sha256']:raise ValueError('runtime_changed_rebuild_new_packet')
    setup='''import os
import sys
import time
if 'torch' in sys.modules:
    raise RuntimeError('Restart the session and run this first cell before any torch import')
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
PILOT_STARTED=time.monotonic()
PILOT_CAP_SECONDS=5400
def remaining():
    seconds=PILOT_CAP_SECONDS-(time.monotonic()-PILOT_STARTED)
    if seconds<=0: raise TimeoutError('The 90-minute notebook cap has elapsed')
    return seconds
'''
    requirements=[k.replace('_','-')+'=='+v for k,v in runtime.PACKAGES.items()]+['sentencepiece','Pillow']
    install="import subprocess\nsubprocess.run([sys.executable,'-m','pip','install','-q','-U']+"+repr(requirements)+",check=True,timeout=min(900,remaining()))\n"
    expected={name:manifest['files']['upload/'+name] for name in ['corpus_200_pairs.jsonl','corpus_200_config.json']}
    launch='''from pathlib import Path
import hashlib
expected='''+repr(expected)+'''
paths={}
for name,digest in expected.items():
    matches=list(Path('/kaggle/input').rglob(name))
    if len(matches)!=1: raise RuntimeError('Attach exactly one private corpus_200 upload Dataset')
    if hashlib.sha256(matches[0].read_bytes()).hexdigest()!=digest: raise RuntimeError('Attached pilot data/config differs from frozen packet')
    paths[name]=matches[0]
command=[sys.executable,'/kaggle/working/corpus_200_pilot.py',
         '--data',str(paths['corpus_200_pairs.jsonl']),'--config',str(paths['corpus_200_config.json'])]
subprocess.run(command+['--check'],check=True,timeout=min(60,remaining()))
# User-authorized scope: 200 pairs, one epoch, 100 updates, 16 held-out cases.
# This cell runs only when the user executes the uploaded notebook in Kaggle.
subprocess.run(command+['--train-authorized','--max-runtime-seconds',str(max(1,int(remaining()))),
    '--output','/kaggle/working/corpus_200_pilot_v1'],check=True,timeout=remaining())
'''
    notes=['# Balanced 200-pair QLoRA pilot — Gemma 3 4B\n',
        'Create a NEW PRIVATE Kaggle notebook. Upload only the two files from `data/corpus_200_pilot_v1/upload/` as a NEW PRIVATE Dataset and attach it.\n',
        'Enable Internet, select GPU T4 x2, and enable the HF_TOKEN Kaggle Secret before Save & Run All. Code exposes GPU 0 only; the token is read from Secrets and never printed or embedded.\n',
        'User-authorized capped pilot: fresh pinned base; all 200 pairs once, 100 updates, 16 paper-held-out cases; 16 base outputs before training and 16 matched outputs from the reloaded adapter. One 90-minute cap includes dependency installation. No automatic retries or additional epochs.\n',
        'Existing author groups appear on both sides of the paper split. This is not an author-held-out test. The root Gemma 1B pipeline and prior adapters are unchanged.\n',
        'Download `corpus_200_pilot_v1/` and the notebook log after completion. It contains evaluation.json, label and gradient audits, training logs, base/fine-tuned generations and epoch_1/adapter/. Preserve partial outputs on failure.\n',
        'Do not send text to detectors yet. Outputs will first be checked for meaning, copying and readability; eight detector cases are preselected. No Pangram account, API key or detector call is used here.\n']
    book={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}},
          'cells':[{'cell_type':'markdown','metadata':{},'source':notes},cell(setup),cell(install),
                   cell('%%writefile /kaggle/working/corpus_200_pilot.py\n'+source),cell(launch)]}
    output.write_text(json.dumps(book,indent=2,ensure_ascii=False)+'\n')
    if output.stat().st_size>=1_000_000:raise ValueError('Kaggle_notebook_source_limit')
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',type=Path,default=ROOT/'data/corpus_200_pilot_v1')
    parser.add_argument('--notebook',type=Path,default=ROOT/'kaggle/corpus_200_pilot_4b.ipynb')
    args=parser.parse_args()
    if not args.packet.exists():
        build_packet(ROOT/'data/data_scaling_v1/stage_0200_salvage_v1/training.private.jsonl',
                     ROOT/'data/author_pairs_salvage_v1/development.private.jsonl',args.packet)
    build_notebook(args.packet,args.notebook)
    print('Prepared '+str(args.notebook)+' and '+str(args.packet/'upload')+'. No upload or GPU job submitted.')


if __name__=='__main__':main()
