"""Build a self-contained notebook without data, secrets, uploads or execution."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def code(text):
    return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
            'source':text.rstrip().splitlines(keepends=True)}


def build():
    source=(ROOT/'src/fama_style/evaluation/reconstruction_4b.py').read_text()
    launch='''from pathlib import Path
import subprocess
import sys
jobs=list(Path('/kaggle/input').rglob('reconstruction_4b_jobs.jsonl'))
configs=list(Path('/kaggle/input').rglob('reconstruction_4b_config.json'))
if len(jobs)!=1 or len(configs)!=1:
    raise RuntimeError('Attach exactly one private Dataset with the two prepared upload files')
command=[sys.executable,'/kaggle/working/reconstruction_4b.py','--jobs',str(jobs[0]),'--config',str(configs[0])]
subprocess.run(command+['--check'],check=True)
subprocess.run(command,check=True)
'''
    book={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}},
        'cells':[{'cell_type':'markdown','metadata':{},'source':[
            '# Gemma 3 4B: direct rewrite versus brief-only reconstruction\n',
            'Prepared inference-only diagnostic: three existing development cases, six first outputs. No adapter or training.\n',
            'Import as a NEW PRIVATE notebook. Attach the two-file private Dataset from the prepared upload/ folder. Enable HF_TOKEN in Secrets, Internet, and GPU T4 x2. Only GPU 0 is used.\n',
            'Outputs stay private. Download the entire reconstruction_4b_v1 output folder. GPU behavior has not been tested locally.\n']},
            code("import os\nimport sys\nif 'torch' in sys.modules:\n    raise RuntimeError('Restart the session and run this cell before importing torch')\nos.environ['CUDA_VISIBLE_DEVICES']='0'\nos.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'\nos.environ['TOKENIZERS_PARALLELISM']='false'"),
            code('%pip install -q -U "transformers==5.20.1" "accelerate==1.15.0" "bitsandbytes==0.50.2" "huggingface-hub==1.33.0" "sentencepiece" "Pillow"'),
            code('%%writefile /kaggle/working/reconstruction_4b.py\n'+source),code(launch),
            code("import json\nfrom pathlib import Path\nfolder=Path('/kaggle/working/reconstruction_4b_v1')\nstate=json.loads((folder/'state.json').read_text())\nprint(json.dumps(state,indent=2))\nif not state['complete'] or state['completed_jobs']!=6 or state['training_called']:\n    raise RuntimeError('The inference check is incomplete; preserve all outputs for review')\nprint('Download the whole reconstruction_4b_v1 folder; quality remains unassessed.')")]}
    path=ROOT/'kaggle/reconstruction_4b.ipynb';path.write_text(json.dumps(book,indent=2,ensure_ascii=False)+'\n')
    if path.stat().st_size>=1_000_000:raise ValueError('Kaggle notebook source limit exceeded')
    print('Prepared '+str(path.relative_to(ROOT))+'; no upload or execution.')


if __name__=='__main__':build()
