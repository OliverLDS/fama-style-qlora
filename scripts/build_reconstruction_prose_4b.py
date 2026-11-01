"""Build a self-contained nine-output inference notebook; never submit it."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def code(text):
    return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
            'source':text.rstrip().splitlines(keepends=True)}


def build():
    source=(ROOT/'src/fama_style/evaluation/reconstruction_prose_4b.py').read_text()
    launch='''from pathlib import Path
import subprocess
import sys
jobs=list(Path('/kaggle/input').rglob('reconstruction_prose_4b_jobs.jsonl'))
configs=list(Path('/kaggle/input').rglob('reconstruction_prose_4b_config.json'))
if len(jobs)!=1 or len(configs)!=1:
    raise RuntimeError('Attach exactly one prepared private prose-comparison Dataset')
command=[sys.executable,'/kaggle/working/reconstruction_prose_4b.py','--jobs',str(jobs[0]),'--config',str(configs[0])]
subprocess.run(command+['--check'],check=True)
subprocess.run(command,check=True)
'''
    book={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}},
        'cells':[{'cell_type':'markdown','metadata':{},'source':[
            '# Matched Gemma 3 4B prose comparison\n',
            'Three existing training-side cases x direct rewrite, detailed brief, compact claims: nine first outputs. No training or adapter.\n',
            'Import as a NEW PRIVATE notebook. Attach the two-file private Dataset from the new upload/ folder. Enable HF_TOKEN in Secrets, Internet, and GPU T4 x2. Only GPU 0 is exposed.\n',
            'The earlier 4B model revision is pinned. Prompts match the local nine-output study exactly. This notebook has not been GPU-executed.\n',
            'Download reconstruction_prose_4b_v1 in full. Outputs remain private. Execution success is not a claim-quality or style score.\n']},
            code("import os\nimport sys\nif 'torch' in sys.modules:\n    raise RuntimeError('Restart the session and run this cell before importing torch')\nos.environ['CUDA_VISIBLE_DEVICES']='0'\nos.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'\nos.environ['TOKENIZERS_PARALLELISM']='false'"),
            code('%pip install -q -U "transformers==5.20.1" "accelerate==1.15.0" "bitsandbytes==0.50.2" "huggingface-hub==1.33.0" "sentencepiece" "Pillow"'),
            code('%%writefile /kaggle/working/reconstruction_prose_4b.py\n'+source),code(launch),
            code("import json\nfrom pathlib import Path\nfolder=Path('/kaggle/working/reconstruction_prose_4b_v1')\nstate=json.loads((folder/'state.json').read_text())\nprint(json.dumps(state,indent=2))\nif not state['complete'] or not state['execution_complete'] or state['completed_jobs']!=9 or state['training_called']:\n    raise RuntimeError('Inference execution incomplete; preserve all saved outputs')\nprint('Nine first outputs saved. Review contract flags and compare claims before interpreting results.')")]}
    path=ROOT/'kaggle/reconstruction_prose_4b.ipynb'
    path.write_text(json.dumps(book,indent=2,ensure_ascii=False)+'\n')
    if path.stat().st_size>=1_000_000:raise ValueError('Kaggle notebook source size exceeded')
    print('Prepared '+str(path.relative_to(ROOT))+'; no upload or execution.')


if __name__=='__main__':build()
