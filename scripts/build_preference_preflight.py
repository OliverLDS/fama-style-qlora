"""Build a private self-contained CPU tokenizer notebook; never upload or run."""
import hashlib
import json
import argparse
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def cell(source):return {'cell_type':'code','metadata':{},'source':source.splitlines(keepends=True),'outputs':[],'execution_count':None}


def build(packet=None):
    packet=packet or ROOT/'data/actual_error_preflight_v2'
    plan=json.loads((packet/'plan.json').read_text())
    source=(ROOT/'src/fama_style/evaluation/preference_token_preflight.py').read_text()
    if hashlib.sha256(source.encode()).hexdigest()!=plan['runtime_source_sha256']:raise ValueError('Frozen preflight source changed')
    expected={k:plan[k] for k in ('data_sha256','config_sha256')}
    launch='''from pathlib import Path
import hashlib
import subprocess
import sys
data=list(Path('/kaggle/input').rglob('actual_error_preflight_pairs.jsonl'))
config=list(Path('/kaggle/input').rglob('actual_error_preflight_config.json'))
if len(data)!=1 or len(config)!=1:
    raise RuntimeError('Attach only the new private tokenizer-preflight Dataset with its two files')
expected='''+repr(expected)+'''
for path,key in ((data[0],'data_sha256'),(config[0],'config_sha256')):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected[key]:raise RuntimeError('Frozen preflight payload changed')
command=[sys.executable,'/kaggle/working/preference_token_preflight.py','--data',str(data[0]),'--config',str(config[0])]
subprocess.run(command+['--check'],check=True)
subprocess.run(command+['--run-tokenizer-check','--output','/kaggle/working/actual_error_preflight_v1'],check=True)
'''
    book={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},
        'language_info':{'name':'python'}},'cells':[{'cell_type':'markdown','metadata':{},'source':[
        '# Actual-error preference preflight — CPU only\n',
        'Use a NEW PRIVATE notebook. Settings / Accelerator: None. Enable Internet and the existing HF_TOKEN Secret.\n',
        'Attach only the two files from the corrected data/actual_error_preflight_v2/upload/ packet. No adapter or old notebook outputs needed.\n',
        'This downloads six pinned tokenizer/config files, never model weights; it audits 34 response sequences and checks 17 chosen sequences against archived exact token-ID hashes.\n',
        'No GPU, generation, model scoring, optimization, training or detector calls. Synthetic loss tests are not measured model scores.\n',
        'Keep the notebook and Dataset private: the inputs contain copyrighted reference paragraphs. Download actual_error_preflight_v1/ and the notebook log after completion.\n',
        'tokenizer_assets/ is an exported credential-free tokenizer-only bundle for future local checks.\n']},
        cell("import os\nimport sys\nfrom pathlib import Path\nif 'torch' in sys.modules or Path('/dev/nvidia0').exists():\n    raise RuntimeError('Use a fresh CPU session: Accelerator None')\nos.environ['CUDA_VISIBLE_DEVICES']=''\nos.environ['USE_TORCH']='0'\nos.environ['USE_TF']='0'\nos.environ['USE_FLAX']='0'\nos.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'\nos.environ['TOKENIZERS_PARALLELISM']='false'\n"),
        cell('%pip install -q -U "transformers==5.20.1" "huggingface-hub==1.33.0" "jinja2==3.1.6" "sentencepiece"\n'),
        cell('%%writefile /kaggle/working/preference_token_preflight.py\n'+source),cell(launch)]}
    path=ROOT/'kaggle/preference_token_preflight.ipynb';path.write_text(json.dumps(book,indent=2,ensure_ascii=False)+'\n')
    if path.stat().st_size>=1_000_000:raise ValueError('Notebook exceeds Kaggle source limit')
    print('Prepared kaggle/preference_token_preflight.ipynb; CPU only, no submission.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--packet',type=Path)
    build(p.parse_args().packet)
