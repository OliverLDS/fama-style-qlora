"""Build one self-contained inference notebook; never uploads or submits it."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def code(text):
    return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
        'source':text.rstrip().splitlines(keepends=True)}


def build():
    source=(ROOT/'src/fama_style/evaluation/multi_author_baseline.py').read_text()
    source=source[:source.rfind("\nif __name__=='__main__':")]
    launch='''
jobs = list(Path('/kaggle/input').rglob('multi_author_base_jobs.jsonl'))
configs = list(Path('/kaggle/input').rglob('multi_author_baseline_config.json'))
if len(jobs) != 1 or len(configs) != 1:
    raise RuntimeError('Attach exactly one private baseline upload Dataset containing the two prepared JSON files')
main(['--jobs', str(jobs[0]), '--config', str(configs[0]), '--output', '/kaggle/working/multi_author_base_v1'])
'''
    book={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},
        'cells':[{'cell_type':'markdown','metadata':{},'source':[
            '# Multi-author prompted-base development comparison — prepared only\n',
            'Do not execute until model execution is explicitly authorized. First complete detector-control feasibility checks.\n',
            'Upload only the two private JSON files from the milestone packet upload/ folder as a private Kaggle Dataset. Attach it, enable HF_TOKEN in Secrets, allow internet, and select GPU T4 x2. GPU 0 alone is exposed.\n',
            'This notebook runs frozen google/gemma-3-1b-it inference, ordinary versus a source-only structural prior. It does not train, load references or read reserved examples. Outputs stay private.\n']},
            code("import os\nos.environ['CUDA_VISIBLE_DEVICES'] = '0'"),
            code('%pip install -q -U "transformers==5.20.1" "accelerate>=1.0" "bitsandbytes>=0.46"'),
            code(source+'\n'+launch)]}
    path=ROOT/'kaggle/multi_author_baseline.ipynb';path.write_text(json.dumps(book,indent=2,ensure_ascii=False)+'\n')
    print('Prepared '+str(path.relative_to(ROOT))+'; no upload or execution.')


if __name__=='__main__':build()
