"""Build a self-contained inference-only notebook using an existing Kaggle output."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fama_style.evaluation import corpus_200_simple_prompt as runtime


def cell(text):return {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
                       'source':text.rstrip().splitlines(keepends=True)}


def build():
    parent=ROOT/'research_runs/data_scaling_v1/corpus_200_pilot_4b_v1/raw/corpus_200_pilot_v1'
    report=parent/'evaluation.json';adapter=parent/'epoch_1/adapter'
    digest=hashlib.sha256(report.read_bytes()).hexdigest();rows,old=runtime.load_inputs(report,adapter,digest)
    setup='''import os
import sys
import time
if 'torch' in sys.modules: raise RuntimeError('Restart and run this cell before importing torch')
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
INFERENCE_STARTED=time.monotonic()
def remaining():
    value=1800-(time.monotonic()-INFERENCE_STARTED)
    if value<=0: raise TimeoutError('The 30-minute inference cap has elapsed')
    return value
'''
    requirements=[k.replace('_','-')+'=='+v for k,v in runtime.PACKAGES.items()]+['sentencepiece','Pillow']
    install="import subprocess\nsubprocess.run([sys.executable,'-m','pip','install','-q','-U']+"+repr(requirements)+",check=True,timeout=min(900,remaining()))\n"
    launch='''from pathlib import Path
import hashlib
reports=[p for p in Path('/kaggle/input').rglob('evaluation.json') if p.parent.name=='corpus_200_pilot_v1']
if len(reports)!=1: raise RuntimeError('Attach the saved output of corpus-200-pilot-4b version 1')
expected_report_sha256='''+repr(digest)+'''
if hashlib.sha256(reports[0].read_bytes()).hexdigest()!=expected_report_sha256:
    raise RuntimeError('Attached output differs from the reviewed version 1 report')
adapter=reports[0].parent/'epoch_1/adapter'
command=[sys.executable,'/kaggle/working/corpus_200_simple_prompt.py',
         '--report',str(reports[0]),'--adapter',str(adapter),'--report-sha256',expected_report_sha256]
subprocess.run(command+['--check'],check=True,timeout=min(60,remaining()))
subprocess.run(command+['--max-runtime-seconds',str(max(1,int(remaining()))),
    '--output','/kaggle/working/corpus_200_simple_prompt_v1'],check=True,timeout=remaining())
'''
    book={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}},
          'cells':[{'cell_type':'markdown','metadata':{},'source':[
              '# Simpler rewrite prompt — existing 200-pair adapter\n',
              'Create a NEW PRIVATE notebook. Add Input → Notebook Output → your corpus_200_pilot_4b version 1 output. No new Dataset upload is needed.\n',
              'Enable HF_TOKEN in Kaggle Secrets, Internet and GPU T4 x2. Code exposes GPU 0 only. Then Save & Run All.\n',
              'Inference only: the same 16 held-out inputs, pinned Gemma 3 4B base and saved adapter, unchanged greedy decoding. 16 base outputs and 16 adapter outputs; zero optimizer steps. A 30-minute cap includes installation.\n',
              'Prompt: Rewrite the following passage in natural, human-written academic economics prose. Return only the rewritten paragraph.\n',
              'Download corpus_200_simple_prompt_v1/ and the notebook log. Earlier outputs and adapter remain intact. Hold the old Pangram packets; updated scans will use these new outputs after review.\n']},cell(setup),cell(install),
              cell('%%writefile /kaggle/working/corpus_200_simple_prompt.py\n'+Path(runtime.__file__).read_text()),cell(launch)]}
    path=ROOT/'kaggle/corpus_200_simple_prompt_4b.ipynb';path.write_text(json.dumps(book,indent=2,ensure_ascii=False)+'\n')
    if path.stat().st_size>=1_000_000:raise ValueError('notebook_too_large')
    record=ROOT/'research_runs/data_scaling_v1/corpus_200_simple_prompt_v1';record.mkdir(exist_ok=True)
    runtime.write_json(record/'preparation.json',{'protocol':runtime.PROTOCOL,'status':'prepared_not_executed',
        'parent_notebook':'oliverlds/corpus-200-pilot-4b','parent_version':1,'parent_evaluation_sha256':digest,
        'runtime_source_sha256':runtime.file_sha(Path(runtime.__file__)),'notebook_sha256':runtime.file_sha(path),
        'instruction':runtime.INSTRUCTION,'decoding':runtime.DECODING,'model_revision':runtime.REVISION,
        'development_ids':[r['example_id'] for r in rows],'detector_case_ids':old['config']['detector_case_ids'],
        'expected_generations':32,'max_seconds':1800,'training_authorized':False,'optimizer_steps':0,
        'automatic_submission':False,'GPU_jobs_submitted':0})
    print('Prepared '+str(path)+'; attach existing notebook output, no new data upload. No GPU job submitted.')


if __name__=='__main__':build()
