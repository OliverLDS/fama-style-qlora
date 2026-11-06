"""Prepare a manual-upload, inference-only input-route and prompt diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fama_style.evaluation import purpose_500_prompt_input as runtime

PREP=ROOT/'research_runs/data_scaling_v1/purpose_500_prompt_input_v1'
PARENT=ROOT/'research_runs/data_scaling_v1/gpt6_purpose_500_4b_v1/raw/gpt6_purpose_500_v1'
PACKET=ROOT/'data/purpose_500_prompt_input_v1'
DATA='purpose_500_diagnostic_inputs.jsonl'
CONFIG='purpose_500_diagnostic_config.json'
NOTEBOOK=ROOT/'kaggle/purpose_500_prompt_input_4b.ipynb'


def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def write_rows(p,r):p.write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in r))


def build_packet():
    if PACKET.exists():raise ValueError('refuse_overwrite_packet')
    manifest=read(PREP/'generation_manifest.json')
    for name,h in manifest['files'].items():
        if runtime.file_sha(PREP/name)!=h:raise ValueError('teacher_preparation_changed')
    report=read(PARENT/'evaluation.json')
    original={r['example_id']:r for r in report['snapshots'][0]['generations']}
    selected=rows(PREP/'selected_contents.private.jsonl');fresh={r['content_id']:r for r in rows(PREP/'fresh_inputs.private.jsonl')}
    if len(selected)!=4 or set(fresh)!={r['content_id'] for r in selected}:raise ValueError('four_fresh_contents_required')
    output=[]
    for r in selected:
        cid=r['content_id']
        if fresh[cid]['target_sha256']!=runtime.sha(original[cid]['reference']):raise ValueError('fresh_reference_changed')
        for route in runtime.ROUTES:
            source=fresh[cid]['source'] if route=='fresh_reconstruction' else original[cid if route=='existing_input' else cid+'::gemini_input_v1']['source']
            for condition in runtime.PROMPTS:
                output.append(dict(example_id=cid+'::'+route+'::'+condition,content_id=cid,
                    paper_id=r['paper_id'],style_group=r['style_group'],reference=r['reference'],
                    reference_sha256=r['target_sha256'],source=source,source_sha256=runtime.sha(source),
                    input_condition=route,prompt_condition=condition,prompt_sha256=runtime.sha(runtime.prompt(source,condition)),
                    canonical_pdf_sha256=report['config']['canonical_pdf_by_paper'][r['paper_id']]))
    PACKET.mkdir(parents=True);(PACKET/'upload').mkdir()
    data=PACKET/'upload'/DATA;write_rows(data,output)
    config=dict(protocol=runtime.PROTOCOL,model_id=runtime.MODEL_ID,model_revision=runtime.REVISION,
        prompts=runtime.PROMPTS,decoding=runtime.DECODING,packages=runtime.PACKAGES,max_runtime_seconds=runtime.MAX_SECONDS,
        training_authorized=False,runtime_source_sha256=runtime.file_sha(Path(runtime.__file__)),dataset_sha256=runtime.file_sha(data),
        parent_report_sha256=runtime.file_sha(PARENT/'evaluation.json'),selected_content_ids=[r['content_id'] for r in selected],
        job_ids=[r['example_id'] for r in output],fresh_source_sha256_by_content={k:v['source_sha256'] for k,v in fresh.items()},
        expected_generations=48,optimizer_steps=0,automatic_submission=False,
        teacher_limitation='Fresh reconstructions use Gemini 3 Flash, not GPT-6; compare prompts within each route and do not isolate procedure from teacher as a causal factor.')
    runtime.write_json(PACKET/'upload'/CONFIG,config)
    runtime.load_inputs(PARENT/'evaluation.json',PARENT/'epoch_1/adapter',data,PACKET/'upload'/CONFIG)
    runtime.write_json(PACKET/'manifest.json',dict(protocol=runtime.PROTOCOL,
        generation_manifest_sha256=runtime.file_sha(PREP/'generation_manifest.json'),
        builder_sha256=runtime.file_sha(Path(__file__)),runtime_sha256=config['runtime_source_sha256'],
        files={str(p.relative_to(PACKET)):runtime.file_sha(p) for p in sorted(PACKET.rglob('*')) if p.is_file()}))


def check():
    m=read(PACKET/'manifest.json')
    if m['builder_sha256']!=runtime.file_sha(Path(__file__)) or m['runtime_sha256']!=runtime.file_sha(Path(runtime.__file__)):raise ValueError('code_changed_new_version_required')
    if m['generation_manifest_sha256']!=runtime.file_sha(PREP/'generation_manifest.json'):raise ValueError('generation_manifest_changed')
    for name,h in m['files'].items():
        if runtime.file_sha(PACKET/name)!=h:raise ValueError('packet_changed')
    return runtime.load_inputs(PARENT/'evaluation.json',PARENT/'epoch_1/adapter',PACKET/'upload'/DATA,PACKET/'upload'/CONFIG)


def notebook():
    check()
    def cell(s):return dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=s.splitlines(keepends=True))
    setup='''import os,sys,time
if 'torch' in sys.modules: raise RuntimeError('Restart and run this first cell before torch')
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
STARTED=time.monotonic()
def remaining():
    seconds=2700-(time.monotonic()-STARTED)
    if seconds<=0: raise TimeoutError('45-minute inference cap elapsed')
    return seconds
'''
    requirements=[k.replace('_','-')+'=='+v for k,v in runtime.PACKAGES.items()]+['sentencepiece','Pillow']
    install="import subprocess\nsubprocess.run([sys.executable,'-m','pip','install','-q','-U']+"+repr(requirements)+",check=True,timeout=min(900,remaining()))\n"
    expected={name:runtime.file_sha(PACKET/'upload'/name) for name in [DATA,CONFIG]}
    launch='''from pathlib import Path
import hashlib
expected='''+repr(expected)+'''
paths={}
for name,digest in expected.items():
    matches=list(Path('/kaggle/input').rglob(name))
    if len(matches)!=1: raise RuntimeError('Attach exactly one private Purpose 500 Prompt Input v1 Dataset')
    if hashlib.sha256(matches[0].read_bytes()).hexdigest()!=digest: raise RuntimeError('Diagnostic data/config hash mismatch')
    paths[name]=matches[0]
reports=[p for p in Path('/kaggle/input').rglob('evaluation.json') if p.parent.name=='gpt6_purpose_500_v1']
if len(reports)!=1: raise RuntimeError('Attach output of oliverlds/gpt6-purpose-500-4b version 1')
if hashlib.sha256(reports[0].read_bytes()).hexdigest()!='''+repr(runtime.file_sha(PARENT/'evaluation.json'))+''':
    raise RuntimeError('Wrong parent version or changed report')
command=[sys.executable,'/kaggle/working/purpose_500_prompt_input.py','--report',str(reports[0]),
 '--adapter',str(reports[0].parent/'epoch_1/adapter'),'--data',str(paths['''+repr(DATA)+''']),
 '--config',str(paths['''+repr(CONFIG)+'''])]
subprocess.run(command+['--check'],check=True,timeout=min(60,remaining()))
subprocess.run(command+['--max-runtime-seconds',str(max(1,int(remaining()))),
 '--output','/kaggle/working/purpose_500_prompt_input_v1'],check=True,timeout=remaining())
'''
    notes='''# Saved 500-pair adapter: input-route × instruction diagnostic

Inference only, 48 outputs, zero training steps. Four fixed held-out contents, one per author group; three sources (existing GPT-6, existing Gemini, fresh Gemini brief reconstruction); two instructions (current simple rewrite, explicit re-expression); both pinned base and saved adapter. Same greedy decoding throughout.

Create a NEW PRIVATE Dataset from the two files in data/purpose_500_prompt_input_v1/upload/. Suggested title: Purpose 500 Prompt Input v1.
Upload this notebook as NEW PRIVATE and attach that Dataset PLUS notebook output oliverlds/gpt6-purpose-500-4b version 1 (Add Input → Notebook Output). Enable Internet, HF_TOKEN in Secrets and GPU T4 x2. Only GPU 0 is visible. Then Save & Run All.

References are saved for evaluation only, never generation messages. Fresh reconstructions were generated automatically using the user's confirmed free Gemini project, from topic, purpose, notes and anchors. This is procedure-matched, not GPT-6-teacher-matched. Raw annotations and mechanical format-conversion audit stay private. A different input or instruction may help; no improvement is presumed.

Both prompts retain the simple return-only-paragraph format. The second explicitly asks for different wording and sentence structure, without an edit-percentage target or a long preservation checklist. The same frozen adapter is used; no optimizer or backward pass exists. Generations, prompt hashes, input routes and prompt conditions are recorded in evaluation.json. The 45-minute cap includes installation; no retries or detector calls.

Download purpose_500_prompt_input_v1/ and the execution log. Preserve partial files if a failure occurs. Preparing this notebook has not submitted a Kaggle job.
'''
    book=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),language_info=dict(name='python')),
        cells=[dict(cell_type='markdown',metadata={},source=notes.splitlines(keepends=True)),cell(setup),cell(install),
               cell('%%writefile /kaggle/working/purpose_500_prompt_input.py\n'+Path(runtime.__file__).read_text()),cell(launch)])
    NOTEBOOK.write_text(json.dumps(book,indent=2,ensure_ascii=False)+'\n')
    if NOTEBOOK.stat().st_size>=1_000_000:raise ValueError('Kaggle_source_limit')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args()
    if not a.check:
        if not PACKET.exists():build_packet()
        notebook()
    check();print('Verified 48-generation inference-only notebook and private packet. No Kaggle job submitted.')
