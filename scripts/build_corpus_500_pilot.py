"""Prepare the mixed-500 notebook and private upload files; never submit a job."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fama_style.research.corpus_500_packet import build as build_packet
from fama_style.training import corpus_500_pilot as runtime


def cell(source):
    return dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=source.rstrip().splitlines(keepends=True))


def build_notebook(packet, output):
    manifest = json.loads((packet / 'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        if hashlib.sha256((packet / name).read_bytes()).hexdigest() != digest:
            raise ValueError('frozen_packet_changed')
    config = json.loads((packet / 'upload/corpus_500_config.json').read_text())
    source = Path(runtime.__file__).read_text()
    if runtime.sha(source) != config['runtime_source_sha256']:
        raise ValueError('runtime_changed_rebuild_new_packet')
    setup = '''import os
import sys
import time
if 'torch' in sys.modules:
    raise RuntimeError('Restart the session and run this first cell before any torch import')
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
PILOT_STARTED=time.monotonic()
def remaining():
    seconds=5400-(time.monotonic()-PILOT_STARTED)
    if seconds<=0: raise TimeoutError('The 90-minute notebook cap has elapsed')
    return seconds
'''
    requirements = [k.replace('_', '-') + '==' + v for k, v in runtime.PACKAGES.items()] + ['sentencepiece', 'Pillow']
    install = "import subprocess\nsubprocess.run([sys.executable,'-m','pip','install','-q','-U']+" + repr(requirements) + ",check=True,timeout=min(900,remaining()))\n"
    expected = {name: manifest['files']['upload/' + name] for name in ['corpus_500_pairs.jsonl', 'corpus_500_config.json']}
    launch = '''from pathlib import Path
import hashlib
expected=''' + repr(expected) + '''
paths={}
for name,digest in expected.items():
    matches=list(Path('/kaggle/input').rglob(name))
    if len(matches)!=1: raise RuntimeError('Attach exactly one private corpus_500 upload Dataset')
    if hashlib.sha256(matches[0].read_bytes()).hexdigest()!=digest: raise RuntimeError('Attached data/config differs from frozen packet')
    paths[name]=matches[0]
command=[sys.executable,'/kaggle/working/corpus_500_pilot.py',
         '--data',str(paths['corpus_500_pairs.jsonl']),'--config',str(paths['corpus_500_config.json'])]
subprocess.run(command+['--check'],check=True,timeout=min(60,remaining()))
# Training starts only when the user runs this notebook through Kaggle's UI.
subprocess.run(command+['--train-authorized','--max-runtime-seconds',str(max(1,int(remaining()))),
    '--output','/kaggle/working/corpus_500_pilot_v1'],check=True,timeout=remaining())
'''
    notes = [
        '# Mixed-teacher 500-pair pilot — Gemma 3 4B\n',
        'Create a NEW PRIVATE Dataset containing only the two files in data/corpus_500_pilot_v1/upload/. Suggested title: Corpus 500 Mixed Teacher Pilot v1.\n',
        'Upload this notebook through the Kaggle UI as a NEW PRIVATE notebook and attach that Dataset. No previous adapter is needed. Enable Internet, HF_TOKEN in Kaggle Secrets, and GPU T4 x2 before Save & Run All. Code exposes GPU 0 only.\n',
        'One epoch over all 500 pairs, 250 updates, fresh pinned base, same adapter recipe and training instruction as the 200-pair pilot. Training and inference instructions differ: inference uses the agreed simple rewrite instruction.\n',
        'Evaluation: 16 original held-out inputs plus eight Gemini input variants for the same reference content. Save 24 base and 24 trained outputs. The references are used for target loss, never inserted into generation prompts.\n',
        'Inference instruction: ' + runtime.INSTRUCTION + '\n',
        'The 90-minute cap includes installation. No automatic retries, extra epochs, Gemini calls or Pangram calls run here. HF_TOKEN is retrieved from Secrets, never printed or embedded.\n',
        'Download corpus_500_pilot_v1/ and the notebook log, including evaluation.json and epoch_1/adapter/. Preserve partial outputs if anything fails. Earlier 200-pair files remain unchanged.\n',
        'After output review, detector packets will include input, base output and trained output. Start with two frozen cases (six texts); full eight-case panel is optional. Do not infer successful humanization if inputs already receive human labels.\n',
        'This mixed-teacher development pilot precedes separate Gemini/GPT-6 tracks. It does not establish distinctive Fama style or independent semantic correctness.\n']
    book = dict(nbformat=4, nbformat_minor=5, metadata=dict(
        kernelspec=dict(name='python3', display_name='Python 3', language='python'), language_info=dict(name='python')),
        cells=[dict(cell_type='markdown', metadata={}, source=notes), cell(setup), cell(install),
               cell('%%writefile /kaggle/working/corpus_500_pilot.py\n' + source), cell(launch)])
    output.write_text(json.dumps(book, indent=2, ensure_ascii=False) + '\n')
    if output.stat().st_size >= 1_000_000:
        raise ValueError('Kaggle_notebook_source_limit')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, default=ROOT / 'data/corpus_500_pilot_v1')
    parser.add_argument('--notebook', type=Path, default=ROOT / 'kaggle/corpus_500_pilot_4b.ipynb')
    args = parser.parse_args()
    if not args.packet.exists():
        build_packet(ROOT / 'data/data_scaling_v1/stage_0500_expansion_v1/training.private.jsonl',
                     ROOT / 'data/corpus_200_pilot_v1',
                     ROOT / 'research_runs/data_scaling_v1/corpus_500_pilot_4b_v1/gemini_development_v1/accepted_inputs.private.jsonl',
                     args.packet)
    build_notebook(args.packet, args.notebook)
    print('Prepared ' + str(args.notebook) + ' and ' + str(args.packet / 'upload') + '. No GPU job submitted.')


if __name__ == '__main__':
    main()
