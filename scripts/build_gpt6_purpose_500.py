"""Freeze the user-adjudicated GPT-6 500-pair release and manual-upload pilot.

No provider calls, model downloads, Kaggle uploads or GPU execution.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fama_style.training import gpt6_purpose_500 as runtime

RELEASE = ROOT / 'data/gpt6_purpose_500_v1'
AUDIT = ROOT / 'research_runs/data_scaling_v1/gpt6_purpose_only_0500_v1/output_audit_v1'
PARENT = ROOT / 'data/corpus_500_pilot_v1'
PAIR_FILE = 'gpt6_purpose_500_pairs.jsonl'
CONFIG_FILE = 'gpt6_purpose_500_config.json'
OLD_CLAUSE = '; yet in our model bonds command a liquidity premium and trade at a discount relative to stocks.'
NEW_CLAUSE = '. In our model, bonds command a liquidity premium, resulting in higher prices and lower expected returns relative to stocks.'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_rows(path, values):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in values))


def verify_files(directory, files):
    for name, expected in files.items():
        if digest(directory / name) != expected:
            raise ValueError('frozen_file_changed: ' + name)


def adjudicate(candidates):
    if [r['id'] for r in candidates] != ['P%03d' % i for i in range(1, 501)]:
        raise ValueError('exact_500_target_slots_required')
    final = copy.deepcopy(candidates)
    audit = []
    decisions = {
        'P139': 'User-authorized explicit pricing/expected-return clarification in target; teacher prose retained.',
        'P400': 'User accepts subsequent-return framing. Historical timing warning retained; acceptance is not independent verification.',
        'P056': 'Prior source-context warning adjudicated nonblocking; no text change or regeneration.',
        'P206': 'Prior conditional-parameter warning adjudicated nonblocking; no text change or regeneration.',
    }
    for original, r in zip(candidates, final):
        if runtime.sha(r['source']) != r['source_sha256'] or runtime.sha(r['target']) != r['target_sha256']:
            raise ValueError('candidate_text_hash_mismatch')
        if r['id'] == 'P139':
            if r['target'].count(OLD_CLAUSE) != 1:
                raise ValueError('P139_expected_original_clause_missing')
            r['target'] = r['target'].replace(OLD_CLAUSE, NEW_CLAUSE)
            r['original_paragraph'] = r['target']
            r['target_sha256'] = runtime.sha(r['target'])
            r['provenance']['target_kind'] = 'user_authorized_pricing_clarification'
        r['provenance'].update(previous_target_sha256=original['target_sha256'], release_protocol=runtime.PROTOCOL)
        r['quality']['historical_candidate_followup'] = original['quality'].get('followup')
        r['quality'].update(text_pair_status='Released · GPT-6 purpose-only 500 · bounded pilot',
                            release_status='user_authorized_bounded_pilot',
                            independent_semantic_verification=False)
        r['quality']['followup'] = dict(action='retain_for_bounded_pilot', level='nonblocking',
            reason=decisions.get(r['id'], 'Retain observed prose and historical scope advisories; no score-based filtering.'))
        r['training_ready'] = True
        audit.append(dict(id=r['id'], decision=r['quality']['followup']['reason'],
            source_sha256_before=original['source_sha256'], source_sha256_after=r['source_sha256'],
            target_sha256_before=original['target_sha256'], target_sha256_after=r['target_sha256'],
            previous_target=original['target'] if r['id']=='P139' else None,
            revised_target=r['target'] if r['id']=='P139' else None,
            historical_followup=original['quality'].get('followup'),
            independent_semantic_verification=False))
    return final, audit


def build(output=RELEASE):
    if output.exists():
        raise ValueError('refuse_overwrite_frozen_release')
    am = json.loads((AUDIT / 'manifest.json').read_text())
    verify_files(AUDIT, am['files'])
    verify_files(AUDIT.parent, am['raw_response_files'])
    verify_files(ROOT, am['dependencies'])
    pm = json.loads((PARENT / 'manifest.json').read_text())
    verify_files(PARENT, pm['files'])
    candidates = rows(AUDIT / 'candidates_0500.private.jsonl')
    final, audit = adjudicate(candidates)
    old_config = json.loads((PARENT / 'upload/corpus_500_config.json').read_text())
    # References and development prose stay byte-for-byte unchanged; prompts alone change.
    dev = [dict(r, prompt=runtime.prompt(r['source'])) for r in rows(PARENT / 'upload/corpus_500_pairs.jsonl') if r['split']=='development']
    export = [dict(example_id=r['id'], content_id=r['id'], parent_id=r['parent_target_id'],
        paper_id=r['paper_id'], style_group=r['style_group'], authors=r['authors'], split='training',
        source=r['source'], target=r['target'], source_sha256=r['source_sha256'], target_sha256=r['target_sha256'],
        prompt=runtime.prompt(r['source'], training=True), pilot_eligible=True,
        input_condition='training_gpt6_purpose_only') for r in final] + dev
    inventory = json.loads((ROOT / 'docs/author_inventory_v3/inventory.json').read_text())
    registry = {r['paper_id']:r['pdf_sha256'] for r in inventory['papers']}
    pdfs = {r['paper_id']:r['provenance']['canonical_pdf_sha256'] for r in final}
    for r in dev:
        expected = registry[r['paper_id']]
        if r['paper_id'] in pdfs and pdfs[r['paper_id']] != expected:
            raise ValueError('paper_identity_conflict')
        pdfs[r['paper_id']] = expected
    output.mkdir(parents=True)
    (output / 'upload').mkdir()
    write_rows(output / 'training_0500.private.jsonl', final)
    write_rows(output / 'adjudications.private.jsonl', audit)
    data = output / 'upload' / PAIR_FILE
    write_rows(data, export)
    config = dict(old_config, protocol=runtime.PROTOCOL, runtime_source_sha256=digest(Path(runtime.__file__)),
        dataset_sha256=digest(data), training_ids=[r['id'] for r in final],
        training_probe_ids=[next(r['id'] for r in final if r['style_group']==g) for g in runtime.GROUPS],
        training_instruction=runtime.TRAIN_INSTRUCTION, inference_instruction=runtime.INSTRUCTION,
        training_cohort_sha256=digest(output / 'training_0500.private.jsonl'),
        canonical_pdf_by_paper=pdfs,
        teacher_track='GPT-6 purpose-only reconstruction; requested High effort, actual settings unverified',
        adjudications_sha256=digest(output / 'adjudications.private.jsonl'),
        parent_mixed_500_manifest_sha256=digest(PARENT / 'manifest.json'),
        evaluation_limits='Previously exposed paper-held-out development; 16 existing inputs and 8 Gemini variants. One target clarification and a changed training prompt prevent isolated causal comparison with mixed-500. Automated metrics are proxies, not author-style or detector success.')
    # Remove stale inherited lineage references, retain the fixed development panel and recipe.
    for key in ('parent_200_packet_manifest_sha256','Gemini_accepted_manifest_sha256'):
        config.pop(key, None)
    runtime.write_json(output / 'upload' / CONFIG_FILE, config)
    runtime.load_inputs(data, output / 'upload' / CONFIG_FILE)
    runtime.write_json(output / 'plan.json', dict(protocol=runtime.PROTOCOL, status='prepared_not_executed',
        training_pairs=500, training_group_counts=dict(Counter(r['style_group'] for r in final)),
        source_text_changes=0, target_text_changes=1, records_discarded=0,
        epochs=1, optimizer_steps=250, development_input_variants=24, matched_generations=48,
        max_runtime_seconds=5400, GPU='one T4', GPU_jobs_submitted=0,
        reserved_test_examples_read=0, input='AI prose only; no brief or reference in user prompt',
        target='human prose with user-authorized P139 clarification',
        historical_audits_preserved=True, independent_semantic_verification=False,
        cross_track_limit='P139 clarified here only; Gemini target registry remains unchanged.'))
    dependencies={str(p.relative_to(ROOT)):digest(p) for p in
        [AUDIT/'manifest.json',AUDIT/'candidates_0500.private.jsonl',PARENT/'manifest.json',
         ROOT/'docs/author_inventory_v3/inventory.json',Path(runtime.__file__),Path(__file__)]}
    runtime.write_json(output / 'manifest.json', dict(protocol=runtime.PROTOCOL, dependencies=dependencies,
        files={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()}))
    return output


def check(packet):
    manifest = json.loads((packet / 'manifest.json').read_text())
    verify_files(packet, manifest['files'])
    verify_files(ROOT, manifest['dependencies'])
    splits, config = runtime.load_inputs(packet / 'upload' / PAIR_FILE, packet / 'upload' / CONFIG_FILE)
    original = rows(AUDIT / 'candidates_0500.private.jsonl')
    final, audit = adjudicate(original)
    if final != rows(packet/'training_0500.private.jsonl') or audit != rows(packet/'adjudications.private.jsonl'):
        raise ValueError('release_adjudication_changed')
    for a,b in zip(final,splits['training']):
        if (a['id'],a['source_sha256'],a['target_sha256'])!=(b['example_id'],b['source_sha256'],b['target_sha256']):
            raise ValueError('release_export_binding')
    return splits, config


def notebook(packet, output):
    check(packet)
    manifest=json.loads((packet/'manifest.json').read_text())
    expected={name:manifest['files']['upload/'+name] for name in (PAIR_FILE,CONFIG_FILE)}
    def cell(s):
        return dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=s.splitlines(keepends=True))
    setup="""import os, sys, time
if 'torch' in sys.modules: raise RuntimeError('Restart and run this first cell before torch')
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
PILOT_STARTED=time.monotonic()
def remaining():
    seconds=5400-(time.monotonic()-PILOT_STARTED)
    if seconds<=0: raise TimeoutError('90-minute notebook cap elapsed')
    return seconds
"""
    requirements=[k.replace('_','-')+'=='+v for k,v in runtime.PACKAGES.items()]+['sentencepiece','Pillow']
    install="import subprocess\nsubprocess.run([sys.executable,'-m','pip','install','-q','-U']+"+repr(requirements)+",check=True,timeout=min(900,remaining()))\n"
    launch="""from pathlib import Path
import hashlib
expected="""+repr(expected)+"""
paths={}
for name,digest in expected.items():
    matches=list(Path('/kaggle/input').rglob(name))
    if len(matches)!=1: raise RuntimeError('Attach exactly one private GPT6 Purpose 500 v1 Dataset')
    if hashlib.sha256(matches[0].read_bytes()).hexdigest()!=digest: raise RuntimeError('Dataset/config hash mismatch')
    paths[name]=matches[0]
command=[sys.executable,'/kaggle/working/gpt6_purpose_500.py','--data',str(paths['"""+PAIR_FILE+"""']),'--config',str(paths['"""+CONFIG_FILE+"""'])]
subprocess.run(command+['--check'],check=True,timeout=min(60,remaining()))
# Execution requires the user's manual Kaggle Save & Run All.
subprocess.run(command+['--train-authorized','--max-runtime-seconds',str(max(1,int(remaining()))),
    '--output','/kaggle/working/gpt6_purpose_500_v1'],check=True,timeout=remaining())
"""
    notes='''# GPT-6 purpose-only 500 — Gemma 3 4B

Create a NEW PRIVATE Kaggle Dataset from the two JSON files in data/gpt6_purpose_500_v1/upload/ (pairs JSONL and config JSON). Suggested title: GPT6 Purpose 500 v1. Keep all prose private.

Upload this notebook as a NEW PRIVATE notebook, attach that Dataset, enable Internet and HF_TOKEN in Kaggle Secrets, and select GPU T4 x2. Code exposes GPU 0 only. Then Save & Run All. No previous adapter is required. Uploading this file through the UI does not itself authorize automatic CLI submission.

AI reconstruction prose → human paragraph, all 500 pairs, 125 per author group. No briefs or original reference text are supplied in generation prompts. P139 uses the user-authorized expected-return clarification; P400 is retained by user acceptance. Earlier warnings and raw outputs remain in the private audit.

Fresh pinned google/gemma-3-4b-it, one epoch, 250 updates, same stable FP32/NF4 QLoRA recipe as the successful mixed-500 pilot. Same simple rewrite instruction for training and evaluation. This changes both training data and training prompt versus the historical mixed-500 run.

24 development inputs (16 prior inputs + 8 Gemini variants), 24 base and 24 trained generations; target-only loss, copying diagnostics and adapter reload checks. No reserved test examples. These are reused development examples, not an independent final test.

HF_TOKEN is retrieved from Kaggle Secrets and never printed or embedded. Accept the model's access conditions on Hugging Face using that account. The 90-minute cap includes installation. No automatic retry, extra epochs, API inference or detector calls.

Download gpt6_purpose_500_v1/ and the execution log after completion, including evaluation.json, state.json, label_audit.json, training_log.json and epoch_1/adapter/. Preserve partial files on failure. No new GPU run has been submitted by preparing this notebook.
'''
    book=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python'),language_info=dict(name='python')),
        cells=[dict(cell_type='markdown',metadata={},source=notes.splitlines(keepends=True)),cell(setup),cell(install),
               cell('%%writefile /kaggle/working/gpt6_purpose_500.py\n'+Path(runtime.__file__).read_text()),cell(launch)])
    text=json.dumps(book,indent=2,ensure_ascii=False)+'\n'
    if len(text.encode())>=1_000_000: raise ValueError('Kaggle_source_size_limit')
    output.write_text(text)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check',action='store_true')
    args=p.parse_args()
    if not args.check:
        if not RELEASE.exists(): build()
        notebook(RELEASE,ROOT/'kaggle/gpt6_purpose_500_4b.ipynb')
    check(RELEASE)
    print('Verified 500 GPT-6 training pairs, 24 development inputs; no GPU job submitted.')


if __name__=='__main__': main()
