"""Verify and summarize the saved-adapter input-route/prompt experiment offline."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from fama_style.evaluation import purpose_500_prompt_input as runtime
from fama_style.research.neutral_reconstruction_results import metrics

DOWNLOAD=ROOT/'research_runs/data_scaling_v1/purpose_500_prompt_input_result_v1'
PACKET=ROOT/'data/purpose_500_prompt_input_v1'
PARENT=ROOT/'research_runs/data_scaling_v1/gpt6_purpose_500_4b_v1/raw/gpt6_purpose_500_v1'


def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def write_rows(p,items):p.write_text(''.join(json.dumps(r,sort_keys=True,ensure_ascii=False)+'\n' for r in items))


def repeated_sentences(text):
    s=[re.sub(r'\s+',' ',x).strip().casefold() for x in re.split(r'(?<=[.!?])\s+',text) if len(x.split())>=6]
    return len(s)-len(set(s))


def aggregate(items):
    return dict(outputs=len(items),empty=sum(r['empty'] for r in items),
        output_limited=sum(r['output_limited'] for r in items),
        exact_word_copies=sum(r['source_metrics']['word_token_copy'] for r in items),
        high_input_overlap=sum(r['source_metrics']['word_sequence_overlap']>=.8 for r in items),
        repeated_long_sentence_outputs=sum(repeated_sentences(r['output'])>0 for r in items),
        multiple_paragraph_outputs=sum(bool(re.search(r'\n\s*\n',r['output'].strip())) for r in items),
        median_source_overlap=statistics.median(r['source_metrics']['word_sequence_overlap'] for r in items),
        median_reference_overlap=statistics.median(r['reference_metrics']['word_sequence_overlap'] for r in items),
        median_source_word_length_ratio=statistics.median(r['source_metrics']['length_ratio'] for r in items))


def verify(download=DOWNLOAD,packet=PACKET,parent=PARENT):
    receipt=read(download/'download_manifest.json');raw=download/'raw';folder=raw/runtime.PROTOCOL
    if receipt['slug']!='oliverlds/purpose-500-prompt-input-4b' or receipt['version']!=1 or not receipt['version_explicitly_verified_via_SDK']:
        raise ValueError('exact_version_required')
    for name,h in receipt['files'].items():
        if runtime.file_sha(raw/name)!=h:raise ValueError('download_changed')
    inputs,original=runtime.load_inputs(parent/'evaluation.json',parent/'epoch_1/adapter',
        packet/'upload/purpose_500_diagnostic_inputs.jsonl',packet/'upload/purpose_500_diagnostic_config.json')
    expected={r['example_id']:r for r in inputs}
    report=read(folder/'evaluation.json');state=read(folder/'state.json')
    for r in [report,state]:
        if r['protocol']!=runtime.PROTOCOL or not r['complete'] or r['training_called'] or r['optimizer_steps']!=0:
            raise ValueError('incomplete_or_training_called')
    if state['generations']!=48 or not report['all_weights_frozen'] or not report['adapter_hashes_unchanged']:
        raise ValueError('frozen_inference_contract')
    if report['instructions']!=runtime.PROMPTS or report['decoding']!=runtime.DECODING or report['previous_adapter_files_sha256']!=original['adapter_files_sha256']:
        raise ValueError('prompt_decode_or_adapter_changed')
    actual_runtime=report['runtime']
    if actual_runtime['runtime_source_sha256']!=runtime.file_sha(Path(runtime.__file__)) or runtime.file_sha(raw/'purpose_500_prompt_input.py')!=actual_runtime['runtime_source_sha256']:
        raise ValueError('runtime_source_changed')
    if actual_runtime['model_id']!=runtime.MODEL_ID or actual_runtime['model_revision']!=runtime.REVISION or actual_runtime['gpu_count']!=1 or 'T4' not in actual_runtime['gpu']:
        raise ValueError('model_or_hardware_changed')
    if actual_runtime['packages']!=original['runtime']['packages']:raise ValueError('packages_changed')
    audits=read(folder/'prompt_audit.json')
    if [r['example_id'] for r in audits]!=list(expected):raise ValueError('prompt_audit_coverage')
    for a in audits:
        r=expected[a['example_id']]
        if a['source_sha256']!=r['source_sha256'] or a['prompt_sha256']!=r['prompt_sha256'] or a['reference_in_messages'] or a['truncated'] or not 0<a['prompt_tokens']<=1024:
            raise ValueError('prompt_boundary_or_reference_leakage')
    items=[]
    if [s['name'] for s in report['snapshots']]!=['base','epoch_1']:raise ValueError('snapshots_changed')
    for snapshot in report['snapshots']:
        if not snapshot['complete'] or [r['example_id'] for r in snapshot['generations']]!=list(expected):raise ValueError('generation_coverage')
        for g in snapshot['generations']:
            row=expected[g['example_id']]
            if any(g.get(k)!=v for k,v in row.items()) or g['output_sha256']!=runtime.sha(g['output']):raise ValueError('generation_text_binding')
            if g['empty']!=(not bool(g['output'].strip())):raise ValueError('empty_flag_changed')
            items.append(dict(g,snapshot=snapshot['name'],source_metrics=metrics(g['source'],g['output']),reference_metrics=metrics(g['reference'],g['output'])))
    logs=list(raw.glob('*.log'))
    if len(logs)!=1:raise ValueError('log_coverage')
    if any(re.search(p,logs[0].read_text()) for p in [r'hf_[A-Za-z0-9]{20,}',r'KGAT_[A-Za-z0-9_\-]{15,}',r'AKIA[0-9A-Z]{16}']):
        raise ValueError('credential_pattern_requires_private_remediation')
    return report,state,items,original


def build():
    output=DOWNLOAD/'review_v1'
    if output.exists():raise ValueError('refuse_overwrite_review')
    report,state,items,original=verify()
    notes=rows(DOWNLOAD/'reading_notes.private.jsonl');index={(r['snapshot'],r['example_id']):r for r in notes}
    if len(notes)!=48 or len(index)!=48:raise ValueError('reading_note_coverage')
    for r in items:
        note=index[r['snapshot'],r['example_id']]
        if note['output_sha256']!=r['output_sha256']:raise ValueError('reading_note_binding')
        r['reading_note']=note
    summary=dict(protocol='purpose_500_prompt_input_review_v1',version=1,execution_complete=True,
        elapsed_seconds=state['elapsed_seconds'],peak_GPU_bytes=state['peak_gpu_allocated_bytes'],
        underlying_contents=4,generations=48,optimizer_steps=0,
        conditions={s:{route:{p:aggregate([r for r in items if r['snapshot']==s and r['input_condition']==route and r['prompt_condition']==p])
                    for p in runtime.PROMPTS} for route in runtime.ROUTES} for s in ['base','epoch_1']},
        paired_prompt_changes={},historical_simple_reproduction={},
        assessment_provenance='Same-agent unblinded reading of all 48 outputs; notes are not independent or a calibrated semantic/style score.',
        semantic_error_rate=None,author_style_score=None,detector_results=None,new_GPU_jobs=0,new_model_calls=0,new_detector_calls=0)
    by={(r['snapshot'],r['content_id'],r['input_condition'],r['prompt_condition']):r for r in items}
    for s in ['base','epoch_1']:
        for route in runtime.ROUTES:
            pairs=[(by[s,c,route,'simple'],by[s,c,route,'reexpress']) for c in {r['content_id'] for r in items}]
            summary['paired_prompt_changes'][s+'::'+route]=dict(
                lower_source_overlap=sum(b['source_metrics']['word_sequence_overlap']<a['source_metrics']['word_sequence_overlap'] for a,b in pairs),
                higher_source_overlap=sum(b['source_metrics']['word_sequence_overlap']>a['source_metrics']['word_sequence_overlap'] for a,b in pairs))
        old={r['example_id']:r for snapshot in original['snapshots'] if snapshot['name']==s for r in snapshot['generations']}
        controls=[r for r in items if r['snapshot']==s and r['prompt_condition']=='simple' and r['input_condition']!='fresh_reconstruction']
        summary['historical_simple_reproduction'][s]=dict(outputs=len(controls),byte_identical=sum(r['output_sha256']==old[r['content_id'] if r['input_condition']=='existing_input' else r['content_id']+'::gemini_input_v1']['output_sha256'] for r in controls))
    output.mkdir();write_rows(output/'outputs_and_reading_notes.private.jsonl',items)
    runtime.write_json(output/'summary.json',summary)
    runtime.write_json(output/'manifest.json',dict(builder_sha256=runtime.file_sha(Path(__file__)),
        download_manifest_sha256=runtime.file_sha(DOWNLOAD/'download_manifest.json'),reading_notes_sha256=runtime.file_sha(DOWNLOAD/'reading_notes.private.jsonl'),
        files={str(p.relative_to(output)):runtime.file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}))
    return summary


if __name__=='__main__':print(json.dumps(build(),indent=2))
