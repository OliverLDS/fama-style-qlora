"""CPU review and matched scan packets for the saved-adapter prompt comparison."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from fama_style.evaluation import corpus_200_simple_prompt as runtime
from fama_style.research.neutral_reconstruction_results import metrics


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, value): path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def aggregate(rows):
    return {
        'outputs': len(rows),
        'material_claim_issues_identified': sum(r['assessment']['material_claim_change'] for r in rows),
        'outputs_with_review_notes': sum(bool(r['assessment']['notes']) for r in rows),
        'empty': sum(r['empty'] for r in rows),
        'output_limited': sum(r['output_limited'] for r in rows),
        'median_source_word_sequence_overlap': statistics.median(r['source_metrics']['word_sequence_overlap'] for r in rows),
        'median_reference_word_sequence_overlap': statistics.median(r['reference_metrics']['word_sequence_overlap'] for r in rows),
        'median_longest_shared_source_word_run': statistics.median(r['source_metrics']['longest_shared_word_run'] for r in rows),
        'outputs_source_overlap_at_least_point8': sum(r['source_metrics']['word_sequence_overlap'] >= .8 for r in rows),
        'exact_word_token_copies': sum(r['source_metrics']['word_token_copy'] for r in rows),
    }


def verify(download, parent):
    receipt = read(download / 'download_manifest.json')
    if (receipt['slug'] != 'oliverlds/corpus-200-simple-prompt-4b' or receipt['version'] != 1
            or not receipt['version_explicitly_verified_via_SDK']):
        raise ValueError('exact_version_required')
    raw = download / 'raw'
    for name, digest in receipt['files'].items():
        if sha(raw / name) != digest: raise ValueError('download_changed')
    prep = read(download / 'preparation.json')
    parent_report = parent / 'raw/corpus_200_pilot_v1/evaluation.json'
    inputs, original = runtime.load_inputs(parent_report, parent_report.parent / 'epoch_1/adapter',
                                          prep['parent_evaluation_sha256'])
    folder = raw / runtime.PROTOCOL
    report, state = read(folder / 'evaluation.json'), read(folder / 'state.json')
    if (report['protocol'] != runtime.PROTOCOL or not report['complete'] or not state['complete']
            or report['training_called'] is not False or state['training_called'] is not False
            or report['optimizer_steps'] != 0 or state['optimizer_steps'] != 0
            or state['generations'] != 32 or not report['adapter_hashes_unchanged']
            or not report['all_weights_frozen']):
        raise ValueError('completed_inference_only_run_required')
    if (report['instruction'] != runtime.INSTRUCTION or report['decoding'] != original['config']['decoding']
            or report['previous_development_ids'] != prep['development_ids']
            or report['previous_detector_case_ids'] != original['config']['detector_case_ids']
            or report['previous_adapter_files_sha256'] != original['adapter_files_sha256']):
        raise ValueError('frozen_comparison_changed')
    for key in ['model_id', 'model_revision', 'packages', 'gpu', 'gpu_count']:
        if report['runtime'][key] != original['runtime'][key]: raise ValueError('runtime_changed')
    if (sha(raw / 'corpus_200_simple_prompt.py') != prep['runtime_source_sha256']
            or report['runtime']['runtime_source_sha256'] != prep['runtime_source_sha256']):
        raise ValueError('runtime_source_changed')
    audit = read(folder / 'prompt_audit.json')
    if len(audit) != 16 or [a['example_id'] for a in audit] != prep['development_ids']:
        raise ValueError('prompt_audit_coverage')
    for row, a in zip(inputs, audit):
        if (a['reference_in_messages'] is not False or a['truncated']
                or a['source_sha256'] != runtime.sha(row['source'])
                or a['prompt_sha256'] != runtime.sha(runtime.prompt(row['source']))
                or not 0 < a['prompt_tokens'] <= 1024):
            raise ValueError('prompt_audit_changed')
    generation = read(folder / 'generation_config.json')
    old_generation = read(parent_report.parent / 'generation_config.json')
    for key in list(runtime.DECODING) + ['eos_token_id', 'pad_token_id', 'use_cache']:
        if generation.get(key) != old_generation.get(key): raise ValueError('decoding_changed')
    if [s['name'] for s in report['snapshots']] != ['base', 'epoch_1']:
        raise ValueError('two_conditions_required')
    for snap in report['snapshots']:
        if not snap['complete'] or len(snap['generations']) != 16: raise ValueError('snapshot_incomplete')
        for row, generated in zip(inputs, snap['generations']):
            if any(generated[k] != row[k] for k in row): raise ValueError('matched_input_changed')
            if (generated['output_sha256'] != runtime.sha(generated['output'])
                    or generated['source_sha256'] != runtime.sha(row['source'])
                    or generated['prompt_sha256'] != runtime.sha(runtime.prompt(row['source']))):
                raise ValueError('output_binding_changed')
    logs = list(raw.glob('*.log'))
    if len(logs) != 1: raise ValueError('one_execution_log_required')
    log = logs[0].read_text()
    hits = sum(len(re.findall(p, log)) for p in
               [r'hf_[A-Za-z0-9]{20,}', r'KGAT_[A-Za-z0-9_\-]{15,}', r'AKIA[0-9A-Z]{16}'])
    if hits: raise ValueError('credential_pattern_requires_private_remediation')
    return report, original, state


def scans(rows, ids, output):
    """Retain all preselected outputs, including those with meaning issues."""
    by = {(r['example_id'], r['snapshot']): r for r in rows}
    if len(ids) != 8 or len(set(ids)) != 8: raise ValueError('eight_prefrozen_cases_required')
    groups = list(dict.fromkeys(by[rid, 'base']['style_group'] for rid in ids))
    grouped = {g: [rid for rid in ids if by[rid, 'base']['style_group'] == g] for g in groups}
    if len(groups) != 4 or any(len(x) != 2 for x in grouped.values()): raise ValueError('group_balance')
    items = []
    for group, cases in grouped.items():
        for day, rid in enumerate(cases, 1):
            for condition in ['base', 'epoch_1']:
                row = by[rid, condition]; text = row['output']; words = len(text.split())
                if runtime.sha(text) != row['output_sha256']: raise ValueError('scan_text_changed')
                items.append(dict(scan_id='SCAN_' + runtime.sha('simple-prompt-output-v1|' + rid + '|' + condition)[:16],
                                  round=day, example_id=rid, paper_id=row['paper_id'], style_group=group,
                                  condition=condition, text=text, text_sha256=runtime.sha(text),
                                  words_estimate=words, credits_estimate=math.ceil(words / 100),
                                  provider_minimum_met=words >= 50, assessment=row['assessment']))
    items.sort(key=lambda r: runtime.sha('simple-prompt-scan-order-v1|' + r['scan_id']))
    summaries = []
    for day in [1, 2]:
        chosen = [r for r in items if r['round'] == day]
        cost = sum(r['credits_estimate'] for r in chosen if r['provider_minimum_met'])
        if len(chosen) != 8 or cost > 20: raise ValueError('round_budget_exceeded_preserve_full_texts')
        summaries.append(dict(round=day, texts=8, estimated_credits=cost,
                              words_estimate=sum(r['words_estimate'] for r in chosen)))
    output.mkdir(); (output / 'texts').mkdir(); (output / 'pdf_reports').mkdir()
    for r in items: (output / 'texts' / (r['scan_id'] + '.txt')).write_text(r['text'])
    (output / 'PRIVATE_KEY.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in items))
    for day, summary in enumerate(summaries, 1):
        cards = [{k: r[k] for k in ['scan_id', 'text', 'words_estimate', 'credits_estimate', 'provider_minimum_met']}
                 for r in items if r['round'] == day]
        data = json.dumps(cards, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
        page = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Simpler prompt — Pangram</title>
<style>body{max-width:900px;margin:30px auto;padding:20px;font:17px system-ui}textarea{width:100%;height:330px;font:17px/1.6 Georgia}button,select{font:inherit;padding:8px;margin:8px}</style>
<h1>Simpler prompt — round __DAY__</h1><p>Eight matched base/adapter outputs. Estimated cost: __COST__ credits.
Use the same default Pangram settings. Copy each full paragraph separately, with no title or scan ID.
Do not edit or combine texts. Download each PDF into <b>pdf_reports/</b> beside this page; default filenames are fine. No CSV entry or renaming is needed.</p>
<p>Use round 2 after your ordinary daily credit refresh. Keep all reports, including unexpected results.
If a scan fails or your balance is insufficient, keep the text unchanged and report what remains.</p>
<button id="prev">Previous</button><select id="pick" aria-label="Item"></select><button id="next">Next</button>
<h2 id="name"></h2><p id="cost"></p><button id="copy">Copy exact text</button><textarea id="text" readonly></textarea><p id="msg"></p>
<script type="application/json" id="data">__DATA__</script><script>
const items=JSON.parse(document.getElementById('data').textContent),el=id=>document.getElementById(id);let at=0;
items.forEach((x,i)=>{let o=document.createElement('option');o.value=i;o.textContent=(i+1)+' / '+items.length+' — '+x.scan_id;el('pick').append(o)});
function show(){let x=items[at];el('pick').value=at;el('name').textContent=x.scan_id;el('text').value=x.text;el('cost').textContent=x.words_estimate+' estimated words; '+x.credits_estimate+' estimated credits';el('msg').textContent=x.provider_minimum_met?'':'Below minimum: retain unscanned; do not pad.'}
el('pick').onchange=()=>{at=Number(el('pick').value);show()};el('prev').onclick=()=>{at=Math.max(0,at-1);show()};el('next').onclick=()=>{at=Math.min(items.length-1,at+1);show()};
el('copy').onclick=async()=>{try{if(navigator.clipboard&&window.isSecureContext)await navigator.clipboard.writeText(items[at].text);else{el('text').focus();el('text').select();if(!document.execCommand('copy'))throw Error('copy')}el('msg').textContent='Exact text copied.'}catch(e){el('msg').textContent='Select the text and copy it manually.'}};show();
</script></html>'''
        (output / f'scan_round_{day}.html').write_text(page.replace('__DAY__', str(day)).replace('__COST__', str(summary['estimated_credits'])).replace('__DATA__', data))
    summary = dict(status='prepared_not_scanned', rounds=summaries, texts=16, cases=8,
                   selection_not_filtered_by_quality_or_scores=True, new_detector_calls=0,
                   actual_provider_word_count_may_differ=True,
                   below_minimum=sum(not r['provider_minimum_met'] for r in items))
    write(output / 'summary.json', summary)
    (output / 'START_HERE.md').write_text('# Replacement Pangram rounds\n\n' +
        '\n'.join(f"Round {r['round']}: {r['texts']} texts, approximately {r['estimated_credits']} credits."
                  for r in summaries) +
        '\n\nUse scan_round_1.html, then scan_round_2.html after the ordinary daily credit refresh. Save PDFs in pdf_reports/; default filenames are fine. No CSV or manual scores required. These replace the older strict-prompt rounds; no input/reference rescans.\n')
    write(output / 'manifest.json', dict(builder_sha256=sha(Path(__file__)),
          files={str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()}))
    return summary


def build(download, parent, output):
    if output.exists(): raise ValueError('refuse_overwrite_review')
    report, original, state = verify(download, parent)
    parent_manifest = read(parent / 'review_v1/manifest.json')
    for name, digest in parent_manifest['files'].items():
        if sha(parent / 'review_v1' / name) != digest: raise ValueError('previous_review_changed')
    judgments = [json.loads(line) for line in (download / 'semantic_assessments.private.jsonl').read_text().splitlines()]
    assessments = {(r['example_id'], r['snapshot']): r for r in judgments}
    if len(judgments) != 32 or len(assessments) != 32: raise ValueError('assessment_coverage')
    rows = []
    for snap in report['snapshots']:
        for generated in snap['generations']:
            assessment = assessments[generated['example_id'], snap['name']]
            if assessment['text_sha256'] != generated['output_sha256']: raise ValueError('judgment_binding')
            rows.append(dict(generated, snapshot=snap['name'], assessment=assessment,
                             source_metrics=metrics(generated['source'], generated['output']),
                             reference_metrics=metrics(generated['reference'], generated['output'])))
    summary = dict(protocol='corpus_200_simple_prompt_review_v1', slug='oliverlds/corpus-200-simple-prompt-4b',
                   version=1, execution_complete=True, optimizer_updates=0, elapsed_seconds=state['elapsed_seconds'],
                   peak_gpu_allocated_bytes=state['peak_gpu_allocated_bytes'], adapter_hashes_unchanged=True,
                   credential_pattern_matches=0, semantic_provenance=read(download / 'assessment_provenance.json'),
                   conditions={}, paired_prompt_effects={}, detector_results=None,
                   new_GPU_jobs=0, new_model_calls=0, new_detector_calls=0,
                   limits=['Descriptive surface overlap is not a style benchmark.',
                           'Same-agent unblinded automated semantic judgments are provisional.',
                           'The original adapter was trained with the stricter prompt.',
                           'Sixteen reused development examples; no reserved test use.',
                           'No meaning threshold is used to exclude outputs from detector evaluation.'])
    previous = [json.loads(line) for line in (parent / 'review_v1/outputs_and_assessments.private.jsonl').read_text().splitlines()]
    for name in ['base', 'epoch_1']:
        old = [r for r in previous if r['snapshot'] == name]
        new = [r for r in rows if r['snapshot'] == name]
        summary['conditions']['strict_' + name] = aggregate(old)
        summary['conditions']['simple_' + name] = aggregate(new)
        paired = [(a, b) for a, b in zip(old, new)]
        if any(a['example_id'] != b['example_id'] for a, b in paired): raise ValueError('paired_order_changed')
        summary['paired_prompt_effects'][name] = dict(
            lower_source_overlap=sum(b['source_metrics']['word_sequence_overlap'] < a['source_metrics']['word_sequence_overlap'] for a, b in paired),
            higher_source_overlap=sum(b['source_metrics']['word_sequence_overlap'] > a['source_metrics']['word_sequence_overlap'] for a, b in paired),
            identical_output=sum(a['output_sha256'] == b['output_sha256'] for a, b in paired))
    output.mkdir()
    (output / 'outputs_and_assessments.private.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    summary['replacement_Pangram'] = scans(rows, report['previous_detector_case_ids'], output / 'pangram_scans_v1')
    write(output / 'summary.json', summary)
    write(output / 'manifest.json', dict(builder_sha256=sha(Path(__file__)),
          download_manifest_sha256=sha(download / 'download_manifest.json'),
          preparation_sha256=sha(download / 'preparation.json'),
          previous_review_manifest_sha256=sha(parent / 'review_v1/manifest.json'),
          assessment_provenance_sha256=sha(download / 'assessment_provenance.json'),
          semantic_assessments_sha256=sha(download / 'semantic_assessments.private.jsonl'),
          files={str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()}))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['download', 'parent', 'output']: parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.download, args.parent, args.output), indent=2))
