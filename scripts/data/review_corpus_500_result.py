"""Verify the mixed-500 run, bind offline judgments and prepare matched scans."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from fama_style.training import corpus_500_pilot as runtime
from fama_style.research.neutral_reconstruction_results import metrics
from fama_style.research.teacher_pair_expanded_pilot_review import checkpoint_header


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def write(p,v):runtime.write_json(p,v)
def write_rows(p,v):p.write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in v))


def verify(download,packet):
    receipt=read(download/'download_manifest.json');raw=download/'raw';folder=raw/runtime.PROTOCOL
    if receipt['slug']!='oliverlds/corpus-500-pilot-4b' or receipt['version']!=1 or not receipt['version_explicitly_verified_via_SDK']:
        raise ValueError('exact_version_required')
    for name,h in receipt['files'].items():
        if sha(raw/name)!=h:raise ValueError('download_changed')
    splits,c=runtime.load_inputs(packet/'upload/corpus_500_pairs.jsonl',packet/'upload/corpus_500_config.json')
    report=read(folder/'evaluation.json');state=read(folder/'state.json')
    if report['config']!=c or not report['complete'] or not state['complete'] or not report['training_called'] or not state['training_called']:
        raise ValueError('incomplete_run_or_config_changed')
    if any(x['optimizer_steps']!=250 or x['training_examples_seen']!=500 for x in [report,state]):raise ValueError('training_budget_changed')
    if not report['adapter_weights_changed'] or not report['adapter_reload_weights_match']:raise ValueError('adapter_checks')
    if sha(raw/'corpus_500_pilot.py')!=c['runtime_source_sha256'] or report['runtime']['runtime_source_sha256']!=c['runtime_source_sha256']:raise ValueError('runtime_changed')
    for k,v in [('model_id',runtime.MODEL_ID),('model_revision',runtime.REVISION),('gpu_count',1)]:
        if report['runtime'][k]!=v:raise ValueError('runtime_contract')
    if 'T4' not in report['runtime']['gpu'] or any(report['runtime']['packages'][k]!=v for k,v in runtime.PACKAGES.items()):raise ValueError('GPU_or_packages_changed')
    updates=read(folder/'training_log.json');grads=read(folder/'gradient_checks.json')
    if len(updates)!=250 or len(grads)!=250:raise ValueError('training_log_coverage')
    order=list(range(500));random.Random(runtime.HYPER['seed']).shuffle(order)
    expected=[c['training_ids'][i] for i in order];actual=[i for x in updates for i in x['example_ids']]
    if actual!=expected or actual!=report['training_order_ids'] or len(set(actual))!=500:raise ValueError('one_epoch_training_order')
    for i,(u,g) in enumerate(zip(updates,grads),1):
        if u['step']!=i or g['step']!=i or not g['finite'] or not math.isfinite(g['norm']) or g['norm']<=0 or not math.isfinite(u['loss']) or u['learning_rate']!=runtime.learning_rate(i):raise ValueError('update_or_gradient')
    labels=read(folder/'label_audit.json')
    if [r['example_id'] for r in labels]!=c['training_ids']+c['development_ids']:raise ValueError('label_coverage')
    if any(not a['target_roundtrip_matches'] or a['truncated'] or not 0<a['prompt_tokens']<a['full_tokens']<=1536 or a['supervised_tokens']!=a['full_tokens']-a['prompt_tokens'] for a in labels):raise ValueError('label_or_length')
    sanity=report['likelihood_sanity']
    if not sanity['passed'] or not math.isclose(sanity['manual_mean_nll'],sanity['builtin_mean_nll'],rel_tol=1e-5,abs_tol=1e-5):raise ValueError('likelihood_sanity')
    for name,h in report['adapter_files_sha256'].items():
        if sha(folder/'epoch_1/adapter'/name)!=h:raise ValueError('adapter_changed')
    header=checkpoint_header(folder/'epoch_1/adapter')
    if header['tensor_count']!=136 or header['parameters']!=2228224 or header['dtypes']!=['F32']:raise ValueError('adapter_layout')
    by={r['example_id']:r for group in splits.values() for r in group};outputs=[]
    if [s['name'] for s in report['snapshots']]!=['base','epoch_1']:raise ValueError('snapshot_names')
    for snap,step in zip(report['snapshots'],[0,250]):
        if not snap['complete'] or snap['optimizer_step']!=step:raise ValueError('snapshot_boundary')
        for field,ids in [('development_losses',c['development_ids']),('training_probe_losses',c['training_probe_ids'])]:
            values=snap[field]
            if len(values)!=len(ids) or {r['example_id'] for r in values}!=set(ids) or any(not math.isfinite(r['loss']) or r['supervised_tokens']<=0 for r in values):raise ValueError('loss_coverage')
            value=sum(r['loss']*r['supervised_tokens'] for r in values)/sum(r['supervised_tokens'] for r in values)
            if not math.isclose(value,snap['development_token_weighted_nll' if field=='development_losses' else 'training_token_weighted_nll'],abs_tol=1e-12):raise ValueError('loss_aggregation')
        if [r['example_id'] for r in snap['generations']]!=c['development_ids']:raise ValueError('generation_coverage')
        for g in snap['generations']:
            original=by[g['example_id']]
            if any(g[k]!=original[k] for k in ['source','paper_id','style_group','content_id','input_condition']) or g['reference']!=original['target'] or g['prompt_sha256']!=runtime.sha(original['prompt']) or g['output_sha256']!=runtime.sha(g['output']):raise ValueError('generation_binding')
            outputs.append(dict(g,snapshot=snap['name'],source_metrics=metrics(g['source'],g['output']),reference_metrics=metrics(g['reference'],g['output'])))
    logs=list(raw.glob('*.log'))
    if len(logs)!=1:raise ValueError('log_coverage')
    log=logs[0].read_text()
    if any(re.search(p,log) for p in [r'hf_[A-Za-z0-9]{20,}',r'KGAT_[A-Za-z0-9_\-]{15,}',r'AKIA[0-9A-Z]{16}']):raise ValueError('credential_pattern_requires_private_remediation')
    return report,state,outputs,header


def aggregate(items):
    result=dict(outputs=len(items),empty=sum(r['empty'] for r in items),output_limited=sum(r['output_limited'] for r in items),
        median_source_overlap=statistics.median(r['source_metrics']['word_sequence_overlap'] for r in items),
        median_reference_overlap=statistics.median(r['reference_metrics']['word_sequence_overlap'] for r in items),
        median_longest_source_run=statistics.median(r['source_metrics']['longest_shared_word_run'] for r in items),
        exact_word_copies=sum(r['source_metrics']['word_token_copy'] for r in items),
        source_overlap_at_least_point8=sum(r['source_metrics']['word_sequence_overlap']>=.8 for r in items))
    if all('assessment' in r for r in items):
        result['material_claim_issues_identified']=sum(r['assessment']['material_claim_change'] for r in items)
        if all('minor_notes' in r['assessment'] for r in items):
            result['outputs_with_minor_shift_notes']=sum(bool(r['assessment']['minor_notes']) for r in items)
        if all('readable' in r['assessment'] for r in items):
            result['readable']=sum(r['assessment']['readable'] for r in items)
    return result


def prepare(download,packet):
    report,state,items,header=verify(download,packet)
    path=download/'masked_bundle.private.jsonl'
    if path.exists():raise ValueError('refuse_overwrite_masked_review')
    by={(r['example_id'],r['snapshot']):r for r in items};bundle=[];key=[]
    for i,rid in enumerate(report['config']['development_ids'],1):
        base=by[rid,'base'];names=['base','epoch_1']
        random.Random(runtime.sha('mixed500-review-v1|'+rid)).shuffle(names)
        b=dict(case=i,example_id=rid,input_condition=base['input_condition'],source=base['source'],reference=base['reference'],outputs={})
        for label,name in zip(['A','B'],names):
            r=by[rid,name];b['outputs'][label]=r['output'];key.append(dict(case=i,example_id=rid,label=label,snapshot=name,text_sha256=r['output_sha256']))
        bundle.append(b)
    write_rows(path,bundle);write_rows(download/'masked_key.private.jsonl',key)
    write_rows(download/'outputs_metrics.private.jsonl',items)
    write(download/'execution_checks.json',dict(complete=True,training_examples=500,updates=250,label_checks=524,
        adapter=header,credential_pattern_matches=0,elapsed_seconds=state['elapsed_seconds'],peak_GPU_bytes=state['peak_gpu_allocated_bytes'],
        conditions={condition:{s:aggregate([r for r in items if r['input_condition']==condition and r['snapshot']==s]) for s in ['base','epoch_1']} for condition in ['existing_input','gemini_input']}))


def scan_packet(items,ids,output):
    if output.exists():raise ValueError('refuse_overwrite_scans')
    if len(ids)!=2 or len(set(ids))!=2:raise ValueError('two_prefrozen_priority_cases_required')
    by={(r['example_id'],r['snapshot']):r for r in items};cards=[]
    for rid in ids:
        base=by[rid,'base'];trained=by[rid,'epoch_1']
        if base['input_condition']!='gemini_input' or any(base[k]!=trained[k] for k in ['source','reference','content_id','input_condition']):raise ValueError('matched_Gemini_scan_case')
        for condition,text in [('input',base['source']),('base',base['output']),('epoch_1',trained['output'])]:
            n=len(text.split());h=runtime.sha(text)
            cards.append(dict(scan_id='SCAN_'+runtime.sha('mixed500-pangram-v1|'+rid+'|'+condition)[:16],
                example_id=rid,condition=condition,text=text,text_sha256=h,words_estimate=n,
                credits_estimate=math.ceil(n/100),provider_minimum_met=n>=50))
    cards.sort(key=lambda r:runtime.sha('mixed500-scan-order|'+r['scan_id']))
    output.mkdir();(output/'texts').mkdir();(output/'pdf_reports').mkdir()
    for r in cards:(output/'texts'/(r['scan_id']+'.txt')).write_text(r['text'])
    write_rows(output/'PRIVATE_KEY.jsonl',cards)
    data=json.dumps([{k:r[k] for k in ['scan_id','text','words_estimate','credits_estimate']} for r in cards],ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Mixed-500 Pangram scans</title>
<style>body{font:17px system-ui;max-width:900px;margin:30px auto;padding:20px}textarea{width:100%;height:380px;font:17px/1.6 Georgia}button,select{font:inherit;padding:8px;margin:8px}</style>
<h1>Mixed-500: six matched texts</h1><p>Two preselected Gemini-input cases, each with input, base output and fine-tuned output.
Estimated total: __COST__ credits. Use your usual default Pangram settings. Scan each full text separately.
Do not include its ID in the scan or change the text. Preserve mathematical markup exactly as shown.
Save all six exported PDFs in the <b>pdf_reports</b> folder beside this page. Default filenames are fine; no CSV or manual scores required.</p>
<p>Keep every result, including human labels and outputs with meaning errors. If credits run out, continue after the normal daily refresh.</p>
<button id="prev">Previous</button><select id="pick" aria-label="Text"></select><button id="next">Next</button>
<h2 id="name"></h2><p id="cost"></p><button id="copy">Copy exact text</button><p id="message"></p><textarea id="text" readonly aria-label="Exact scan text"></textarea>
<script>const cards=__DATA__;const pick=document.getElementById('pick');
cards.forEach((c,i)=>{const o=document.createElement('option');o.value=i;o.textContent=(i+1)+' / '+cards.length+' — '+c.scan_id;pick.appendChild(o)});
function show(){const c=cards[+pick.value];document.getElementById('name').textContent=c.scan_id;document.getElementById('text').value=c.text;document.getElementById('cost').textContent=c.words_estimate+' words; estimated '+c.credits_estimate+' credits';document.getElementById('message').textContent=''}
pick.onchange=show;document.getElementById('prev').onclick=()=>{pick.value=Math.max(0,+pick.value-1);show()};document.getElementById('next').onclick=()=>{pick.value=Math.min(cards.length-1,+pick.value+1);show()};
document.getElementById('copy').onclick=async()=>{const t=document.getElementById('text');try{await navigator.clipboard.writeText(t.value)}catch(e){t.select();document.execCommand('copy')}document.getElementById('message').textContent='Copied exact text.'};show();</script></html>'''
    cost=sum(r['credits_estimate'] for r in cards)
    (output/'scan_round_1.html').write_text(page.replace('__DATA__',data).replace('__COST__',str(cost)))
    summary=dict(cases=2,texts=6,estimated_credits=cost,actual_provider_count_may_differ=True,selection='Frozen before generation; not filtered by semantic judgments or detector scores',detector_calls=0)
    write(output/'summary.json',summary)
    (output/'START_HERE.md').write_text('Open scan_round_1.html. Scan six full texts with the usual Pangram settings and save the six PDFs in pdf_reports/. No manual score entry or renaming required. Estimated cost: '+str(cost)+' credits. Preserve all results.\n')
    write(output/'manifest.json',dict(builder_sha256=sha(Path(__file__)),files={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()}))
    return summary


def finalize(download,packet):
    output=download/'review_v1'
    if output.exists():raise ValueError('refuse_overwrite_review')
    report,state,items,header=verify(download,packet)
    assessments=rows(download/'semantic_assessments.private.jsonl');by={(r['example_id'],r['snapshot']):r for r in assessments}
    if len(assessments)!=48 or len(by)!=48:raise ValueError('assessment_coverage')
    for r in items:
        a=by[r['example_id'],r['snapshot']]
        if a['text_sha256']!=r['output_sha256']:raise ValueError('assessment_text_changed')
        r['assessment']=a
    summary=dict(protocol='corpus_500_pilot_review_v1',slug='oliverlds/corpus-500-pilot-4b',version=1,
        execution_complete=True,training_pairs=500,optimizer_updates=250,label_checks=524,adapter=header,
        elapsed_seconds=state['elapsed_seconds'],peak_GPU_bytes=state['peak_gpu_allocated_bytes'],
        credential_pattern_matches=0,assessment_provenance=read(download/'assessment_provenance.json'),
        conditions={condition:{s:aggregate([r for r in items if r['input_condition']==condition and r['snapshot']==s]) for s in ['base','epoch_1']} for condition in ['existing_input','gemini_input']},
        target_nll_by_input_condition={s['name']:s['development_nll_by_input_condition'] for s in report['snapshots']},
        detector_results=None,new_GPU_jobs=0,new_model_calls=0,new_detector_calls=0,
        limits=['Provisional same-agent partially masked semantic judgments; not independent or calibrated.',
            'Twenty-four input variants share sixteen references; eight reused development papers, no sealed test.',
            'Reference overlap and lower NLL do not establish distinctive Fama-style improvement.',
            'One development reference has an inherited five-years versus t-1 through t-6 inconsistency.',
            'Some outputs emit literal HTML subscripts; exact detector texts retain that formatting.'])
    parent=ROOT/'research_runs/data_scaling_v1/corpus_200_simple_prompt_v1'
    old=read(parent/'raw/corpus_200_simple_prompt_v1/evaluation.json')
    old_rows=rows(parent/'review_v1/outputs_and_assessments.private.jsonl')
    current={(r['example_id'],r['snapshot']):r for r in items if r['input_condition']=='existing_input'}
    old_base=old['snapshots'][0]['generations']
    if len(old_base)!=16 or any(current[r['example_id'],'base']['source']!=r['source'] or current[r['example_id'],'base']['reference']!=r['reference'] for r in old_base):raise ValueError('parent_comparison_input_changed')
    adapter200=[dict(r,source_metrics=metrics(r['source'],r['output']),reference_metrics=metrics(r['reference'],r['output'])) for r in old_rows if r['snapshot']=='epoch_1']
    pairs=[(r,current[r['example_id'],'epoch_1']) for r in adapter200]
    summary['comparison_to_200_simple_prompt']=dict(same_inputs=16,base_outputs_byte_identical=sum(r['output_sha256']==current[r['example_id'],'base']['output_sha256'] for r in old_base),
        adapter_200=aggregate(adapter200),adapter_500=summary['conditions']['existing_input']['epoch_1'],
        higher_source_overlap=sum(b['source_metrics']['word_sequence_overlap']>a['source_metrics']['word_sequence_overlap'] for a,b in pairs),
        higher_reference_overlap=sum(b['reference_metrics']['word_sequence_overlap']>a['reference_metrics']['word_sequence_overlap'] for a,b in pairs),
        semantic_count_comparison_limit='200 judgments reused from the prior review, not newly harmonized. Identical base outputs were flagged 10/16 previously versus 11/16 now; count differences demonstrate judgment instability.')
    output.mkdir();write_rows(output/'outputs_and_assessments.private.jsonl',items)
    summary['Pangram_preparation']=scan_packet(items,report['config']['detector_priority_ids'],output/'pangram_scans_v1')
    write(output/'summary.json',summary)
    write(output/'manifest.json',dict(code_sha256=sha(Path(__file__)),download_manifest_sha256=sha(download/'download_manifest.json'),
        semantic_assessments_sha256=sha(download/'semantic_assessments.private.jsonl'),assessment_provenance_sha256=sha(download/'assessment_provenance.json'),
        files={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()}))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--download',type=Path,default=ROOT/'research_runs/data_scaling_v1/corpus_500_pilot_4b_v1');p.add_argument('--packet',type=Path,default=ROOT/'data/corpus_500_pilot_v1');p.add_argument('--finalize',action='store_true');a=p.parse_args()
    if a.finalize:
        result=finalize(a.download,a.packet);print(json.dumps({k:result[k] for k in ['conditions','comparison_to_200_simple_prompt','Pangram_preparation']},indent=2))
    else:
        prepare(a.download,a.packet);print(json.dumps(read(a.download/'execution_checks.json'),indent=2))
