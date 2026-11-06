"""Prepare only the unscanned remainder of the frozen eight-case Gemini panel."""
import argparse
import importlib.util
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location('review500',Path(__file__).with_name('review_corpus_500_result.py'))
r=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(r)


def verify_files(folder):
    for name,h in r.read(folder/'manifest.json')['files'].items():
        if r.sha(folder/name)!=h:raise ValueError('frozen_evidence_changed')


def select(config,outputs,previous):
    ids=config['detector_case_ids'];priority=config['detector_priority_ids']
    if len(ids)!=8 or len(set(ids))!=8 or len(priority)!=2 or not set(priority)<=set(ids):
        raise ValueError('frozen_eight_cases_required')
    observed={(x['example_id'],x['condition']) for x in previous}
    if len(previous)!=6 or observed!={(rid,c) for rid in priority for c in ['input','base','epoch_1']}:
        raise ValueError('completed_initial_triplets_required')
    pairs={(x['example_id'],x['snapshot']):x for x in outputs}
    if len(pairs)!=len(outputs):raise ValueError('duplicate_output')
    remaining=[rid for rid in ids if rid not in priority];cards=[]
    for rid in remaining:
        base=pairs[rid,'base'];trained=pairs[rid,'epoch_1']
        if base['input_condition']!='gemini_input' or any(base[k]!=trained[k] for k in ['source','reference','content_id','input_condition','style_group','paper_id']):
            raise ValueError('matched_Gemini_inputs_required')
        for condition,text in [('input',base['source']),('base',base['output']),('epoch_1',trained['output'])]:
            if condition!='input' and r.runtime.sha(text)!=pairs[rid,condition]['output_sha256']:
                raise ValueError('output_text_changed')
            n=len(text.split())
            cards.append(dict(scan_id='SCAN_'+r.runtime.sha('mixed500-pangram-v1|'+rid+'|'+condition)[:16],
                example_id=rid,condition=condition,style_group=base['style_group'],paper_id=base['paper_id'],
                text=text,text_sha256=r.runtime.sha(text),words_estimate=n,credits_estimate=math.ceil(n/100),
                provider_minimum_met=n>=50,round=2+remaining.index(rid)//3))
    if len(cards)!=18 or len({x['scan_id'] for x in cards})!=18:raise ValueError('eighteen_text_coverage')
    if any(not x['provider_minimum_met'] for x in cards):raise ValueError('short_text_do_not_pad_or_omit')
    # Three intact input/base/adapter triplets per round. No quality filtering.
    for number in [2,3]:
        group=[x for x in cards if x['round']==number]
        if len(group)!=9 or sum(x['credits_estimate'] for x in group)>20:
            raise ValueError('daily_budget_requires_new_plan_do_not_truncate')
    return cards


def render(cards,number,cost):
    data=json.dumps([{k:x[k] for k in ['scan_id','text','words_estimate','credits_estimate']} for x in cards],ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Mixed-500 remaining scans</title>
<style>body{max-width:900px;margin:30px auto;padding:20px;font:17px system-ui}textarea{width:100%;height:380px;font:17px/1.6 Georgia}button,select{font:inherit;padding:8px;margin:8px}</style>
<h1>Mixed-500 — round __ROUND__</h1><p>Nine full texts from three preselected cases: each input, base output and fine-tuned output. Estimated cost: __COST__ credits.</p>
<p>Use your usual default Pangram settings. Copy one full text per scan, without its ID or title. Do not edit, shorten, combine or remove mathematical markup.
Save every exported PDF in the <b>pdf_reports</b> folder beside this page. Default filenames are fine; no CSV or manual scores required.</p>
<p>Start when you have enough ordinary daily credits. If credits run out, keep the exact texts and continue after the normal refresh. Round 3 can follow on the next day.
Keep all results, including human labels and texts with meaning errors. These eighteen texts exclude the six already scanned.</p>
<button id="prev">Previous</button><select id="pick" aria-label="Text"></select><button id="next">Next</button>
<h2 id="name"></h2><p id="cost"></p><button id="copy">Copy exact text</button><p id="message"></p><textarea id="text" readonly aria-label="Exact scan text"></textarea>
<script>const cards=__DATA__;const pick=document.getElementById('pick');
cards.forEach((c,i)=>{let o=document.createElement('option');o.value=i;o.textContent=(i+1)+' / '+cards.length+' — '+c.scan_id;pick.appendChild(o)});
function show(){let c=cards[+pick.value];document.getElementById('name').textContent=c.scan_id;document.getElementById('text').value=c.text;document.getElementById('cost').textContent=c.words_estimate+' words; estimated '+c.credits_estimate+' credits';document.getElementById('message').textContent=''}
pick.onchange=show;document.getElementById('prev').onclick=()=>{pick.value=Math.max(0,+pick.value-1);show()};document.getElementById('next').onclick=()=>{pick.value=Math.min(cards.length-1,+pick.value+1);show()};
document.getElementById('copy').onclick=async()=>{let t=document.getElementById('text');try{await navigator.clipboard.writeText(t.value)}catch(e){t.select();document.execCommand('copy')}document.getElementById('message').textContent='Copied exact text.'};show();</script></html>'''.replace('__DATA__',data).replace('__ROUND__',str(number)).replace('__COST__',str(cost))


def build(run,packet,output):
    if output.exists():raise ValueError('refuse_overwrite_remaining_packet')
    verify_files(run/'review_v1');verify_files(run/'pangram_import_v1')
    report,_,_,_=r.verify(run,packet)
    outputs=r.rows(run/'review_v1/outputs_and_assessments.private.jsonl')
    previous=r.rows(run/'pangram_import_v1/observations.private.jsonl')
    cards=select(report['config'],outputs,previous)
    if {x['scan_id'] for x in cards}&{x['scan_id'] for x in previous}:raise ValueError('completed_scan_repeated')
    output.mkdir();(output/'texts').mkdir();(output/'pdf_reports').mkdir()
    for x in cards:(output/'texts'/(x['scan_id']+'.txt')).write_text(x['text'])
    r.write_rows(output/'PRIVATE_KEY.jsonl',cards)
    rounds=[]
    for number in [2,3]:
        group=sorted([x for x in cards if x['round']==number],key=lambda x:r.runtime.sha('mixed500-scan-order|'+x['scan_id']))
        cost=sum(x['credits_estimate'] for x in group)
        (output/f'scan_round_{number}.html').write_text(render(group,number,cost))
        rounds.append(dict(round=number,cases=3,texts=9,estimated_credits=cost))
    summary=dict(protocol='corpus_500_remaining_pangram_v1',status='prepared_not_scanned',remaining_cases=6,texts=18,
        estimated_credits=sum(x['credits_estimate'] for x in cards),rounds=rounds,
        previous_cases=2,previous_reports=6,complete_panel_cases=8,complete_panel_reports=24,
        source_selection='Original eight detector IDs minus the two completed priority cases; no post-score selection',
        new_detector_calls=0,new_model_calls=0,new_GPU_jobs=0,actual_provider_counts_may_differ=True)
    r.write(output/'summary.json',summary)
    (output/'START_HERE.md').write_text('Remaining mixed-500 Pangram scans\n\nRound 2: scan_round_2.html — 9 texts, approximately '+str(rounds[0]['estimated_credits'])+' credits.\nRound 3: scan_round_3.html — 9 texts, approximately '+str(rounds[1]['estimated_credits'])+' credits.\n\nUse normal daily credit refreshes. Save all 18 PDFs in pdf_reports/ beside these pages. Default filenames are fine; no CSV or manual scores. Do not rescan the previous six texts. No further training starts while this evaluation is pending.\n')
    (output/'start_here.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Remaining mixed-500 scans</title><body style="font:18px system-ui;max-width:800px;margin:40px auto"><h1>Remaining 18 Pangram scans</h1><p>The previous six reports are already imported. Scan these two rounds using normal daily credits; save all PDFs in the pdf_reports folder beside this page.</p><p><a href="scan_round_2.html">Round 2: nine texts, estimated '+str(rounds[0]['estimated_credits'])+' credits</a></p><p><a href="scan_round_3.html">Round 3: nine texts, estimated '+str(rounds[1]['estimated_credits'])+' credits</a></p><p>Default PDF filenames are fine. No manual score entry. No further training has started.</p></body></html>')
    r.write(output/'manifest.json',dict(builder_sha256=r.sha(Path(__file__)),
        parent_review_manifest_sha256=r.sha(run/'review_v1/manifest.json'),
        previous_import_manifest_sha256=r.sha(run/'pangram_import_v1/manifest.json'),
        frozen_config_sha256=r.sha(packet/'upload/corpus_500_config.json'),
        files={str(p.relative_to(output)):r.sha(p) for p in sorted(output.rglob('*')) if p.is_file()}))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=ROOT/'research_runs/data_scaling_v1/corpus_500_pilot_4b_v1');p.add_argument('--packet',type=Path,default=ROOT/'data/corpus_500_pilot_v1');a=p.parse_args()
    print(json.dumps(build(a.run,a.packet,a.run/'review_v1/pangram_remaining_v1'),indent=2))
