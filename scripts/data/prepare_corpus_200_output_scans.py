"""Freeze matched base/adapter Pangram scans from the completed pilot outputs."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def sha(text):return hashlib.sha256(text.encode()).hexdigest()
def file_sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def build(download,packet,output):
    if output.exists():raise ValueError('refuse_overwrite_frozen_output_scans')
    review=download/'review_v1';manifest=json.loads((review/'manifest.json').read_text())
    for name,h in manifest['files'].items():
        if file_sha(review/name)!=h:raise ValueError('review_changed')
    config=json.loads((packet/'upload/corpus_200_config.json').read_text())
    executed=json.loads((download/'raw/corpus_200_pilot_v1/evaluation.json').read_text())['config']
    if config!=executed:raise ValueError('prefrozen_detector_case_config_changed')
    reviewed=[json.loads(l) for l in (review/'outputs_and_assessments.private.jsonl').read_text().splitlines()]
    expected=config['detector_case_ids'];items=[]
    if len(expected)!=8 or len(set(expected))!=8:raise ValueError('eight_prefrozen_cases_required')
    by={(r['example_id'],r['snapshot']):r for r in reviewed}
    groups=[]
    for rid in expected:
        group=by[rid,'base']['style_group']
        if group not in groups:groups.append(group)
    by_group={g:[rid for rid in expected if by[rid,'base']['style_group']==g] for g in groups}
    if len(groups)!=4 or any(len(ids)!=2 for ids in by_group.values()):raise ValueError('two_cases_per_group_required')
    for group,ids in by_group.items():
        for day,rid in enumerate(ids,1):
            for condition in ['base','epoch_1']:
                r=by[rid,condition];text=r['output']
                if sha(text)!=r['output_sha256']:raise ValueError('output_changed')
                words=len(text.split())
                if words<50:raise ValueError('below_provider_minimum_preserve_and_report_not_pad')
                items.append({'scan_id':'SCAN_'+sha('corpus200-output-v1|'+rid+'|'+condition)[:16],
                    'round':day,'example_id':rid,'paper_id':r['paper_id'],'style_group':group,
                    'condition':condition,'text':text,'text_sha256':sha(text),'words_estimate':words,
                    'credits_estimate':math.ceil(words/100),'assessment':r['assessment']})
    output.mkdir(parents=True);(output/'texts').mkdir();(output/'pdf_reports').mkdir()
    items.sort(key=lambda r:sha('corpus200-output-blind-order-v1|'+r['scan_id']))
    for r in items:(output/'texts'/f"{r['scan_id']}.txt").write_text(r['text'])
    (output/'PRIVATE_KEY.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items))
    summaries=[]
    for day in [1,2]:
        selected=[r for r in items if r['round']==day];credits=sum(r['credits_estimate'] for r in selected)
        if len(selected)!=8 or credits>20 or Counter(r['condition'] for r in selected)!=Counter({'base':4,'epoch_1':4}):
            raise ValueError('daily_balanced_budget_failed')
        cards=[{k:r[k] for k in ['scan_id','text','words_estimate','credits_estimate']} for r in selected]
        data=json.dumps(cards,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
        page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>200-pair model outputs — Pangram round __DAY__</title>
<style>body{max-width:900px;margin:30px auto;padding:20px;font:17px system-ui}textarea{width:100%;height:330px;font:17px/1.6 Georgia}button,select{font:inherit;padding:8px;margin:8px}small{color:#465465}</style>
<h1>200-pair model outputs — round __DAY__</h1>
<p>Eight new model outputs: matched base and trained-model text from four held-out cases.
This does not repeat scans of dataset inputs or human references. Estimated cost: <b>__CREDITS__ credits</b>.
Use the same default AI-detection settings for every item and check your remaining balance first.</p>
<p>Copy each full text into Pangram separately. Include no title or scan ID. Keep the text unchanged; do not combine, trim, correct or regenerate it.
Download every PDF report into the <b>pdf_reports</b> folder beside this page. Default filenames are fine; no CSV fields or scores need to be typed.</p>
<p>Retain all results, including unexpected ones. If a scan fails, leave it unscanned and report that limitation.
Round 2 should wait for the ordinary daily credit refresh. Nothing on this local page calls a detector or contains credentials.</p>
<p><a href="https://www.pangram.com/">Open Pangram</a></p>
<button id="prev">Previous</button><select id="pick" aria-label="Scan item"></select><button id="next">Next</button>
<h2 id="name"></h2><small id="cost"></small><p><button id="copy">Copy exact text</button></p>
<textarea id="text" readonly aria-label="Exact scan text"></textarea><p id="msg" aria-live="polite"></p>
<script type="application/json" id="data">__DATA__</script><script>
const items=JSON.parse(document.getElementById('data').textContent),el=id=>document.getElementById(id);let at=0;
items.forEach((x,i)=>{const o=document.createElement('option');o.value=i;o.textContent=(i+1)+' / '+items.length+' — '+x.scan_id;el('pick').append(o)});
function show(){const x=items[at];el('pick').value=at;el('name').textContent=x.scan_id;el('text').value=x.text;el('cost').textContent=x.words_estimate+' estimated words; '+x.credits_estimate+' credits';el('msg').textContent=''}
el('pick').onchange=()=>{at=Number(el('pick').value);show()};el('prev').onclick=()=>{at=Math.max(0,at-1);show()};el('next').onclick=()=>{at=Math.min(items.length-1,at+1);show()};
el('copy').onclick=async()=>{try{if(navigator.clipboard&&window.isSecureContext){await navigator.clipboard.writeText(items[at].text)}else{el('text').focus();el('text').select();if(!document.execCommand('copy'))throw Error('copy')}el('msg').textContent='Exact text copied.'}catch(e){el('msg').textContent='Select the text and copy it manually.'}};show();
</script></html>'''.replace('__DAY__',str(day)).replace('__CREDITS__',str(credits)).replace('__DATA__',data)
        (output/f'scan_round_{day}.html').write_text(page)
        summaries.append({'round':day,'texts':8,'cases':4,'estimated_credits':credits,
                          'words_estimate':sum(r['words_estimate'] for r in selected)})
    summary={'protocol':'corpus_200_model_output_scans_v1','status':'prepared_not_scanned','rounds':summaries,
        'texts':16,'cases':8,'estimated_credits':sum(r['credits_estimate'] for r in items),
        'selection':'Exact eight development cases frozen before generation; four matched cases per day, one per group.',
        'known_quality_flags_in_trained_selected_outputs':{'material_claim_change':sum(r['assessment']['material_claim_change'] for r in items if r['condition']=='epoch_1'),
            'multiple_minor_shifts':sum(r['assessment']['minor_semantic_shifts']>1 for r in items if r['condition']=='epoch_1')},
        'selection_not_filtered_by_quality_or_scores':True,'dataset_inputs_or_human_references_rescanned':False,
        'new_detector_calls':0,'no_manual_CSV_required':True,
        'limits':['Detector changes must be interpreted jointly with meaning and copying; no claim of learned author style.',
                  'One model output is shorter than its source because of substantive omissions; do not silently exclude it.',
                  'Two ordinary quota periods are needed at the published 20-credit daily limit; actual word counting/balance may differ.'],
        'provider_minimum_source':'https://pangram.cv/knowledge-hub/minimum-and-maximum-input-sizes',
        'provider_credit_source':'https://www.pangram.com/knowledge-hub/what-is-an-ai-detection-credit',
        'policy_checked':'2026-11-04'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'START_HERE.md').write_text('# Scan the new model outputs\n\nOpen scan_round_1.html: eight texts, about 15 credits. After the normal daily refresh, use scan_round_2.html: eight texts, about 14 credits.\n\nSave every Pangram PDF in pdf_reports/. Default filenames are fine. No scores or CSV fields need to be entered. These are new base-versus-trained Gemma outputs, not repeated dataset scans.\n\nThe cases were selected before generation. Known semantic errors remain included and must be reported with detector results.\n')
    (output/'manifest.json').write_text(json.dumps({'review_manifest_sha256':file_sha(review/'manifest.json'),
        'config_sha256':file_sha(packet/'upload/corpus_200_config.json'),'builder_sha256':file_sha(Path(__file__)),
        'files':{str(p.relative_to(output)):file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}},indent=2)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['download','packet','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(build(a.download,a.packet,a.output),indent=2))
